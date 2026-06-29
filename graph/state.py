"""
state.py — the shared state schema for the LangGraph orchestrator.

Every node in graph/workflow.py reads from and writes to this single
state object, which is why it's defined first and on its own: it's the
contract that lets eight independently-built agents pass data down a
pipeline without each one knowing the others' internals.

Two distinct types live here, on purpose:

  1. BugFixRequest (Pydantic) — the INPUT boundary. This is what the
     Streamlit UI (or any caller) hands in. Pydantic validates it once,
     up front, so a malformed request fails loudly at the edge instead
     of halfway through the pipeline with a cryptic KeyError. This is
     the "Pydantic model for the LangGraph state object" we said we'd
     lock in before building the orchestrator.

  2. GraphState (TypedDict) — the INTERNAL graph state. LangGraph nodes
     return *partial* dict updates that get merged into this, so a plain
     TypedDict is the idiomatic, friction-free choice here (a frozen
     Pydantic model would fight the merge-on-every-node pattern). Every
     key is pre-populated by initial_state() so nodes can read any field
     without defensive .get() calls everywhere.

The `history` field uses an `add` reducer so each node *appends* a log
entry rather than overwriting — that running log is what the
explainability view in Streamlit renders later (investigated_files →
root_cause → reasoning → validation → outcome).
"""

from __future__ import annotations

import operator
import subprocess
from typing import Annotated, Any, Optional, TypedDict

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------
# Input boundary — validated once, up front.
# --------------------------------------------------------------------------
class BugFixRequest(BaseModel):
    """
    Everything the pipeline needs to run, validated at the boundary.

    Only `bug_report` and `repo_path` are truly required for a local run;
    the rest have sensible defaults or are derived from the repo itself in
    initial_state().
    """

    bug_report: str = Field(..., description="Plain-English description of the bug.")
    repo_path: str = Field(..., description="Local filesystem path to the cloned repo.")

    # Identity for the RAG cache. Defaults to the repo_path so repeated
    # runs against the same local repo reuse the embedded collection
    # instead of re-embedding (which burns Gemini quota).
    repo_id: Optional[str] = None
    # The commit the repo is checked out at — part of the RAG cache key.
    # Derived via `git rev-parse HEAD` in initial_state() if not given.
    commit_hash: Optional[str] = None

    # PR-related. repo_full_name ("owner/repo") and create_pr are only
    # needed if you want the orchestrator to actually open a GitHub PR.
    # For a local sample repo with no remote, leave create_pr False — the
    # pipeline stops at "pr_ready" with the description prepared.
    repo_full_name: Optional[str] = None
    base_branch: Optional[str] = None  # derived from the repo's current branch if None
    create_pr: bool = False
    # Open the PR as a draft. Recommended for AI-assisted contributions to
    # repos you don't own, so a human reviews before it's marked ready.
    draft_pr: bool = False

    # Optional explicit seed files for the call-graph trace (e.g. parsed
    # from a stack trace). If None, the navigate node derives seeds from
    # the Issue Agent's guesses + RAG hits.
    seed_files: Optional[list[str]] = None

    # Decision knobs, defaulted to the values we settled on in planning.
    confidence_threshold: float = 0.7  # auto-PR only at/above this confidence
    max_retries: int = 3               # patch attempts before escalating to human review
    max_rag_results: int = 5           # cap retrieved chunks (token-budget guard)

    @field_validator("bug_report", "repo_path")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be empty")
        return v.strip()


# --------------------------------------------------------------------------
# Internal graph state — what flows between nodes.
# --------------------------------------------------------------------------
class GraphState(TypedDict, total=False):
    # ---- inputs (copied from BugFixRequest, set once at start) ----
    bug_report: str
    repo_path: str
    repo_id: str
    commit_hash: str
    repo_full_name: Optional[str]
    base_branch: str
    create_pr: bool
    draft_pr: bool
    seed_files: Optional[list[str]]
    confidence_threshold: float
    max_retries: int
    max_rag_results: int

    # ---- pipeline outputs (written as nodes run) ----
    issue: Optional[dict]                 # Issue Agent output
    rag_chunks: Optional[list]            # RAG retrieval (Repository Indexing)
    call_graph_trace: Optional[dict]      # Code Navigation output
    root_cause: Optional[dict]            # Root Cause Agent output

    target_file: Optional[str]            # the file currently being patched
    file_backup: Optional[bytes]          # original bytes of target_file, for revert-between-retries
    patch: Optional[dict]                 # {"diff": str, "scope_check": dict}
    diff_applies: Optional[dict]          # {"applies": bool, "error": str | None}
    validation: Optional[dict]            # validators.test_validator.validate_patch() output
    attempt: int                          # number of patch attempts made so far

    pr: Optional[dict]                    # {"title": str, "description": str}
    pr_result: Optional[dict]             # branch/commit/push/PR outcome

    # ---- terminal outcome ----
    status: str                           # see STATUS_* constants below
    outcome_reason: str                   # human-readable explanation of the final status

    # ---- running log for explainability (appended, not overwritten) ----
    history: Annotated[list, operator.add]


# Terminal status values. Kept as constants so the UI and tests can
# branch on them without magic strings drifting out of sync.
STATUS_RUNNING = "running"
STATUS_PR_CREATED = "pr_created"          # live PR opened on GitHub
STATUS_PR_COMMITTED_LOCAL = "pr_committed_local"  # branch+commit done, no remote/token to push
STATUS_PR_READY = "pr_ready"              # validated fix + description ready, no commit attempted
STATUS_NEEDS_HUMAN_REVIEW = "needs_human_review"  # low confidence or exhausted retries
STATUS_ERROR = "error"                    # an agent raised — pipeline aborted


def _git_head(repo_path: str) -> str:
    """Returns the current HEAD commit hash, or a stable fallback string
    if repo_path isn't a usable git repo (the RAG cache key just needs to
    be *consistent*, not necessarily a real SHA)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        head = result.stdout.strip()
        return head or "no-head"
    except (subprocess.SubprocessError, OSError):
        return "no-head"


def _git_current_branch(repo_path: str) -> str:
    """Returns the repo's current branch name, defaulting to 'main' if it
    can't be determined (e.g. detached HEAD or not a git repo)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        branch = result.stdout.strip()
        return branch if branch and branch != "HEAD" else "main"
    except (subprocess.SubprocessError, OSError):
        return "main"


def initial_state(request: BugFixRequest) -> GraphState:
    """
    Builds the starting GraphState from a validated request, filling in
    every key (with None/defaults) so nodes never hit a missing-key error,
    and deriving repo identity from git where the caller didn't specify it.
    """
    repo_id = request.repo_id or request.repo_path
    commit_hash = request.commit_hash or _git_head(request.repo_path)
    base_branch = request.base_branch or _git_current_branch(request.repo_path)

    return GraphState(
        # inputs
        bug_report=request.bug_report,
        repo_path=request.repo_path,
        repo_id=repo_id,
        commit_hash=commit_hash,
        repo_full_name=request.repo_full_name,
        base_branch=base_branch,
        create_pr=request.create_pr,
        draft_pr=request.draft_pr,
        seed_files=request.seed_files,
        confidence_threshold=request.confidence_threshold,
        max_retries=request.max_retries,
        max_rag_results=request.max_rag_results,
        # outputs (not computed yet)
        issue=None,
        rag_chunks=None,
        call_graph_trace=None,
        root_cause=None,
        target_file=None,
        file_backup=None,
        patch=None,
        diff_applies=None,
        validation=None,
        attempt=0,
        pr=None,
        pr_result=None,
        status=STATUS_RUNNING,
        outcome_reason="",
        history=[],
    )


def log_entry(node: str, summary: str, **details: Any) -> dict:
    """Small helper so every node logs to `history` in the same shape.
    The Streamlit explainability view reads this list to show, step by
    step, what the pipeline did and why."""
    entry = {"node": node, "summary": summary}
    if details:
        entry["details"] = details
    return entry
