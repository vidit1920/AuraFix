"""
nodes.py — the individual steps of the orchestrated pipeline.

Each function here is a LangGraph *node*: it takes the shared GraphState,
does one agent's worth of work, and returns a partial dict that LangGraph
merges back into the state. Nodes never mutate the state dict in place and
never decide control flow themselves — routing lives in
`route_after_validation` and the graph wiring in graph/workflow.py. That
separation is deliberate: the retry cap and escalation rules should exist
in exactly one place, not be smeared across every node that might trigger
a retry.

The one genuinely tricky bit is the retry loop's interaction with the
working tree. The Test Validator applies the patch *for real* before
running pytest, so after a failed attempt the file on disk holds a broken
patch. Before generating the next attempt we must restore the file to its
original bytes — otherwise the new diff would be computed against the
already-patched file and compound. We snapshot the original bytes once (on
the first attempt) and restore from that snapshot, which is git-independent
and robust on Windows (where `git checkout` proved flaky when a zipped
repo's .git folder didn't survive extraction).
"""

from __future__ import annotations

import os
import re

from agents.issue_agent import understand_issue
from agents.root_cause_agent import analyze_root_cause
from agents.patch_agent import generate_patch, validate_diff_applies, apply_diff
from analyzers.call_graph import build_repo_structure, trace_call_path
from rag.vector_store import get_or_build_collection, query_collection
from validators.test_validator import validate_patch
from github_integration.pr_agent import (
    build_pr_title,
    build_pr_description,
    create_branch_and_commit,
    open_pull_request,
)

from graph.state import (
    GraphState,
    log_entry,
    STATUS_NEEDS_HUMAN_REVIEW,
    STATUS_PR_CREATED,
    STATUS_PR_COMMITTED_LOCAL,
    STATUS_PR_READY,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _norm(path: str) -> str:
    """Forward-slash normalize, matching how every other module stores
    repo-relative paths (see analyzers/call_graph.py for the rationale)."""
    return path.replace("\\", "/")


def _derive_seed_files(state: GraphState, structures: dict) -> list[str]:
    """
    Picks where the call-graph trace should start.

    Priority:
      1. Explicit seed_files from the request (e.g. parsed from a stack
         trace) — most reliable, so honored first.
      2. The Issue Agent's possible_files guesses, resolved to real repo
         paths (matched exactly, or by basename since the Issue Agent
         guesses "auth_service.py" while the repo stores "auth/auth_service.py").
      3. The filenames RAG retrieved — these are ground-truth paths.
      4. Fallback: every file in the repo. For a small repo this is fine;
         trace_call_path bounds the traversal by max_depth regardless.
    """
    real_paths = set(structures.keys())
    by_basename: dict[str, list[str]] = {}
    for p in real_paths:
        by_basename.setdefault(os.path.basename(p), []).append(p)

    def resolve(guess: str) -> list[str]:
        g = _norm(guess)
        if g in real_paths:
            return [g]
        return by_basename.get(os.path.basename(g), [])

    seeds: list[str] = []

    explicit = state.get("seed_files")
    if explicit:
        for g in explicit:
            seeds.extend(resolve(g))

    issue = state.get("issue") or {}
    for g in issue.get("possible_files", []):
        seeds.extend(resolve(g))

    for chunk in state.get("rag_chunks") or []:
        fn = chunk.get("filename")
        if fn in real_paths:
            seeds.append(fn)

    # De-dupe while preserving order.
    seen = set()
    ordered = [s for s in seeds if not (s in seen or seen.add(s))]

    return ordered or sorted(real_paths)


def _resolve_target_file(root_cause: dict, repo_path: str) -> str | None:
    """Picks the file to patch from the diagnosis: the first affected file
    that actually exists on disk. Returns None if none do (the diagnosis
    pointed at files not in this repo)."""
    for f in root_cause.get("affected_files", []):
        candidate = _norm(f)
        if os.path.isfile(os.path.join(repo_path, candidate)):
            return candidate
    return None


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------
def understand_issue_node(state: GraphState) -> dict:
    """Agent 1: parse the raw bug report into structured JSON."""
    issue = understand_issue(state["bug_report"])
    return {
        "issue": issue,
        "history": [log_entry(
            "issue_agent",
            f"Parsed bug report -> module '{issue.get('module')}', severity '{issue.get('severity')}'.",
            possible_files=issue.get("possible_files"),
            root_problem_category=issue.get("root_problem_category"),
        )],
    }


def index_repo_node(state: GraphState) -> dict:
    """Repository Indexing + RAG retrieval: build (or reuse) the embedded
    collection for this repo+commit, then retrieve the chunks most similar
    to the bug report. Capped at max_rag_results as a token-budget guard."""
    collection = get_or_build_collection(
        repo_path=state["repo_path"],
        repo_id=state["repo_id"],
        commit_hash=state["commit_hash"],
    )
    rag_chunks = query_collection(
        collection,
        state["bug_report"],
        n_results=state["max_rag_results"],
    )
    return {
        "rag_chunks": rag_chunks,
        "history": [log_entry(
            "repo_indexing",
            f"Retrieved {len(rag_chunks)} relevant code chunk(s) via RAG.",
            files=sorted({c["filename"] for c in rag_chunks}),
        )],
    }


def navigate_code_node(state: GraphState) -> dict:
    """Code Navigation: build the AST call graph and trace the real
    execution path outward from the seed files. This is the ground-truth
    signal the Root Cause Agent trusts over semantic RAG guesses."""
    structures = build_repo_structure(state["repo_path"])
    seeds = _derive_seed_files(state, structures)
    trace = trace_call_path(structures, seed_files=seeds, max_depth=3)
    return {
        "seed_files": seeds,
        "call_graph_trace": trace,
        "history": [log_entry(
            "code_navigation",
            f"Traced execution path across {len(trace['files_visited'])} file(s) from {len(seeds)} seed(s).",
            files_visited=trace["files_visited"],
        )],
    }


def root_cause_node(state: GraphState) -> dict:
    """Agent 4: combine issue + call graph (ground truth) + RAG (semantic)
    into a single root-cause diagnosis with a confidence score."""
    diagnosis = analyze_root_cause(
        issue_report=state["issue"],
        call_graph_trace=state.get("call_graph_trace"),
        rag_chunks=state.get("rag_chunks"),
    )
    # Normalize affected file paths so downstream patching/diffing matches
    # git's forward-slash convention regardless of what the model emitted.
    diagnosis["affected_files"] = [_norm(f) for f in diagnosis.get("affected_files", [])]
    return {
        "root_cause": diagnosis,
        "history": [log_entry(
            "root_cause_agent",
            f"Diagnosed root cause (confidence {diagnosis.get('confidence')}).",
            root_cause=diagnosis.get("root_cause"),
            affected_files=diagnosis.get("affected_files"),
            affected_function=diagnosis.get("affected_function"),
        )],
    }


def generate_patch_node(state: GraphState) -> dict:
    """Agent 5 (Fix): generate a minimal diff for the primary affected file.

    Handles both the first attempt and retries:
      - First attempt: resolve + snapshot the target file's original bytes.
      - Retry: restore the original bytes (undo the prior failed patch) and
        feed the previous diff + failure detail back to the model so it
        revises rather than repeats.
    """
    attempt = state["attempt"] + 1
    repo_path = state["repo_path"]
    root_cause = state["root_cause"]

    target_file = state.get("target_file") or _resolve_target_file(root_cause, repo_path)
    if target_file is None:
        # The diagnosis pointed at files that don't exist in this repo —
        # retrying won't conjure them. Surface a clear, non-looping error.
        raise FileNotFoundError(
            "Root Cause Agent named affected files that don't exist in the repo: "
            f"{root_cause.get('affected_files')}"
        )

    full_path = os.path.join(repo_path, target_file)

    # Snapshot original bytes on the first attempt; reuse the snapshot on
    # retries. Stored in state so it survives across the retry cycle.
    file_backup = state.get("file_backup")
    if file_backup is None:
        with open(full_path, "rb") as f:
            file_backup = f.read()

    retry_context = None
    if attempt > 1:
        # Undo the previous (failed) patch before regenerating, so the new
        # diff is computed against the original file, not a broken one.
        with open(full_path, "wb") as f:
            f.write(file_backup)

        prev_validation = state.get("validation") or {}
        targeted = prev_validation.get("targeted_result") or {}
        retry_context = {
            "previous_diff": (state.get("patch") or {}).get("diff", ""),
            "attempt": attempt,
            "max_attempts": state["max_retries"],
            "failed_tests": targeted.get("failed_tests", []),
            "apply_error": prev_validation.get("apply_error"),
            "test_error": targeted.get("error"),
        }

    patch = generate_patch(root_cause, target_file, repo_path, retry_context=retry_context)

    summary = (
        f"Generated patch for {target_file} (attempt {attempt}/{state['max_retries']}, "
        f"{patch['scope_check']['changed_lines']} line(s) changed)."
    )
    return {
        "target_file": target_file,
        "file_backup": file_backup,
        "patch": patch,
        "attempt": attempt,
        "history": [log_entry("patch_agent", summary,
                              scope_check=patch["scope_check"],
                              is_retry=attempt > 1)],
    }


def validate_node(state: GraphState) -> dict:
    """Agent 6 (Test Validator): validation-layer-zero (does the diff even
    apply?) then apply-for-real + pytest. Returns structured pass/fail data
    including tracebacks, which the retry loop feeds back to the Fix Agent."""
    repo_path = state["repo_path"]
    diff = state["patch"]["diff"]

    applies = validate_diff_applies(diff, repo_path)
    if not applies["applies"]:
        # Layer zero failed — no point running pytest on a patch that never
        # touched the file. Record it as a validation failure so routing
        # retries (with the apply error fed back) or escalates.
        validation = {
            "targeted_result": None,
            "full_suite_result": None,
            "overall_passed": False,
            "apply_error": applies["error"],
        }
        return {
            "diff_applies": applies,
            "validation": validation,
            "history": [log_entry(
                "test_validator",
                "Diff did not apply cleanly (validation layer zero failed).",
                error=applies["error"],
            )],
        }

    apply_diff(diff, repo_path)
    validation = validate_patch(
        repo_path,
        state["root_cause"]["affected_files"],
        run_full_suite_on_pass=True,
    )

    targeted = validation.get("targeted_result") or {}
    summary = (
        "All tests passed."
        if validation["overall_passed"]
        else f"Tests failed ({targeted.get('failed_count', '?')} failing)."
    )
    return {
        "diff_applies": applies,
        "validation": validation,
        "history": [log_entry("test_validator", summary,
                              overall_passed=validation["overall_passed"],
                              targeted=targeted)],
    }


def create_pr_node(state: GraphState) -> dict:
    """Agent 7 (PR): pure templating of facts already in state, then —
    only if the caller opted into live PR creation and a remote/token
    exist — branch, commit, push, and open the PR. For a local sample
    repo (the default), it stops at 'pr_ready' with the description built."""
    root_cause = state["root_cause"]
    diff = state["patch"]["diff"]
    targeted = (state["validation"].get("targeted_result")) or {}

    title = build_pr_title(root_cause)
    description = build_pr_description(root_cause, diff, targeted)
    pr = {"title": title, "description": description}

    # Local mode: no live PR requested, or no repo to open it against.
    if not state.get("create_pr") or not state.get("repo_full_name"):
        return {
            "pr": pr,
            "status": STATUS_PR_READY,
            "outcome_reason": (
                "Validated fix is ready and the PR description is prepared. "
                "Live PR creation was not requested (local mode)."
            ),
            "history": [log_entry("pr_agent",
                                  "Built PR title + description (local mode — no PR opened).",
                                  title=title)],
        }

    # Live mode: branch + commit locally first.
    repo_path = state["repo_path"]
    base_branch = state["base_branch"]
    slug = re.sub(r"[^a-z0-9]+", "-", (root_cause.get("affected_function") or "fix").lower()).strip("-")
    branch_name = f"bugfix/aurafix-{slug or 'fix'}"

    commit = create_branch_and_commit(repo_path, branch_name, commit_message=title, base_branch=base_branch)
    if not commit["success"]:
        return {
            "pr": pr,
            "pr_result": commit,
            "status": STATUS_NEEDS_HUMAN_REVIEW,
            "outcome_reason": f"Fix validated but could not commit it: {commit['error']}",
            "history": [log_entry("pr_agent", "Branch/commit failed.", error=commit["error"])],
        }

    # Push to the right place and open the PR. open_pull_request
    # auto-selects: direct push if you can write to the upstream repo,
    # otherwise fork -> push to your fork -> cross-fork PR (the normal
    # open-source contributor flow). If anything fails we still have a
    # local commit, so we report that honestly rather than faking a PR.
    result = open_pull_request(
        repo_path=repo_path,
        upstream_full_name=state["repo_full_name"],
        branch_name=branch_name,
        title=title,
        description=description,
        base_branch=base_branch,
        draft=state.get("draft_pr", False),
    )

    if result["success"]:
        mode_note = "from your fork" if result["mode"] == "fork" else "directly to the upstream repo"
        return {
            "pr": pr,
            "pr_result": {"branch": branch_name, "committed": True, "url": result["url"],
                          "mode": result["mode"], "fork_full_name": result.get("fork_full_name")},
            "status": STATUS_PR_CREATED,
            "outcome_reason": f"Pull request opened {mode_note}: {result['url']}",
            "history": [log_entry("pr_agent", f"Pull request opened ({result['mode']} mode).",
                                  url=result["url"], mode=result["mode"],
                                  fork=result.get("fork_full_name"))],
        }

    return {
        "pr": pr,
        "pr_result": {"branch": branch_name, "committed": True, "mode": result.get("mode"),
                      "fork_full_name": result.get("fork_full_name"), "error": result["error"]},
        "status": STATUS_PR_COMMITTED_LOCAL,
        "outcome_reason": f"Fix committed locally on '{branch_name}', but opening the PR failed: {result['error']}",
        "history": [log_entry("pr_agent", "Committed locally; PR creation failed.", error=result["error"])],
    }


def human_review_node(state: GraphState) -> dict:
    """Terminal escalation: reached when retries are exhausted, the diff
    never applied, or the fix validated but confidence is below threshold.

    If the fix did NOT validate, restore the original file so the repo is
    left clean rather than holding a broken patch. If it DID validate but
    confidence is low, keep the patch applied — it's a real, passing fix,
    just flagged for a human to sign off on before a PR is opened."""
    validation = state.get("validation") or {}
    passed = validation.get("overall_passed", False)

    if not passed:
        # Roll back the last failed patch so the working tree is clean.
        target_file = state.get("target_file")
        backup = state.get("file_backup")
        if target_file and backup is not None:
            with open(os.path.join(state["repo_path"], target_file), "wb") as f:
                f.write(backup)

        if (state.get("diff_applies") or {}).get("applies") is False:
            reason = "Generated patch never applied cleanly; escalated for human review."
        else:
            reason = (
                f"Could not produce a passing fix within {state['max_retries']} attempt(s); "
                "escalated for human review. The repo was restored to its original state."
            )
        summary = "Escalated to human review (no passing fix found)."
    else:
        conf = (state.get("root_cause") or {}).get("confidence")
        reason = (
            f"Fix validated (tests pass) but confidence {conf} is below the "
            f"{state['confidence_threshold']} threshold for auto-PR — flagged for human review."
        )
        summary = "Escalated to human review (low confidence despite passing tests)."

    return {
        "status": STATUS_NEEDS_HUMAN_REVIEW,
        "outcome_reason": reason,
        "history": [log_entry("human_review", summary)],
    }


# --------------------------------------------------------------------------
# Routing (the conditional edge after validation)
# --------------------------------------------------------------------------
def route_after_validation(state: GraphState) -> str:
    """
    The one decision point that earns LangGraph its place here. Returns the
    name of the next node:

      - passing tests + confidence >= threshold  → "create_pr"
      - passing tests + low confidence           → "human_review"
      - failing tests + retries remaining        → "generate_patch" (retry)
      - failing tests + retries exhausted         → "human_review"

    The retry cap is non-negotiable: without it, a genuinely hard bug loops
    forever, silently burning Gemini quota until the rate limit kills the run.
    """
    validation = state["validation"]

    if validation["overall_passed"]:
        confidence = (state.get("root_cause") or {}).get("confidence", 0.0)
        if confidence >= state["confidence_threshold"]:
            return "create_pr"
        return "human_review"

    if state["attempt"] >= state["max_retries"]:
        return "human_review"
    return "generate_patch"
