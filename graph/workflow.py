"""
workflow.py — the LangGraph orchestrator that ties all agents together.

This is the single callable thing the Streamlit dashboard (Agent 8) will
drive: hand it a BugFixRequest, get back a fully-populated GraphState with
the diagnosis, the patch, the validation result, the PR (or the reason one
wasn't opened), and a step-by-step history for the explainability view.

The graph is mostly linear — issue → index → multi-agent retrieval →
navigate → root cause → patch → validate. Retrieval (rag/agentic.py) is its
own small agent loop:

  plan_retrieval ──► retrieve ──► critique_retrieval ──► navigate_code
                        ▲                 │
                        └── follow-ups ───┘   (critic not satisfied, < max rounds)

and one conditional edge after validation encodes every interesting
decision in the fix loop:

        ┌─────────────────────────────────────────────┐
        │                                             │ retry (< max)
        ▼                                             │
  generate_patch ──► validate ──► route_after_validation
                                     │   │   │
                  passing+confident ─┘   │   └─ passing but low confidence
                          │              │              │
                          ▼     retries exhausted       ▼
                      create_pr         │          human_review ──► END
                          │             ▼               ▲
                          ▼        human_review ────────┘
                         END

The retry edge loops back to the Fix Agent (capped at max_retries, then
escalates to human review) — that loop is the whole reason this is a
LangGraph and not a straight function call.

Public API:
    build_workflow()            -> compiled LangGraph app
    run_pipeline(request)       -> final GraphState (never raises; errors
                                   are captured into status=STATUS_ERROR)
"""

from __future__ import annotations

import traceback

from langgraph.graph import StateGraph, START, END

from graph.state import (
    BugFixRequest,
    GraphState,
    initial_state,
    log_entry,
    STATUS_ERROR,
)
from graph import nodes


# How many supersteps LangGraph will run before bailing. The deepest path
# is the retry loop: each attempt is generate_patch + validate (2 steps),
# plus the ~6 linear nodes around it. 50 leaves comfortable headroom over
# the default of 25 even at the max retry count.
RECURSION_LIMIT = 50


def build_workflow():
    """Constructs and compiles the LangGraph. Compilation is cheap, so the
    UI can call this once at startup and reuse the returned app across runs."""
    builder = StateGraph(GraphState)

    builder.add_node("understand_issue", nodes.understand_issue_node)
    builder.add_node("index_repo", nodes.index_repo_node)
    builder.add_node("plan_retrieval", nodes.plan_retrieval_node)
    builder.add_node("retrieve", nodes.retrieve_node)
    builder.add_node("critique_retrieval", nodes.critique_retrieval_node)
    builder.add_node("navigate_code", nodes.navigate_code_node)
    builder.add_node("root_cause", nodes.root_cause_node)
    builder.add_node("generate_patch", nodes.generate_patch_node)
    builder.add_node("validate", nodes.validate_node)
    builder.add_node("create_pr", nodes.create_pr_node)
    builder.add_node("human_review", nodes.human_review_node)

    # Linear spine.
    builder.add_edge(START, "understand_issue")
    builder.add_edge("understand_issue", "index_repo")
    # Multi-agent RAG: planner -> retrievers -> critic, looping back to the
    # retrievers while the critic asks for follow-up searches.
    builder.add_edge("index_repo", "plan_retrieval")
    builder.add_edge("plan_retrieval", "retrieve")
    builder.add_edge("retrieve", "critique_retrieval")
    builder.add_conditional_edges(
        "critique_retrieval",
        nodes.route_after_critique,
        {"retrieve": "retrieve", "navigate_code": "navigate_code"},
    )
    builder.add_edge("navigate_code", "root_cause")
    builder.add_edge("root_cause", "generate_patch")
    builder.add_edge("generate_patch", "validate")

    # The one conditional edge — retry / PR / escalate.
    builder.add_conditional_edges(
        "validate",
        nodes.route_after_validation,
        {
            "create_pr": "create_pr",
            "generate_patch": "generate_patch",  # retry loop
            "human_review": "human_review",
        },
    )

    # Both terminal nodes end the run.
    builder.add_edge("create_pr", END)
    builder.add_edge("human_review", END)

    return builder.compile()


# Compile once at import so callers (and run_pipeline) reuse a single app.
_APP = build_workflow()


def run_pipeline(request: BugFixRequest | dict, on_step=None) -> GraphState:
    """
    Runs the full pipeline for a single bug report.

    Args:
        request: a BugFixRequest, or a plain dict that can construct one
            (convenient for the Streamlit form / JSON API).
        on_step: optional callback, called with each node's history entry
            ({"node", "summary", ...}) as soon as that node finishes. Lets the
            UI show live progress instead of one long silent spinner.

    Returns:
        The final GraphState. This never raises: if any agent throws, the
        exception is captured into status=STATUS_ERROR with the traceback
        in outcome_reason, so the UI always has a state to render rather
        than an unhandled crash mid-pipeline.
    """
    if isinstance(request, dict):
        request = BugFixRequest(**request)

    state = initial_state(request)

    config = {"recursion_limit": RECURSION_LIMIT}
    try:
        if on_step is None:
            return _APP.invoke(state, config=config)

        # Stream full-state snapshots; each node appends one history entry, so
        # the newest entry identifies the step that just completed.
        final = state
        seen = 0
        for snapshot in _APP.stream(state, config=config, stream_mode="values"):
            final = snapshot
            history = snapshot.get("history") or []
            for entry in history[seen:]:
                try:
                    on_step(entry)
                except Exception:  # noqa: BLE001 — a UI callback must never kill a run
                    pass
            seen = len(history)
        return final
    except Exception as e:  # noqa: BLE001 — top-level guard for the UI
        from agents.llm_client import ProviderError
        state["status"] = STATUS_ERROR
        if isinstance(e, ProviderError):
            # Plain-language and key-free: safe to show users in production.
            state["error_kind"] = "provider"
            state["outcome_reason"] = str(e)
        else:
            state["outcome_reason"] = f"Pipeline aborted: {type(e).__name__}: {e}"
        state["history"] = state.get("history", []) + [
            log_entry("pipeline", "Aborted with an unhandled error.",
                      error=str(e), traceback=traceback.format_exc())
        ]
        return state


def print_report(final: GraphState) -> None:
    """Pretty-prints a run's outcome — a terminal stand-in for the Streamlit
    explainability view, useful for the __main__ demo and debugging."""
    print("\n" + "=" * 70)
    print(f"STATUS: {final.get('status')}")
    print(f"REASON: {final.get('outcome_reason')}")
    print("=" * 70)

    rc = final.get("root_cause")
    if rc:
        print(f"\nRoot cause   : {rc.get('root_cause')}")
        print(f"Affected     : {', '.join(rc.get('affected_files', []))} :: {rc.get('affected_function')}()")
        print(f"Confidence   : {rc.get('confidence')}")

    val = final.get("validation") or {}
    targeted = val.get("targeted_result") or {}
    if targeted:
        print(f"Tests        : {targeted.get('passed_count')}/{targeted.get('total')} passed "
              f"(overall_passed={val.get('overall_passed')})")

    print("\n--- Pipeline history ---")
    for i, step in enumerate(final.get("history", []), 1):
        print(f"{i:>2}. [{step['node']}] {step['summary']}")

    pr = final.get("pr")
    if pr:
        print("\n--- Generated PR description ---")
        print(pr["description"])


if __name__ == "__main__":
    # End-to-end demo against the bundled sample repo. Snapshots the repo's
    # working tree first and restores it afterward so the run is repeatable
    # (on the success path the orchestrator deliberately leaves the fix
    # applied, which we then revert here purely for demo cleanliness).
    import subprocess
    import sys

    # The PR description (and GitHub-facing output) uses ✅/❌; Windows'
    # default cp1252 console can't encode those, so print as UTF-8 here.
    # This only affects this terminal demo — Streamlit renders UTF-8 natively.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    REPO = "sample-repo"

    print("Running AuraFix pipeline against the sample repo...\n")
    request = BugFixRequest(
        bug_report="Login page crashes when password field is empty",
        repo_path=REPO,
        repo_id="local-sample-repo",
        create_pr=False,  # local repo has no GitHub remote — stop at pr_ready
    )

    final = run_pipeline(request)
    print_report(final)

    print("\n--- Restoring sample repo to a clean state (demo cleanup) ---")
    subprocess.run(["git", "checkout", "--", "."], cwd=REPO,
                   capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--short"], cwd=REPO,
                            capture_output=True, text=True)
    print("Clean." if not status.stdout.strip() else f"Remaining changes:\n{status.stdout}")
