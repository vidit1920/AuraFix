"""
test_workflow.py — deterministic, offline tests for the LangGraph orchestrator.

The point of these tests is to prove the *orchestration logic* is correct —
state passing, the retry loop, the retry cap → human-review escalation, the
confidence gate, and PR templating — without depending on the live Gemini
API (which is non-deterministic and rate-limited on the free tier).

Strategy: mock only the three LLM-backed steps (issue parsing, RAG
retrieval, root-cause analysis) and the embedding/vector calls. The Fix
Agent is NOT mocked at the node level — instead we mock the single
`call_llm` it calls, feeding it canned "corrected file content". That keeps
everything below it real: the real difflib diff computation, the real
`git apply`, and a real `pytest` run against the sample repo. So a passing
test here means a real patch really applied and real tests really passed —
the graph is exercised end to end, just with a deterministic "model".

Run with:  python -m pytest tests/test_workflow.py -v
"""

import os

import pytest

import graph.nodes as nodes
import agents.patch_agent as patch_agent
from graph.state import (
    BugFixRequest,
    STATUS_PR_READY,
    STATUS_NEEDS_HUMAN_REVIEW,
)
from graph.workflow import run_pipeline

REPO = "sample-repo"
TARGET = os.path.join(REPO, "auth", "auth_service.py")

# Snapshot the pristine target file once, at import — the repo is clean at
# session start, so this captures the original buggy source. Tests restore
# from this snapshot, which is git-independent (the sample ships as a plain
# folder in the repo, not a nested git repo).
with open(TARGET, "rb") as _f:
    _PRISTINE_BYTES = _f.read()


# --------------------------------------------------------------------------
# Fixtures + canned model outputs
# --------------------------------------------------------------------------
@pytest.fixture
def clean_repo():
    """Restore the sample's patched file to its pristine bytes after each
    test — the orchestrator applies patches for real, so every test must
    leave the repo clean for the next one."""
    yield
    with open(TARGET, "wb") as f:
        f.write(_PRISTINE_BYTES)


def _pristine_source() -> str:
    """Restore the target file and return its original content (LF-normalized),
    so the canned 'fixes' below are computed against the real file the diff
    will be applied to."""
    with open(TARGET, "wb") as f:
        f.write(_PRISTINE_BYTES)
    return _PRISTINE_BYTES.decode("utf-8").replace("\r\n", "\n")


def _good_fix(original: str) -> str:
    """A correct minimal fix: guard the None case before .strip()."""
    return original.replace(
        "    password = password.strip()  # BUG: crashes if password is None",
        '    if password is None:\n'
        '        return {"success": False, "message": "Password required"}\n'
        "    password = password.strip()",
    )


def _broken_fix(original: str) -> str:
    """A patch that applies cleanly but introduces a syntax error, so pytest
    fails to even collect the test module — exercises the collection-error
    path that must NOT be treated as a pass."""
    return original.replace(
        "    password = password.strip()  # BUG: crashes if password is None",
        "    password = password.strip(  # deliberately broken syntax",
    )


def _install_common_mocks(monkeypatch, confidence=1.0):
    """Mock the LLM/embedding-backed steps the graph calls, leaving the call
    graph (pure AST), diff application, and pytest real."""
    issue = {
        "module": "authentication",
        "severity": "high",
        "suspected_area": "login flow",
        "possible_files": ["login.py", "auth_service.py"],
        "root_problem_category": "null/missing input validation",
        "investigation_areas": ["input validation", "login form handling"],
    }
    diagnosis = {
        "root_cause": "authenticate() calls .strip() on password without a None check",
        "confidence": confidence,
        "affected_files": ["auth/auth_service.py"],
        "affected_function": "authenticate",
        "reasoning": [
            "login.py passes password straight to authenticate()",
            "auth_service.py calls .strip() without checking for None",
        ],
    }
    rag_chunks = [{
        "text": "password = password.strip()  # BUG",
        "filename": "auth/auth_service.py",
        "chunk_index": 1,
        "distance": 0.1,
    }]

    monkeypatch.setattr(nodes, "understand_issue", lambda bug_report: issue)
    monkeypatch.setattr(nodes, "get_or_build_collection", lambda **kw: object())
    monkeypatch.setattr(nodes, "query_collection", lambda c, q, n_results=5: rag_chunks)
    monkeypatch.setattr(nodes, "analyze_root_cause",
                        lambda issue_report, call_graph_trace=None, rag_chunks=None: dict(diagnosis))


def _request(**overrides) -> BugFixRequest:
    kwargs = dict(
        bug_report="Login page crashes when password field is empty",
        repo_path=REPO,
        repo_id="local-sample-repo",
        create_pr=False,
    )
    kwargs.update(overrides)
    return BugFixRequest(**kwargs)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_happy_path_first_attempt(monkeypatch, clean_repo):
    """A correct fix on the first attempt: validates (real pytest, 4/4),
    high confidence → status pr_ready with a PR description built."""
    original = _pristine_source()
    _install_common_mocks(monkeypatch, confidence=1.0)
    # The "model" returns the corrected full file content; real patch_agent
    # turns it into a real diff.
    monkeypatch.setattr(patch_agent, "call_llm", lambda *a, **k: _good_fix(original))

    final = run_pipeline(_request())

    assert final["status"] == STATUS_PR_READY
    assert final["attempt"] == 1
    assert final["validation"]["overall_passed"] is True
    assert final["validation"]["targeted_result"]["total"] == 4
    assert final["validation"]["targeted_result"]["passed_count"] == 4
    assert "auth/auth_service.py" in final["pr"]["description"]
    # The history should record one of each pipeline stage.
    nodes_run = [h["node"] for h in final["history"]]
    assert nodes_run.count("patch_agent") == 1
    assert "pr_agent" in nodes_run


def test_retry_then_succeed(monkeypatch, clean_repo):
    """First patch is broken (syntax error → collection fails → NOT a pass),
    so the graph must loop back to the Fix Agent and a second, correct patch
    must succeed. Proves the retry edge + the collection-error guard."""
    original = _pristine_source()
    _install_common_mocks(monkeypatch, confidence=1.0)

    outputs = iter([_broken_fix(original), _good_fix(original)])
    monkeypatch.setattr(patch_agent, "call_llm", lambda *a, **k: next(outputs))

    final = run_pipeline(_request())

    assert final["status"] == STATUS_PR_READY
    assert final["attempt"] == 2          # retried exactly once
    assert final["validation"]["overall_passed"] is True
    # Two patch attempts were logged, the second a retry.
    patch_logs = [h for h in final["history"] if h["node"] == "patch_agent"]
    assert len(patch_logs) == 2
    assert patch_logs[1]["details"]["is_retry"] is True


def test_exhausts_retries_then_human_review(monkeypatch, clean_repo):
    """Every patch is broken: the graph must hit the retry cap (3) and
    escalate to human review, then restore the repo to a clean state."""
    original = _pristine_source()
    _install_common_mocks(monkeypatch, confidence=1.0)
    monkeypatch.setattr(patch_agent, "call_llm", lambda *a, **k: _broken_fix(original))

    final = run_pipeline(_request(max_retries=3))

    assert final["status"] == STATUS_NEEDS_HUMAN_REVIEW
    assert final["attempt"] == 3
    assert final["validation"]["overall_passed"] is False
    # The failed patch must have been rolled back — target file restored.
    with open(TARGET, "rb") as f:
        assert f.read() == _PRISTINE_BYTES


def test_low_confidence_gate_blocks_pr(monkeypatch, clean_repo):
    """A fix that validates (tests pass) but whose diagnosis confidence is
    below the threshold must be escalated to human review, not auto-PR'd."""
    original = _pristine_source()
    _install_common_mocks(monkeypatch, confidence=0.4)  # below default 0.7
    monkeypatch.setattr(patch_agent, "call_llm", lambda *a, **k: _good_fix(original))

    final = run_pipeline(_request())

    assert final["status"] == STATUS_NEEDS_HUMAN_REVIEW
    assert final["validation"]["overall_passed"] is True   # tests DID pass
    assert "confidence" in final["outcome_reason"].lower()
