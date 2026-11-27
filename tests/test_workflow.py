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
import rag.agentic as agentic
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


class FakeCollection:
    """Stands in for a ChromaDB collection: just enough surface for the
    indexing node and the lexical/structural retrievers."""

    def __init__(self, chunks):
        self._chunks = chunks

    def count(self):
        return len(self._chunks)

    def get(self, include=None):
        return {"documents": [c["text"] for c in self._chunks],
                "metadatas": [{"filename": c["filename"], "chunk_index": c["chunk_index"]}
                              for c in self._chunks]}


def fake_retrieval_llm(critic_rounds_insufficient=0, calls=None):
    """Canned planner/critic responses. The critic reports 'insufficient'
    (with a follow-up search) for the first N rounds, then 'sufficient'."""
    state = {"critic": 0}

    def _call(prompt, system_instruction=None, **kw):
        if calls is not None:
            calls.append(system_instruction)
        if "Query Planner" in (system_instruction or ""):
            return {"queries": [{"text": "password strip None", "intent": "code"},
                                {"text": "login crashes on empty password", "intent": "symptom"}],
                    "keywords": ["password", "strip"], "symbols": ["authenticate"]}
        state["critic"] += 1
        if state["critic"] <= critic_rounds_insufficient:
            return {"relevant_ids": ["auth/auth_service.py::1"], "sufficient": False,
                    "missing": "the caller", "follow_up_queries": ["handle_login_request"]}
        return {"relevant_ids": ["auth/auth_service.py::1"], "sufficient": True,
                "missing": "", "follow_up_queries": []}
    return _call


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
    monkeypatch.setattr(nodes, "get_or_build_collection", lambda **kw: FakeCollection(rag_chunks))
    monkeypatch.setattr(nodes, "query_collection", lambda c, q, n_results=5: rag_chunks)
    # Multi-agent RAG: deterministic Query Planner + Relevance Critic.
    monkeypatch.setattr(agentic, "call_llm_json", fake_retrieval_llm())
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


def test_no_tests_escalates_without_retrying(monkeypatch, clean_repo):
    """A repo with no tests can never validate, so retrying only burns LLM
    calls. The graph must escalate after the first attempt, not loop."""
    original = _pristine_source()
    _install_common_mocks(monkeypatch, confidence=1.0)
    monkeypatch.setattr(patch_agent, "call_llm", lambda *a, **k: _good_fix(original))
    no_tests = {"passed": False, "total": 0, "passed_count": 0, "failed_count": 0,
                "failed_tests": [], "error": "No tests were collected.", "no_tests": True}
    monkeypatch.setattr(nodes, "validate_patch", lambda *a, **k: {
        "targeted_result": no_tests, "full_suite_result": None, "overall_passed": False})

    final = run_pipeline(_request(max_retries=3))

    assert final["status"] == STATUS_NEEDS_HUMAN_REVIEW
    assert final["attempt"] == 1
    assert "no tests" in final["outcome_reason"].lower()


def test_on_step_reports_each_stage(monkeypatch, clean_repo):
    """run_pipeline(on_step=...) must report every history entry, in order,
    and return the same final state as a plain run."""
    original = _pristine_source()
    _install_common_mocks(monkeypatch, confidence=1.0)
    monkeypatch.setattr(patch_agent, "call_llm", lambda *a, **k: _good_fix(original))

    seen = []
    final = run_pipeline(_request(), on_step=seen.append)

    assert final["status"] == STATUS_PR_READY
    assert [e["node"] for e in seen] == [h["node"] for h in final["history"]]
    assert seen[0]["node"] == "issue_agent" and seen[-1]["node"] == "pr_agent"



def test_retrieval_agents_run_and_feed_root_cause(monkeypatch, clean_repo):
    """Planner -> retrievers -> critic run in order before navigation, and
    the critic's pick reaches the Root Cause Agent with provenance."""
    original = _pristine_source()
    _install_common_mocks(monkeypatch, confidence=1.0)
    monkeypatch.setattr(patch_agent, "call_llm", lambda *a, **k: _good_fix(original))
    seen = {}
    monkeypatch.setattr(nodes, "analyze_root_cause",
                        lambda issue_report, call_graph_trace=None, rag_chunks=None:
                        seen.setdefault("rag", rag_chunks) and {
                            "root_cause": "x", "confidence": 1.0,
                            "affected_files": ["auth/auth_service.py"], "affected_function": "authenticate",
                            "reasoning": ["r"]})

    final = run_pipeline(_request())

    order = [h["node"] for h in final["history"]]
    assert order[:5] == ["issue_agent", "repo_indexing", "query_planner", "retrievers", "relevance_critic"]
    assert order.index("relevance_critic") < order.index("code_navigation")
    top = seen["rag"][0]
    assert top["filename"] == "auth/auth_service.py" and top["kept_by_critic"] is True
    assert "semantic" in top["sources"]
    assert final["retrieval_trace"][0]["critic"]["sufficient"] is True


def test_critic_follow_up_loops_back_to_retrievers(monkeypatch, clean_repo):
    """An unsatisfied critic sends the graph back to the retrievers with its
    follow-up searches, once, then the run continues."""
    original = _pristine_source()
    _install_common_mocks(monkeypatch, confidence=1.0)
    monkeypatch.setattr(agentic, "call_llm_json", fake_retrieval_llm(critic_rounds_insufficient=1))
    monkeypatch.setattr(patch_agent, "call_llm", lambda *a, **k: _good_fix(original))

    final = run_pipeline(_request(max_retrieval_rounds=2))

    nodes_run = [h["node"] for h in final["history"]]
    assert nodes_run.count("retrievers") == 2
    assert nodes_run.count("relevance_critic") == 2
    assert final["retrieval_trace"][1]["queries"] == ["handle_login_request"]
    assert final["status"] == STATUS_PR_READY


def test_retrieval_rounds_are_capped(monkeypatch, clean_repo):
    """A critic that is never satisfied cannot loop forever."""
    original = _pristine_source()
    _install_common_mocks(monkeypatch, confidence=1.0)
    monkeypatch.setattr(agentic, "call_llm_json", fake_retrieval_llm(critic_rounds_insufficient=99))
    monkeypatch.setattr(patch_agent, "call_llm", lambda *a, **k: _good_fix(original))
    # Every search turns up one new chunk, so only the round cap can stop it.
    counter = {"n": 0}

    def fresh(c, q, n_results=5):
        counter["n"] += 1
        return [{"text": f"x = {counter['n']}", "filename": "auth/db.py", "chunk_index": 100 + counter["n"],
                 "distance": 0.5}]
    monkeypatch.setattr(nodes, "query_collection", fresh)

    final = run_pipeline(_request(max_retrieval_rounds=3))

    assert [h["node"] for h in final["history"]].count("retrievers") == 3
    assert final["status"] == STATUS_PR_READY


def test_follow_up_that_finds_nothing_new_skips_the_critic_call(monkeypatch, clean_repo):
    """If a follow-up round adds no new code, the critic is not asked again
    (it would see exactly the same evidence): one LLM call saved."""
    original = _pristine_source()
    _install_common_mocks(monkeypatch, confidence=1.0)
    calls = []
    monkeypatch.setattr(agentic, "call_llm_json", fake_retrieval_llm(critic_rounds_insufficient=99, calls=calls))
    monkeypatch.setattr(patch_agent, "call_llm", lambda *a, **k: _good_fix(original))

    final = run_pipeline(_request(max_retrieval_rounds=3))

    critic_calls = [c for c in calls if "Relevance Critic" in (c or "")]
    assert len(critic_calls) == 1
    assert [h["node"] for h in final["history"]].count("retrievers") == 2
    assert "no new code" in final["history"][[h["node"] for h in final["history"]].index("relevance_critic", 5)]["summary"]
    assert final["status"] == STATUS_PR_READY


def test_fast_mode_makes_no_retrieval_llm_calls(monkeypatch, clean_repo):
    """Fast retrieval uses the rule-based plan and fused ranking only."""
    original = _pristine_source()
    _install_common_mocks(monkeypatch, confidence=1.0)
    calls = []
    monkeypatch.setattr(agentic, "call_llm_json", fake_retrieval_llm(calls=calls))
    monkeypatch.setattr(patch_agent, "call_llm", lambda *a, **k: _good_fix(original))

    final = run_pipeline(_request(retrieval_mode="fast"))

    assert calls == []
    assert final["status"] == STATUS_PR_READY
    assert final["rag_chunks"][0]["filename"] == "auth/auth_service.py"


def test_missing_toolchain_escalates_without_retrying(monkeypatch, clean_repo):
    """Regression (found live on sample-java without Maven): a missing test
    toolchain used to burn all 3 fix attempts. It must escalate at once."""
    original = _pristine_source()
    _install_common_mocks(monkeypatch, confidence=1.0)
    monkeypatch.setattr(patch_agent, "call_llm", lambda *a, **k: _good_fix(original))
    missing = {"passed": False, "total": 0, "passed_count": 0, "failed_count": 0, "failed_tests": [],
               "error": "'mvn' is not installed on this server, so the Java tests could not run.",
               "toolchain_missing": True}
    monkeypatch.setattr(nodes, "validate_patch", lambda *a, **k: {
        "targeted_result": missing, "full_suite_result": None, "overall_passed": False})

    final = run_pipeline(_request(max_retries=3))

    assert final["status"] == STATUS_NEEDS_HUMAN_REVIEW
    assert final["attempt"] == 1
    assert "mvn" in final["outcome_reason"] and "not verified" in final["outcome_reason"]
    validator = [h for h in final["history"] if h["node"] == "test_validator"][0]
    assert validator["summary"].startswith("Tests could not run")
