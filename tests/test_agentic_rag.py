"""
test_agentic_rag.py — offline tests for the multi-agent RAG (rag/agentic.py).

The retrievers run for real against the bundled sample repo (real chunker,
real AST call graph, real BM25); only the vector search and the two LLM
agents (planner, critic) are replaced with deterministic stand-ins.

Run with:  python -m pytest tests/test_agentic_rag.py -v
"""

import pytest

import rag.agentic as agentic
from rag.agentic import (
    BM25,
    code_tokens,
    critique_candidates,
    heuristic_plan,
    plan_retrieval,
    ranked,
    retrieve_candidates,
)
from rag.chunker import chunk_repository
from analyzers.call_graph import build_repo_structure
from agents.runtime import LLMSettings, SFR_MODEL_ID, sfr_available

REPO = "sample-repo"
ISSUE = {
    "module": "authentication", "severity": "high", "suspected_area": "login flow",
    "possible_files": ["login.py", "auth_service.py"],
    "root_problem_category": "null handling", "investigation_areas": ["password validation"],
}
BUG = "Login page crashes when the password field is empty"


@pytest.fixture(scope="module")
def corpus():
    return chunk_repository(REPO)


@pytest.fixture(scope="module")
def structures():
    return build_repo_structure(REPO)


def _no_semantic(query, n):
    return []


def _fail_llm(*a, **k):
    raise RuntimeError("429 RESOURCE_EXHAUSTED")


# --------------------------------------------------------------------------
# Tokeniser + BM25
# --------------------------------------------------------------------------
def test_code_tokens_split_identifiers():
    toks = code_tokens("checkLogin calls password_hash.strip()")
    assert {"checklogin", "check", "login", "password_hash", "password", "hash", "strip"} <= set(toks)
    assert "calls" in toks and "the" not in code_tokens("the")


def test_bm25_prefers_exact_identifier():
    docs = [code_tokens("def authenticate(username, password): password.strip()"),
            code_tokens("def get_user(username): return USERS.get(username)")]
    scores = BM25(docs).scores(code_tokens("password strip"))
    assert scores[0] > scores[1] == 0


# --------------------------------------------------------------------------
# Agent 1: Query Planner
# --------------------------------------------------------------------------
def test_planner_falls_back_when_llm_fails(monkeypatch):
    monkeypatch.setattr(agentic, "call_llm_json", _fail_llm)
    plan = plan_retrieval(BUG, ISSUE)
    assert plan["source"] == "heuristic" and "RESOURCE_EXHAUSTED" in plan["fallback_reason"]
    assert plan["queries"][0]["text"] == BUG
    assert "auth_service" in plan["symbols"]


def test_planner_normalises_llm_output(monkeypatch):
    monkeypatch.setattr(agentic, "call_llm_json", lambda *a, **k: {
        "queries": [{"text": "password.strip() on None", "intent": "code"}, "auth flow", {"bad": 1}],
        "keywords": ["strip", "strip", 7], "symbols": ["authenticate"]})
    plan = plan_retrieval(BUG, ISSUE)
    assert plan["source"] == "llm"
    texts = [q["text"] for q in plan["queries"]]
    assert texts[0] == BUG and "password.strip() on None" in texts and "auth flow" in texts
    assert plan["keywords"] == ["strip"] and plan["symbols"] == ["authenticate"]


def test_fast_mode_plan_never_calls_llm(monkeypatch):
    monkeypatch.setattr(agentic, "call_llm_json", _fail_llm)
    plan = plan_retrieval(BUG, ISSUE, use_llm=False)
    assert plan["source"] == "heuristic" and "fallback_reason" not in plan


# --------------------------------------------------------------------------
# Agent 2: Retrievers + fusion (real sample repo)
# --------------------------------------------------------------------------
def test_lexical_and_structural_find_the_buggy_function(corpus, structures):
    plan = {"queries": [{"text": "password strip None", "intent": "code"}],
            "keywords": ["strip"], "symbols": ["authenticate"]}
    res = retrieve_candidates(plan["queries"], plan, corpus, structures, _no_semantic)
    top = ranked(res["candidates"])[0]
    assert top["filename"] == "auth/auth_service.py"
    assert {"lexical", "structural"} <= set(top["sources"])
    assert res["found"]["semantic"] == 0 and not res["errors"]


def test_structural_follows_call_graph_to_caller(corpus, structures):
    plan = {"queries": [], "keywords": [], "symbols": ["authenticate"]}
    res = retrieve_candidates([], plan, corpus, structures, _no_semantic)
    files = {c["filename"] for c in res["candidates"].values()}
    # login.py's handle_login_request() calls authenticate(): one hop up.
    assert {"auth/auth_service.py", "login.py"} <= files


def test_fusion_rewards_agreement_and_survives_a_failed_retriever(corpus, structures):
    target = next(c for c in corpus if c["filename"] == "auth/auth_service.py" and "def authenticate" in c["text"])
    other = next(c for c in corpus if c["filename"] == "auth/db.py")

    def semantic(q, n):
        return [{**other, "distance": 0.1}, {**target, "distance": 0.2}]

    plan = {"queries": [{"text": "password strip", "intent": "code"}], "keywords": [], "symbols": []}
    res = retrieve_candidates(plan["queries"], plan, corpus, structures, semantic)
    assert ranked(res["candidates"])[0]["id"] == f"{target['filename']}::{target['chunk_index']}"

    def broken(q, n):
        raise ConnectionError("vector store down")
    res2 = retrieve_candidates(plan["queries"], plan, corpus, structures, broken)
    assert "semantic" in res2["errors"] and res2["candidates"]


def test_source_outranks_its_test_file(corpus, structures):
    """Regression: for 'password strip', tests/test_auth_service.py (which
    repeats those words) used to outrank the buggy function itself."""
    plan = {"queries": [{"text": "password strip", "intent": "code"}], "keywords": [], "symbols": []}
    res = retrieve_candidates(plan["queries"], plan, corpus, structures, _no_semantic)
    assert ranked(res["candidates"])[0]["filename"] == "auth/auth_service.py"
    assert agentic.is_test_file("tests/test_auth_service.py") and agentic.is_test_file("src/FooTest.java")
    assert not agentic.is_test_file("auth/auth_service.py") and not agentic.is_test_file("contest/latest.py")


def test_follow_up_round_accumulates(corpus, structures):
    plan = {"queries": [{"text": "password strip", "intent": "code"}], "keywords": [], "symbols": []}
    first = retrieve_candidates(plan["queries"], plan, corpus, structures, _no_semantic)
    second = retrieve_candidates([{"text": "password strip", "intent": "code"}], plan, corpus,
                                 structures, _no_semantic, previous=first["candidates"], round_no=2)
    cid = ranked(first["candidates"])[0]["id"]
    assert second["candidates"][cid]["score"] > first["candidates"][cid]["score"]
    assert second["candidates"][cid]["first_round"] == 1


# --------------------------------------------------------------------------
# Agent 3: Relevance Critic
# --------------------------------------------------------------------------
def _cands(corpus, structures):
    plan = heuristic_plan(BUG, ISSUE)
    plan["symbols"].append("authenticate")
    return retrieve_candidates(plan["queries"], plan, corpus, structures, _no_semantic)["candidates"]


def test_critic_keeps_valid_ids_pads_and_asks_for_more(monkeypatch, corpus, structures):
    cands = _cands(corpus, structures)
    keep = next(c["id"] for c in ranked(cands) if c["filename"] == "auth/auth_service.py")
    monkeypatch.setattr(agentic, "call_llm_json", lambda *a, **k: {
        "relevant_ids": ["made/up.py::0", keep], "sufficient": False,
        "missing": "db layer", "follow_up_queries": ["get_user lookup"]})
    v = critique_candidates(BUG, ISSUE, cands, max_results=3, round_no=1, max_rounds=2)
    assert v["kept"] == [keep]
    assert v["chunks"][0]["kept_by_critic"] is True and len(v["chunks"]) == 3
    assert v["needs_more"] is True and v["follow_up_queries"] == ["get_user lookup"]

    last = critique_candidates(BUG, ISSUE, cands, max_results=3, round_no=2, max_rounds=2)
    assert last["needs_more"] is False and last["follow_up_queries"] == []


def test_critic_falls_back_to_fused_ranking(monkeypatch, corpus, structures):
    cands = _cands(corpus, structures)
    monkeypatch.setattr(agentic, "call_llm_json", _fail_llm)
    v = critique_candidates(BUG, ISSUE, cands, max_results=2, round_no=1, max_rounds=2)
    assert v["source"] == "heuristic" and v["needs_more"] is False
    assert [c["filename"] for c in v["chunks"]] == [c["filename"] for c in ranked(cands)[:2]]


def test_critic_with_no_candidates():
    v = critique_candidates(BUG, ISSUE, {}, max_results=5, round_no=1, max_rounds=2, use_llm=False)
    assert v["chunks"] == [] and v["needs_more"] is False


# --------------------------------------------------------------------------
# SFR embedding backend
# --------------------------------------------------------------------------
def test_sfr_resolution():
    r = LLMSettings(chat_provider="groq", chat_model="m", chat_api_key="k",
                    embed_provider="sfr").resolved_embedding()
    assert (r.provider, r.model, r.api_key) == ("sfr", SFR_MODEL_ID, "")


@pytest.mark.skipif(not sfr_available(), reason="SFR extras not installed (requirements-sfr.txt)")
def test_sfr_embeds_code_meaningfully():
    """Real model: 1024-dim unit vectors, and a code query lands nearer the
    matching function than an unrelated one. Downloads ~870 MB on first run."""
    from agents.providers import _sfr_embed
    q, good, bad = _sfr_embed([
        "function that strips the password before checking it",
        "def authenticate(username, password):\n    password = password.strip()",
        "def get_user(username):\n    return USERS.get(username)",
    ])
    assert len(q) == 1024
    assert abs(sum(x * x for x in q) - 1.0) < 1e-3
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731
    assert dot(q, good) > dot(q, bad)


def test_friendly_provider_errors():
    from agents.llm_client import friendly_error
    assert "rejected" in friendly_error("400 INVALID_ARGUMENT. {'reason': 'API_KEY_INVALID'}")
    assert "per minute" in friendly_error("429 RESOURCE_EXHAUSTED")
    assert "overloaded" in friendly_error("503 UNAVAILABLE. high demand")
    assert "not found" in friendly_error("404 NOT_FOUND models/foo is not found")
    assert friendly_error("error on line 4291 of parser") is None
