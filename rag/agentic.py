"""
agentic.py — multi-agent RAG for code retrieval.

Single-shot RAG embeds the raw bug report and takes the top-k vectors. That
misses code worded differently from the report, and it cannot tell when what
it found is not enough to diagnose the bug. Here retrieval is a small team of
agents, each with one job, run as nodes of the LangGraph (graph/nodes.py):

  1. Query Planner (LLM)     turns the bug report + triaged issue into several
                             targeted searches (symptom wording, code-like
                             wording, concepts) plus exact keywords and likely
                             symbol names.
  2. Retriever agents        run every planned search through three lenses,
     (tools, no LLM)         because each catches what the others miss:
       - Semantic   vector search with the session's embedding model
                    (SFR-Embedding-Code-400M when installed).
       - Lexical    BM25 over code-aware tokens (snake_case / camelCase split):
                    strong on exact identifiers and error strings.
       - Structural symbol lookup in the AST call graph plus one hop to
                    callers / callees: finds code that runs together.
  3. Fusion                  Reciprocal Rank Fusion merges every ranked list;
                             a chunk found by several agents rises.
  4. Relevance Critic (LLM)  reads the top candidates, keeps the ones that bear
                             on the bug, and decides whether the context is
                             sufficient. If not, it writes follow-up searches
                             and the graph loops back to the retrievers
                             (bounded by max rounds).

Every LLM step has a deterministic fallback, so a rate-limited or failing model
lowers retrieval quality instead of aborting the run. "fast" mode skips both
LLM agents.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from typing import Callable

from agents.llm_client import call_llm_json

RRF_K = 60                 # standard Reciprocal Rank Fusion constant
MAX_QUERIES = 5
MAX_KEYWORDS = 10
MAX_SYMBOLS = 8
MAX_FOLLOW_UPS = 3
PER_LIST = 8               # results each retriever returns per search
CRITIC_CANDIDATES = 10     # how many fused candidates the critic reads
# Characters of each candidate shown to the critic. Chunks are ~800 chars
# (rag/chunker.py), so this shows them whole: a truncated chunk once hid the
# very function the critic was looking for and made it ask for a wasted round.
CRITIC_SNIPPET = 2000

AGENTS = ("semantic", "lexical", "structural")

_STOP = set(
    "the a an and or of to in on for is are was were be been being with when then this that "
    "it its as at by from not no but if into than too very can could should would will just "
    "only also there their they you your we our after before while does did do has have had "
    "get gets got using use used page".split()
)


# --------------------------------------------------------------------------
# Tokenising + BM25 (lexical retriever)
# --------------------------------------------------------------------------
def code_tokens(text: str) -> list[str]:
    """Tokens that match how code is written: identifiers are kept whole
    (lower-cased) and also split into their snake_case / camelCase parts, so
    'checkLogin' matches 'check_login', 'login' and 'check'."""
    out: list[str] = []
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text or ""):
        parts = [p for p in re.split(r"_+|(?<=[a-z0-9])(?=[A-Z])", word) if p]
        if len(parts) > 1:
            out.append(word.lower())
        for p in parts:
            p = p.lower()
            if len(p) > 1 and p not in _STOP:
                out.append(p)
    return out


class BM25:
    """Okapi BM25 over pre-tokenised documents. Small and dependency-free;
    repos here are thousands of chunks at most."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.dl = [len(d) for d in docs]
        n = len(docs)
        self.avgdl = (sum(self.dl) / n) if n else 1.0
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        q = [t for t in dict.fromkeys(query) if t in self.idf]
        out = []
        for tf, dl in zip(self.tf, self.dl):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


# --------------------------------------------------------------------------
# Agent 1: Query Planner
# --------------------------------------------------------------------------
PLANNER_SYSTEM = (
    "You are the Query Planner in a team of code-retrieval agents. Other agents will run your "
    "searches against an unfamiliar repository with semantic (embedding) search, keyword (BM25) "
    "search, and a symbol index. Plan searches that together are likely to surface the code "
    "responsible for the bug. You have not seen the code; do not invent facts about it."
)

PLANNER_PROMPT = """Bug report: "{bug}"

Triage notes from an earlier agent (guesses, not facts):
{issue}

Return ONLY valid JSON, no markdown fences:
{{
  "queries": [{{"text": string, "intent": "symptom" | "code" | "concept"}}],
  "keywords": string[],
  "symbols": string[]
}}
Rules:
- 3 to 5 queries. Include at least one "code" query phrased like the code itself
  (for example "if password is None return False") and one "symptom" query.
- keywords: up to 10 exact identifiers, error messages or literals likely to appear in code.
- symbols: up to 8 likely function, method or class names."""


def _clean_list(value, limit: int, max_len: int = 200) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for v in value:
        if isinstance(v, str) and v.strip():
            out.append(v.strip()[:max_len])
    return list(dict.fromkeys(out))[:limit]


def _file_stems(issue: dict) -> list[str]:
    return [os.path.splitext(os.path.basename(f))[0] for f in (issue.get("possible_files") or [])
            if isinstance(f, str) and f.strip()]


def heuristic_plan(bug_report: str, issue: dict | None) -> dict:
    """Deterministic plan used in fast mode and whenever the planner fails."""
    issue = issue or {}
    queries = [{"text": bug_report.strip(), "intent": "symptom"}]
    concept = " ".join(str(issue.get(k, "")) for k in ("suspected_area", "root_problem_category")).strip()
    if concept:
        queries.append({"text": concept, "intent": "concept"})
    areas = " ".join(str(a) for a in (issue.get("investigation_areas") or []))
    if areas.strip():
        queries.append({"text": areas.strip(), "intent": "concept"})

    quoted = re.findall(r"[`'\"]([^`'\"]{2,60})[`'\"]", bug_report)
    idents = [w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", bug_report)
              if "_" in w or re.search(r"[a-z][A-Z]", w)]
    stems = _file_stems(issue)
    return {
        "queries": _dedupe_queries(queries)[:MAX_QUERIES],
        "keywords": list(dict.fromkeys(quoted + idents + stems))[:MAX_KEYWORDS],
        "symbols": list(dict.fromkeys(idents + stems))[:MAX_SYMBOLS],
    }


def _dedupe_queries(queries: list[dict]) -> list[dict]:
    seen, out = set(), []
    for q in queries:
        key = q["text"].lower()
        if q["text"] and key not in seen:
            seen.add(key)
            out.append(q)
    return out


def plan_retrieval(bug_report: str, issue: dict | None, use_llm: bool = True) -> dict:
    """Returns {"queries": [{"text","intent"}], "keywords", "symbols",
    "source": "llm" | "heuristic", "fallback_reason"?}."""
    fallback = heuristic_plan(bug_report, issue)
    if not use_llm:
        return {**fallback, "source": "heuristic"}
    try:
        raw = call_llm_json(
            PLANNER_PROMPT.format(bug=bug_report.strip(), issue=json.dumps(issue or {}, indent=2)),
            system_instruction=PLANNER_SYSTEM, max_json_retries=1,
        )
        queries = []
        for q in raw.get("queries") or []:
            if isinstance(q, dict) and isinstance(q.get("text"), str) and q["text"].strip():
                intent = q.get("intent") if q.get("intent") in ("symptom", "code", "concept") else "concept"
                queries.append({"text": q["text"].strip()[:300], "intent": intent})
            elif isinstance(q, str) and q.strip():
                queries.append({"text": q.strip()[:300], "intent": "concept"})
        if not queries:
            raise ValueError("planner returned no queries")
        # The raw report is always searched too: the planner refines, it
        # never replaces what the user actually wrote.
        queries = _dedupe_queries([{"text": bug_report.strip(), "intent": "symptom"}] + queries)
        return {
            "queries": queries[:MAX_QUERIES + 1],
            "keywords": _clean_list(raw.get("keywords"), MAX_KEYWORDS, 80) or fallback["keywords"],
            "symbols": _clean_list(raw.get("symbols"), MAX_SYMBOLS, 80) or fallback["symbols"],
            "source": "llm",
        }
    except Exception as e:  # noqa: BLE001 — any planner failure falls back
        return {**fallback, "source": "heuristic", "fallback_reason": f"{type(e).__name__}: {str(e)[:160]}"}


# --------------------------------------------------------------------------
# Agent 2: Retrievers (semantic, lexical, structural) + fusion
# --------------------------------------------------------------------------
def chunk_id(chunk: dict) -> str:
    return f"{chunk['filename']}::{chunk['chunk_index']}"


# Test files repeat the identifiers under test many times, so keyword and
# vector search over-rank them versus the code that actually holds the bug.
# They stay retrievable (expected behaviour is useful context) but weigh less.
TEST_FILE_WEIGHT = 0.6


def is_test_file(filename: str) -> bool:
    """tests/x.py, test_x.py, x_test.go, x.spec.ts, FooTest.java, FooTests.cs.
    Not files that merely contain the letters (latest.py, contest/)."""
    parts = filename.replace("\\", "/").split("/")
    base = parts[-1]
    if any(p.lower() in ("test", "tests", "__tests__", "spec") for p in parts[:-1]):
        return True
    if base.lower().startswith("test_") or re.search(r"[_.\-](test|tests|spec)\.\w+$", base.lower()):
        return True
    return re.search(r"[a-z0-9](Test|Tests|Spec)\.\w+$", base) is not None


def _definition_chunk(corpus_by_file: dict, filename: str, name: str) -> dict | None:
    """The chunk of `filename` that defines `name` (or, failing that, mentions
    it as a call)."""
    chunks = corpus_by_file.get(filename) or []
    defn = re.compile(rf"(?:def|function|class|void|[\w<>\[\],]+)\s+{re.escape(name)}\s*\(")
    for c in chunks:
        if defn.search(c["text"]):
            return c
    call = re.compile(rf"\b{re.escape(name)}\s*\(")
    for c in chunks:
        if call.search(c["text"]):
            return c
    return None


def _structural_hits(terms: list[str], structures: dict, corpus_by_file: dict) -> list[tuple[dict, float, str]]:
    """Ranks function definitions against the planned symbols/keywords, then
    adds one hop of callers and callees. Returns (chunk, score, reason)."""
    term_parts = {t.lower(): set(code_tokens(t)) for t in terms if t}
    if not term_parts:
        return []

    defs: dict[str, list[str]] = {}
    for fname, info in structures.items():
        for func in (info or {}).get("functions", {}):
            defs.setdefault(func, []).append(fname)

    scored: dict[tuple[str, str], tuple[float, str]] = {}

    def bump(fname: str, func: str, score: float, reason: str) -> None:
        key = (fname, func)
        if score > scored.get(key, (0.0, ""))[0]:
            scored[key] = (score, reason)

    for func, files in defs.items():
        f_low, f_parts = func.lower(), set(code_tokens(func))
        best, why = 0.0, ""
        for term, parts in term_parts.items():
            if f_low == term:
                s = 3.0
            elif len(term) >= 4 and (term in f_low or f_low in term):
                s = 2.0
            elif parts and f_parts:
                overlap = len(parts & f_parts) / len(f_parts)
                s = 1.5 * overlap if overlap >= 0.5 else 0.0
            else:
                s = 0.0
            if s > best:
                best, why = s, term
        if best > 0:
            for fname in files:
                bump(fname, func, best, f"symbol match '{why}'")

    # One hop through the call graph: callees of matched functions and their callers.
    matched = {func for (_, func) in scored}
    for fname, info in structures.items():
        for func, meta in (info or {}).get("functions", {}).items():
            calls = set(meta.get("calls") or [])
            if func in matched:
                for callee in calls & set(defs):
                    for cf in defs[callee]:
                        bump(cf, callee, 0.75, f"called by {func}()")
            for m in calls & matched:
                bump(fname, func, 0.6, f"calls {m}()")

    hits = []
    for (fname, func), (score, reason) in scored.items():
        chunk = _definition_chunk(corpus_by_file, fname, func)
        if chunk:
            hits.append((chunk, score, reason))
    hits.sort(key=lambda h: h[1], reverse=True)

    out, seen = [], set()
    for chunk, score, reason in hits:
        cid = chunk_id(chunk)
        if cid not in seen:
            seen.add(cid)
            out.append((chunk, score, reason))
    return out[:PER_LIST * 2]


def retrieve_candidates(
    queries: list[dict],
    plan: dict,
    corpus: list[dict],
    structures: dict,
    semantic_search: Callable[[str, int], list[dict]],
    previous: dict | None = None,
    round_no: int = 1,
) -> dict:
    """Runs the three retrievers for every query and fuses the ranked lists
    with Reciprocal Rank Fusion, accumulating onto `previous` candidates (so a
    chunk found again in a follow-up round rises).

    Returns {"candidates": {id: candidate}, "found": {agent: n_unique},
    "errors": {agent: message}}."""
    cands: dict[str, dict] = {k: {**v, "sources": dict(v.get("sources", {})),
                                  "reasons": list(v.get("reasons", []))}
                              for k, v in (previous or {}).items()}
    found = {a: set() for a in AGENTS}
    errors: dict[str, str] = {}

    def add(chunk: dict, agent: str, rank: int, reason: str = "", distance: float | None = None,
            weight: float = 1.0) -> None:
        cid = chunk_id(chunk)
        c = cands.get(cid)
        if c is None:
            c = cands[cid] = {"id": cid, "text": chunk["text"], "filename": chunk["filename"],
                              "chunk_index": chunk["chunk_index"], "score": 0.0, "distance": None,
                              "sources": {}, "reasons": [], "first_round": round_no}
        if is_test_file(chunk["filename"]):
            weight *= TEST_FILE_WEIGHT
        c["score"] += weight / (RRF_K + rank)
        c["sources"][agent] = min(rank, c["sources"].get(agent, rank))
        if distance is not None and (c["distance"] is None or distance < c["distance"]):
            c["distance"] = float(distance)
        if reason and reason not in c["reasons"] and len(c["reasons"]) < 4:
            c["reasons"].append(reason)
        found[agent].add(cid)

    # Semantic retriever.
    try:
        for q in queries:
            for rank, hit in enumerate(semantic_search(q["text"], PER_LIST), 1):
                add(hit, "semantic", rank, f"semantic: \"{q['text'][:60]}\"", hit.get("distance"))
    except Exception as e:  # noqa: BLE001 — one retriever failing must not sink the others
        errors["semantic"] = f"{type(e).__name__}: {str(e)[:160]}"

    # Lexical retriever (BM25 over code tokens).
    try:
        if corpus:
            bm25 = BM25([code_tokens(c["text"]) for c in corpus])
            kw_tokens = code_tokens(" ".join(plan.get("keywords") or []))
            for q in queries:
                scores = bm25.scores(code_tokens(q["text"]) + kw_tokens)
                ranked = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: scores[i], reverse=True)
                for rank, i in enumerate(ranked[:PER_LIST], 1):
                    add(corpus[i], "lexical", rank, f"keywords: \"{q['text'][:60]}\"")
    except Exception as e:  # noqa: BLE001
        errors["lexical"] = f"{type(e).__name__}: {str(e)[:160]}"

    # Structural retriever (symbol index + call graph), one ranked list per round.
    try:
        by_file: dict[str, list[dict]] = {}
        for c in corpus:
            by_file.setdefault(c["filename"], []).append(c)
        for lst in by_file.values():
            lst.sort(key=lambda c: c["chunk_index"])
        terms = list(plan.get("symbols") or []) + list(plan.get("keywords") or [])
        for q in queries:
            if q.get("intent") == "code":
                terms += re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", q["text"])
        for rank, (chunk, _score, reason) in enumerate(_structural_hits(terms, structures, by_file), 1):
            add(chunk, "structural", rank, reason, weight=1.5)
    except Exception as e:  # noqa: BLE001
        errors["structural"] = f"{type(e).__name__}: {str(e)[:160]}"

    return {"candidates": cands, "found": {a: len(s) for a, s in found.items()}, "errors": errors}


def ranked(candidates: dict) -> list[dict]:
    """Fused order: RRF score, then number of agents that agree."""
    return sorted(candidates.values(), key=lambda c: (c["score"], len(c["sources"])), reverse=True)


# --------------------------------------------------------------------------
# Agent 3: Relevance Critic
# --------------------------------------------------------------------------
CRITIC_SYSTEM = (
    "You are the Relevance Critic in a team of code-retrieval agents. You judge retrieved code "
    "strictly: a chunk is relevant only if it helps locate or fix the bug's root cause (the faulty "
    "logic itself, or code that directly feeds it). Tests and unrelated helpers are not relevant "
    "unless they reveal the expected behaviour."
)

CRITIC_PROMPT = """Bug report: "{bug}"
Triage: {triage}

Candidate code chunks from the search agents:
{candidates}

Decide:
1. relevant_ids: ids of chunks that help diagnose or fix this bug, most relevant first.
2. sufficient: true if these chunks include the function where the faulty logic most
   likely lives (not only its callers or tests).
3. If not sufficient: "missing" (what code is still needed) and up to 3 follow_up_queries
   phrased like the code or concept you need.

Return ONLY valid JSON, no markdown fences:
{{"relevant_ids": string[], "sufficient": boolean, "missing": string, "follow_up_queries": string[]}}"""


def _format_candidates(cands: list[dict]) -> str:
    blocks = []
    for c in cands:
        snippet = c["text"][:CRITIC_SNIPPET]
        blocks.append(f"[id: {c['id']}] found by: {', '.join(sorted(c['sources']))}\n{snippet}")
    return "\n\n".join(blocks)


def critique_candidates(
    bug_report: str,
    issue: dict | None,
    candidates: dict,
    max_results: int,
    round_no: int,
    max_rounds: int,
    use_llm: bool = True,
    previous_kept: list[str] | None = None,
    new_found: int | None = None,
) -> dict:
    """Selects the final context. Returns {"chunks": [...], "sufficient",
    "needs_more", "follow_up_queries", "missing", "kept", "source",
    "fallback_reason"?, "skipped"?}.

    `new_found` is how many candidates the latest retrieval round added. When a
    follow-up round adds nothing, re-asking the critic cannot change the answer,
    so the previous verdict is reused without an LLM call."""
    order = ranked(candidates)
    top = order[:CRITIC_CANDIDATES]

    def select(relevant_ids: list[str]) -> list[dict]:
        by_id = {c["id"]: c for c in order}
        chosen = [by_id[i] for i in relevant_ids if i in by_id]
        # Pad with the next-best fused candidates so the Root Cause Agent
        # still has context if the critic was too strict.
        for c in order:
            if len(chosen) >= max_results:
                break
            if c not in chosen:
                chosen.append(c)
        kept = set(relevant_ids)
        return [_as_rag_chunk(c, c["id"] in kept) for c in chosen[:max_results]]

    base = {"follow_up_queries": [], "missing": "", "needs_more": False}
    if not top:
        return {**base, "chunks": [], "sufficient": False, "kept": [], "source": "heuristic"}
    if not use_llm:
        return {**base, "chunks": select([]), "sufficient": True, "kept": [], "source": "heuristic"}
    if round_no > 1 and new_found == 0:
        kept = [i for i in (previous_kept or []) if i in candidates]
        return {**base, "chunks": select(kept), "sufficient": False, "kept": kept, "source": "llm",
                "skipped": True, "missing": "Follow-up searches found no new code."}

    triage =json.dumps({k: (issue or {}).get(k) for k in ("suspected_area", "root_problem_category")})
    try:
        raw = call_llm_json(
            CRITIC_PROMPT.format(bug=bug_report.strip(), triage=triage, candidates=_format_candidates(top)),
            system_instruction=CRITIC_SYSTEM, max_json_retries=1,
        )
        valid = {c["id"] for c in top}
        relevant = [i for i in _clean_list(raw.get("relevant_ids"), CRITIC_CANDIDATES, 300) if i in valid]
        sufficient = bool(raw.get("sufficient")) if relevant else False
        follow_ups = _clean_list(raw.get("follow_up_queries"), MAX_FOLLOW_UPS, 300)
        needs_more = (not sufficient) and round_no < max_rounds and bool(follow_ups)
        return {
            "chunks": select(relevant),
            "sufficient": sufficient,
            "needs_more": needs_more,
            "follow_up_queries": follow_ups if needs_more else [],
            "missing": str(raw.get("missing") or "")[:300],
            "kept": relevant,
            "source": "llm",
        }
    except Exception as e:  # noqa: BLE001 — critic failure keeps the fused ranking
        return {**base, "chunks": select([]), "sufficient": True, "kept": [], "source": "heuristic",
                "fallback_reason": f"{type(e).__name__}: {str(e)[:160]}"}


def _as_rag_chunk(c: dict, kept_by_critic: bool) -> dict:
    """The shape downstream agents consume (filename/chunk_index/text), plus
    provenance for the explainability report."""
    return {
        "text": c["text"],
        "filename": c["filename"],
        "chunk_index": c["chunk_index"],
        "distance": c["distance"],
        "score": round(c["score"], 5),
        "sources": sorted(c["sources"]),
        "reasons": list(c.get("reasons", [])),
        "kept_by_critic": kept_by_critic,
    }
