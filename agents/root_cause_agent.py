"""
root_cause_agent.py — Agent 4, the most important agent in the pipeline.

Every upstream agent (Issue Understanding, Repository Indexing, Code
Navigation) exists purely to feed this agent better context. Every
downstream agent (Patch Generator, Validation, PR Agent) is only as
good as what this agent decides the bug actually is.

This agent combines three distinct sources of evidence:
  1. issue_report      — the human's bug description, already parsed
                          into structured JSON by the Issue Agent.
  2. call_graph_trace   — GROUND TRUTH. Actual function calls extracted
                          via AST parsing (analyzers/call_graph.py).
                          Trust this over semantic guesses when they
                          conflict.
  3. rag_chunks         — semantically similar code chunks retrieved
                          from ChromaDB (rag/vector_store.py). May
                          include false positives — verify relevance
                          before treating as fact.

The prompt explicitly tells the model to weigh these sources
differently, because without that instruction an LLM has no principled
way to resolve disagreement between "this sounds related" (RAG) and
"this is what actually executes" (call graph) — it'll just pick
whichever sounds more confident, which is exactly the failure mode we
want to avoid in the most load-bearing agent in the system.

Phase 1 scope (deliberately linear, no retry loop yet): this agent
runs once per investigation and returns whatever confidence it has.
A retry loop that re-queries RAG/call-graph on low confidence is a
clean Phase 2 addition — documented here, not built yet.
"""

import json
from agents.llm_client import call_llm_json

SYSTEM_INSTRUCTION = """You are a Staff Software Engineer investigating a bug. \
You will be given three sources of evidence with different reliability \
levels. The call graph trace is GROUND TRUTH, extracted directly from the \
source code via static analysis — trust it over guesses. The RAG chunks \
are retrieved by semantic similarity to the bug description and may \
include false positives that merely sound related without being part of \
the actual execution path — verify relevance before relying on them. Be \
honest about your confidence; if the evidence is thin or contradictory, \
say so rather than guessing confidently."""

PROMPT_TEMPLATE = """Bug report (parsed):
{issue_report}

Actual execution path (verified via static analysis — trust this over
semantic guesses below):
{call_graph_section}

Potentially related code (retrieved by semantic search — may include
false positives, verify relevance before using):
{rag_section}

Identify:
1. The exact root cause — one specific, falsifiable claim about what is
   wrong in the code (not a restatement of the symptom).
2. The exact file and function where it occurs.
3. Step-by-step reasoning that traces from the bug symptom to the cause,
   citing specific evidence from the execution path or code chunks above.
4. A confidence score from 0.0 to 1.0. Be honest — a confident-sounding
   guess with thin evidence should score low, not high.

Return ONLY valid JSON matching this exact schema, no markdown fences, no
commentary:
{{
  "root_cause": string,
  "confidence": float,
  "affected_files": string[],
  "affected_function": string,
  "reasoning": string[]
}}"""

REQUIRED_KEYS = {"root_cause", "confidence", "affected_files", "affected_function", "reasoning"}


def _format_call_graph_section(call_graph_trace: dict | None) -> str:
    if not call_graph_trace or not call_graph_trace.get("execution_path"):
        return "(no call graph trace available)"

    lines = []
    for step in call_graph_trace["execution_path"]:
        lines.append(
            f"- {step['file']} :: {step['function']}() [line {step['line']}] "
            f"calls: {step['calls_into']}"
        )
    return "\n".join(lines)


def _format_rag_section(rag_chunks: list | None) -> str:
    if not rag_chunks:
        return "(no related chunks retrieved)"

    lines = []
    for chunk in rag_chunks:
        lines.append(
            f"--- {chunk['filename']} [chunk {chunk['chunk_index']}] "
            f"(similarity distance: {chunk['distance']:.4f}) ---\n{chunk['text']}"
        )
    return "\n\n".join(lines)


def analyze_root_cause(
    issue_report: dict,
    call_graph_trace: dict | None = None,
    rag_chunks: list | None = None,
) -> dict:
    """
    Produces a structured root cause diagnosis from combined evidence.

    Args:
        issue_report: output of agents.issue_agent.understand_issue().
        call_graph_trace: output of analyzers.call_graph.trace_call_path().
            Pass None if navigation hasn't run yet — the agent will
            note the absence rather than fail, though accuracy will
            be lower without it.
        rag_chunks: output of rag.vector_store.query_collection().
            Same as above — optional but recommended.

    Returns:
        A dict matching REQUIRED_KEYS. Raises ValueError if the model's
        output is missing keys or confidence is out of range — failing
        loudly here is better than letting a malformed diagnosis
        silently corrupt the Patch Generator downstream.
    """
    prompt = PROMPT_TEMPLATE.format(
        issue_report=json.dumps(issue_report, indent=2),
        call_graph_section=_format_call_graph_section(call_graph_trace),
        rag_section=_format_rag_section(rag_chunks),
    )

    try:
        result = call_llm_json(prompt, system_instruction=SYSTEM_INSTRUCTION)
    except json.JSONDecodeError as e:
        raise ValueError(f"Root Cause Agent: model did not return valid JSON: {e}")

    missing = REQUIRED_KEYS - result.keys()
    if missing:
        raise ValueError(f"Root Cause Agent: model output missing keys: {missing}")

    if not isinstance(result["confidence"], (int, float)) or not (0.0 <= result["confidence"] <= 1.0):
        raise ValueError(
            f"Root Cause Agent: confidence must be a number between 0.0 and 1.0, "
            f"got {result['confidence']!r}"
        )

    if not isinstance(result["affected_files"], list):
        raise ValueError("Root Cause Agent: affected_files must be a list")

    if not isinstance(result["reasoning"], list):
        raise ValueError("Root Cause Agent: reasoning must be a list")

    return result


if __name__ == "__main__":
    # Quick manual test — run with: python -m agents.root_cause_agent
    # Wires together everything built so far: Issue Agent -> Call Graph
    # -> RAG -> Root Cause Agent, against the sample repo's real bug.
    from agents.issue_agent import understand_issue
    from analyzers.call_graph import build_repo_structure, trace_call_path
    from rag.vector_store import get_or_build_collection, query_collection
    import subprocess

    bug_report = "Login page crashes when password field is empty"

    print(f"Bug report: {bug_report}\n")

    print("--- Step 1: Issue Agent ---")
    issue = understand_issue(bug_report)
    print(json.dumps(issue, indent=2))

    print("\n--- Step 2: Code Navigation (call graph) ---")
    structures = build_repo_structure("sample-repo")
    # Seed from login.py directly, since that's the real entry point —
    # in the full pipeline this seed would come from issue["possible_files"]
    # intersected with files that actually exist in the repo.
    trace = trace_call_path(structures, seed_files=["login.py"], max_depth=3)
    print(f"Files visited: {trace['files_visited']}")

    print("\n--- Step 3: RAG retrieval ---")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd="sample-repo",
        capture_output=True, text=True
    ).stdout.strip()
    collection = get_or_build_collection("sample-repo", "local-sample-repo", commit)
    rag_results = query_collection(collection, bug_report, n_results=5)
    for r in rag_results:
        print(f"  {r['filename']} chunk {r['chunk_index']} (distance={r['distance']:.4f})")

    print("\n--- Step 4: Root Cause Analysis ---")
    diagnosis = analyze_root_cause(issue, call_graph_trace=trace, rag_chunks=rag_results)
    print(json.dumps(diagnosis, indent=2))
