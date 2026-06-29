"""
issue_agent.py — Agent 1 in the pipeline.

Takes a raw, human-written bug report and turns it into structured JSON
that every downstream agent (Repository Indexing, Code Navigation, Root
Cause Analysis) can rely on having a consistent shape.

This agent has NO access to the actual repository. It works purely off
the text the user typed. `possible_files` and `suspected_area` here are
educated guesses based on naming conventions in the bug description —
not facts. Downstream agents (especially Code Navigation) exist
specifically to replace these guesses with ground truth once the repo
is available.

Do not add code-reading logic to this agent. If it starts reasoning
about actual file contents, that responsibility belongs in a different
agent — keeping this agent pure-language-only is what keeps its output
fast, cheap, and predictable.
"""

import json
from agents.llm_client import call_llm_json

SYSTEM_INSTRUCTION = """You are a senior software engineer triaging a bug \
report before any code investigation begins. You only have the bug \
description text below — you have not seen the actual codebase. Make your \
best inference from the description alone, and be honest that file names \
are guesses based on common naming conventions, not confirmed facts."""

PROMPT_TEMPLATE = """Bug report: "{bug_report}"

Identify:
1. Affected module/component (best guess from the description)
2. Likely files (based on naming patterns — these are guesses, not facts)
3. Root problem category (e.g. null handling, race condition, off-by-one,
   auth/permissions, state management, network/timeout)
4. Required investigation areas (what should be checked in the codebase)
5. Severity (low / medium / high / critical) based on user-facing impact

Return ONLY valid JSON matching this exact schema, no markdown fences, no
commentary:
{{
  "module": string,
  "severity": "low" | "medium" | "high" | "critical",
  "suspected_area": string,
  "possible_files": string[],
  "root_problem_category": string,
  "investigation_areas": string[]
}}"""

REQUIRED_KEYS = {
    "module",
    "severity",
    "suspected_area",
    "possible_files",
    "root_problem_category",
    "investigation_areas",
}

VALID_SEVERITIES = {"low", "medium", "high", "critical"}


def understand_issue(bug_report: str) -> dict:
    """
    Parses a raw bug report into structured JSON.

    Args:
        bug_report: the raw text a user typed describing the bug.

    Returns:
        A dict matching REQUIRED_KEYS. Raises ValueError if the model's
        output is missing keys or has an invalid severity — better to fail
        loudly here than let a malformed issue silently corrupt every
        downstream agent.
    """
    if not bug_report or not bug_report.strip():
        raise ValueError("bug_report cannot be empty")

    prompt = PROMPT_TEMPLATE.format(bug_report=bug_report.strip())

    try:
        result = call_llm_json(prompt, system_instruction=SYSTEM_INSTRUCTION)
    except json.JSONDecodeError as e:
        raise ValueError(f"Issue Agent: model did not return valid JSON: {e}")

    missing = REQUIRED_KEYS - result.keys()
    if missing:
        raise ValueError(f"Issue Agent: model output missing keys: {missing}")

    if result["severity"] not in VALID_SEVERITIES:
        raise ValueError(
            f"Issue Agent: invalid severity '{result['severity']}', "
            f"expected one of {VALID_SEVERITIES}"
        )

    if not isinstance(result["possible_files"], list):
        raise ValueError("Issue Agent: possible_files must be a list")

    if not isinstance(result["investigation_areas"], list):
        raise ValueError("Issue Agent: investigation_areas must be a list")

    return result


if __name__ == "__main__":
    # Quick manual test — run with: python -m agents.issue_agent
    sample_bug = "Login page crashes when password field is empty"
    print(f"Bug report: {sample_bug}\n")
    output = understand_issue(sample_bug)
    print(json.dumps(output, indent=2))
