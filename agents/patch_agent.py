"""
patch_agent.py — Agent 5 in the pipeline.

Takes a Root Cause Agent diagnosis and produces a minimal unified diff
that fixes the actual root cause.

DESIGN NOTE — why this asks the model for full file content, not a diff:
Early versions of this agent asked the model to hand-write a unified
diff directly, including the `@@ -start,count +start,count @@` hunk
header. In practice the model reliably gets the *content* of a fix
right but unreliably gets the *line numbers* in that header right —
it tends to count lines approximately rather than precisely, which
produces a diff that `git apply` rejects outright even though the
actual code change inside it is correct.

The fix: ask the model for the corrected FULL file content instead
(still under the same minimal-change rules), then compute the actual
diff ourselves with Python's `difflib`, which counts real line numbers
by construction — there's no way for it to be wrong, since it's
diffing two real strings rather than asking an LLM to count.

This agent is given the EXACT, verbatim content of affected files —
not a paraphrase or summary — because the model needs to reproduce
every untouched line byte-for-byte for the diff to stay minimal.

Guardrails enforced via the prompt (cannot be enforced by code alone,
since the model still controls what it writes inside the file):
  1. Minimal change only — preserve every line except what's necessary
     to fix the root cause. No refactoring, renaming, or "cleanup."
  2. Match existing code style exactly.
  3. No new imports, libraries, or dependencies.
  4. Fix the root cause, not the symptom.
  5. Only modify files in affected_files — never touch other files.

What this module enforces in code:
  - The resulting diff actually applies cleanly (validate_diff_applies).
    This is "validation layer zero" — no point running flake8/pytest on
    a diff that never touched the file.
  - A line-change-count guard (_check_change_scope) that flags if the
    model changed dramatically more lines than expected, as a cheap
    signal that it may have ignored the "minimal change" instruction.
"""

import os
import re
import json
import difflib
import subprocess
import tempfile
from agents.llm_client import call_llm

SYSTEM_INSTRUCTION = """You are a senior software engineer making a surgical \
fix. You will be shown a diagnosed root cause and the exact, verbatim \
content of the affected file. Follow the rules below exactly — they exist \
because automated patches that violate them tend to break review, break \
unrelated functionality, or mask bugs instead of fixing them."""

PROMPT_TEMPLATE = """Root cause diagnosis:
{root_cause_json}

File: {file_path}
Exact current content (verbatim, with line numbers added for your
reference only — do NOT include line numbers in your output):
{numbered_file_content}

Rules — follow exactly, no exceptions:
1. Change only what is necessary to fix the root cause. Every other line
   must be reproduced EXACTLY as shown above — do not refactor, rename,
   reformat, or "improve" anything else.
2. Match the existing code style exactly (indentation, quotes, naming,
   error-handling pattern already used in this file).
3. Do not introduce new imports, libraries, or dependencies.
4. Fix the actual root cause — do not mask the symptom (e.g. do not wrap
   in try/except to suppress a crash without fixing the underlying issue).
5. Output the ENTIRE file content with your fix applied — not a diff,
   not a snippet, the complete file from the first line to the last.
6. Output ONLY the raw file content. No markdown code fences, no line
   numbers, no explanation, no commentary before or after.

Complete corrected file content:"""

# Prepended to the normal prompt ONLY on retry attempts. The orchestrator
# (graph/workflow.py) feeds this in after a patch failed validation. Per
# the planning discussion, a retry that just re-runs the same prompt tends
# to reproduce the same broken patch — the model needs to see (a) the diff
# it already tried and (b) the *actual* pytest tracebacks, not just "it
# failed", to have any chance of doing better.
#
# IMPORTANT: this prompt must NEVER invite the model to write prose into the
# file. An earlier version asked it to "state at the top, comment-free, if
# the diagnosis was wrong" — the model dutifully wrote an English sentence
# as the first line of the .py file, producing a SyntaxError on every retry.
# The output must always be valid file content and nothing else.
RETRY_PROMPT_PREFIX = """You are REVISING a previous fix that FAILED validation.
This is attempt {attempt} of {max_attempts}.

Your previous patch (which did NOT resolve the issue):
```diff
{previous_diff}
```

What went wrong when it was validated:
{failure_detail}

Carefully analyze why the previous patch did not work — a common cause is
introducing a syntax error (e.g. an unbalanced bracket, or writing a note
or explanation as a line of code instead of valid syntax). Then produce a
corrected version of the file that makes the failing test(s) pass.

Your entire output must be valid, runnable file content — every single line
must be legal code or an existing comment. Do NOT write any explanation,
note, or prose anywhere in the file. Follow all the rules below exactly.

---

"""


def _format_failure_detail(retry_context: dict) -> str:
    """Turns the orchestrator's failure data into readable text for the
    retry prompt. Covers every way a patch attempt can fail: the diff not
    applying, a test-run error (e.g. the patch broke collection with a
    syntax error), and ordinary test failures with their tracebacks."""
    parts = []

    if retry_context.get("apply_error"):
        parts.append(
            f"The diff could not be applied to the working tree:\n{retry_context['apply_error']}"
        )

    if retry_context.get("test_error"):
        parts.append(f"The test run reported an error:\n{retry_context['test_error']}")

    for t in retry_context.get("failed_tests") or []:
        parts.append(f"- {t.get('test_name', '<unknown test>')}:\n{t.get('error_message', '(no detail)')}")

    return "\n\n".join(parts) if parts else "(no specific failure detail was captured)"


def _strip_markdown_fences(text: str) -> str:
    """
    Models sometimes wrap output in ``` fences despite instructions not
    to. Strip defensively rather than relying purely on the prompt.
    """
    text = text.strip()
    text = re.sub(r"^```(?:\w+)?\n", "", text)
    text = re.sub(r"\n```$", "", text)
    return text.strip("\n")


def _strip_line_number_prefixes(text: str) -> str:
    """
    Defensive cleanup for a real failure mode: the prompt feeds the file
    with '  N| ' line-number prefixes "for reference only" and explicitly
    tells the model NOT to echo them back — but models (observed with
    gemini-2.5-flash) sometimes do anyway. If that output is used verbatim,
    literal 'N| ' text gets written into the file, producing a syntactically
    broken patch that then fails test collection.

    If essentially every non-empty line carries the prefix, strip it. Gated
    on a high match ratio (0.8) so we never mangle a genuine file that just
    happens to contain a line or two shaped like 'N| ...'.
    """
    lines = text.split("\n")
    prefix = re.compile(r"^\s*\d+\|\s?")
    nonempty = [ln for ln in lines if ln.strip()]
    if not nonempty:
        return text
    matched = sum(1 for ln in nonempty if prefix.match(ln))
    if matched >= 0.8 * len(nonempty):
        return "\n".join(prefix.sub("", ln) for ln in lines)
    return text


def _add_line_numbers(content: str) -> str:
    """Adds '  N| ' prefixes so the model can reference exact line
    positions in its reasoning, without those numbers leaking into
    its output (the prompt explicitly tells it not to include them)."""
    lines = content.split("\n")
    width = len(str(len(lines)))
    return "\n".join(f"{i+1:>{width}}| {line}" for i, line in enumerate(lines))


def _check_change_scope(original: str, new: str, max_changed_ratio: float = 0.5) -> dict:
    """
    Cheap sanity check: flags if a suspiciously large fraction of the
    file changed, which usually means the model ignored the "minimal
    change" instruction and rewrote/reformatted more than necessary.

    This is a heuristic, not a hard block — a genuinely large file with
    a small function might still trip a low threshold on a tiny file.
    Treat the flag as "worth a second look," not "automatically reject."
    """
    original_lines = original.split("\n")
    new_lines = new.split("\n")
    matcher = difflib.SequenceMatcher(a=original_lines, b=new_lines)
    changed_lines = sum(
        max(i2 - i1, j2 - j1)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    )
    ratio = changed_lines / max(len(original_lines), 1)
    return {
        "changed_lines": changed_lines,
        "total_lines": len(original_lines),
        "changed_ratio": round(ratio, 3),
        "suspiciously_large": ratio > max_changed_ratio,
    }


def _git_autocrlf_enabled(repo_path: str) -> bool:
    """
    Checks the repo's effective core.autocrlf setting. Falls back to
    False (i.e. assume diffs should match disk) if git config can't be
    read for any reason — a missing/unreadable config is far more
    likely on a non-Windows CI box than a genuine autocrlf=true setup,
    so defaulting to "match disk" is the safer assumption there.
    """
    try:
        result = subprocess.run(
            ["git", "config", "--get", "core.autocrlf"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip().lower() == "true"
    except (subprocess.SubprocessError, OSError):
        return False


def generate_patch(
    root_cause: dict,
    file_path: str,
    repo_path: str,
    retry_context: dict | None = None,
) -> dict:
    """
    Generates a unified diff for a single affected file.

    Args:
        root_cause: output of agents.root_cause_agent.analyze_root_cause().
        file_path: relative path (within the repo) of the file to patch.
            Must be one of root_cause["affected_files"].
        repo_path: local filesystem path to the cloned repo root.
        retry_context: optional. When the orchestrator retries after a
            failed validation, it passes the previous attempt's data here
            so the model can learn from it instead of repeating the same
            broken patch. Expected shape:
                {
                    "previous_diff": str,
                    "attempt": int,          # 1-based attempt number
                    "max_attempts": int,
                    "failed_tests": [{"test_name": str, "error_message": str}, ...],
                    "apply_error": str | None,  # set if the prev diff didn't apply
                }
            Passing None (the default) reproduces the original first-attempt
            behaviour exactly — this parameter is purely additive.

    Returns:
        {
            "diff": str,              # unified diff, computed by difflib
            "scope_check": dict,      # output of _check_change_scope
        }

    Raises:
        FileNotFoundError: if file_path doesn't exist in the repo.
    """
    full_path = os.path.join(repo_path, file_path)
    if not os.path.isfile(full_path):
        raise FileNotFoundError(
            f"Patch Agent: affected file does not exist in repo: {file_path}"
        )

    # Defensive normalization: git diff headers always use forward
    # slashes. Upstream agents already normalize at the source, but
    # this is cheap insurance against any future caller that doesn't.
    file_path = file_path.replace("\\", "/")

    # Read in binary to get the file's exact bytes, then normalize to
    # \n for all internal processing. We deliberately do NOT try to
    # preserve \r\n in the diff based on what's physically on disk —
    # that was the wrong signal. What actually determines what `git
    # apply` expects is the repo's core.autocrlf setting:
    #   - autocrlf=true (the Windows Git-for-Windows default): git
    #     stores/diffs everything as LF internally and converts to
    #     CRLF only on checkout. A diff must be LF-only here, even
    #     though the file on disk is CRLF.
    #   - autocrlf=false: the diff should match whatever is on disk.
    # Getting this backwards is exactly what caused "patch does not
    # apply" even when the diff looked visually identical to the file.
    with open(full_path, "rb") as f:
        raw_bytes = f.read()
    original_content = raw_bytes.decode("utf-8").replace("\r\n", "\n")

    prompt = PROMPT_TEMPLATE.format(
        root_cause_json=json.dumps(root_cause, indent=2),
        file_path=file_path,
        numbered_file_content=_add_line_numbers(original_content),
    )

    # On a retry, prepend the failure context so the model revises rather
    # than regenerates from scratch. The base prompt (file content + rules)
    # still follows, so the model has everything: what it tried, why it
    # failed, and the exact bytes to fix.
    if retry_context:
        prompt = RETRY_PROMPT_PREFIX.format(
            attempt=retry_context.get("attempt", "?"),
            max_attempts=retry_context.get("max_attempts", "?"),
            previous_diff=retry_context.get("previous_diff", "(previous diff unavailable)"),
            failure_detail=_format_failure_detail(retry_context),
        ) + prompt

    raw_output = call_llm(prompt, system_instruction=SYSTEM_INSTRUCTION)
    new_content = _strip_markdown_fences(raw_output)
    new_content = _strip_line_number_prefixes(new_content)
    new_content = new_content.replace("\r\n", "\n")

    # Preserve the original's trailing-newline convention. The model
    # may or may not include a final newline; matching the original
    # avoids a spurious "no newline at end of file" diff line that
    # would otherwise show up as a change even when nothing else did.
    if original_content.endswith("\n") and not new_content.endswith("\n"):
        new_content += "\n"

    scope_check = _check_change_scope(original_content, new_content)

    # Decide CRLF vs LF for the diff based on core.autocrlf, not on
    # what's physically on disk. autocrlf=true (default on Windows Git
    # installs) means git's internal diffing always uses LF, and
    # handles CRLF conversion itself on checkout — so we must emit an
    # LF-only diff for git apply to accept it, even on a CRLF file.
    use_crlf = (not _git_autocrlf_enabled(repo_path)) and (b"\r\n" in raw_bytes)
    if use_crlf:
        diff_original = original_content.replace("\n", "\r\n")
        diff_new = new_content.replace("\n", "\r\n")
    else:
        diff_original = original_content
        diff_new = new_content

    diff_lines = difflib.unified_diff(
        diff_original.splitlines(keepends=True),
        diff_new.splitlines(keepends=True),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
    )
    diff_text = "".join(diff_lines)

    return {
        "diff": diff_text,
        "scope_check": scope_check,
    }


def _write_diff_tempfile(diff_text: str) -> str:
    """
    Writes diff_text to a temp file in BINARY mode, with no newline
    translation. This matters specifically on Windows: Python's text
    mode ("w") translates \\n -> os.linesep on write, which on Windows
    is \\r\\n. That would corrupt the diff's own structural markers
    (the --- / +++ / @@ lines are always \\n-terminated, regardless of
    the target file's line-ending convention) and any content lines
    we deliberately set to \\r\\n to match the original file — causing
    `git apply` to reject a diff that looks visually correct.
    """
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".diff", delete=False) as f:
        data = diff_text.encode("utf-8")
        if not diff_text.endswith("\n"):
            data += b"\n"
        f.write(data)
        return f.name


def validate_diff_applies(diff_text: str, repo_path: str) -> dict:
    """
    Checks whether a diff would apply cleanly, WITHOUT actually applying
    it (git apply --check is non-destructive). This is validation layer
    zero — there's no point running flake8 or pytest on a diff that
    can't even be applied to the working tree.

    Returns:
        {"applies": bool, "error": str | None}
    """
    diff_path = _write_diff_tempfile(diff_text)

    try:
        result = subprocess.run(
            ["git", "apply", "--check", diff_path],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        return {
            "applies": result.returncode == 0,
            "error": result.stderr.strip() if result.returncode != 0 else None,
        }
    finally:
        os.unlink(diff_path)


def apply_diff(diff_text: str, repo_path: str) -> dict:
    """
    Actually applies a diff to the working tree. Call validate_diff_applies
    first — this function does not check before applying, since the
    orchestrator should make that decision explicitly rather than have
    it happen implicitly inside this function.

    Returns:
        {"applied": bool, "error": str | None}
    """
    diff_path = _write_diff_tempfile(diff_text)

    try:
        result = subprocess.run(
            ["git", "apply", diff_path],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        return {
            "applied": result.returncode == 0,
            "error": result.stderr.strip() if result.returncode != 0 else None,
        }
    finally:
        os.unlink(diff_path)


if __name__ == "__main__":
    # Quick manual test — run with: python -m agents.patch_agent
    # Reuses the Root Cause Agent's real output against the sample repo.
    from agents.issue_agent import understand_issue
    from analyzers.call_graph import build_repo_structure, trace_call_path
    from rag.vector_store import get_or_build_collection, query_collection
    from agents.root_cause_agent import analyze_root_cause

    bug_report = "Login page crashes when password field is empty"
    repo_path = "sample-repo"

    print("--- Re-running upstream agents to get a fresh diagnosis ---")
    issue = understand_issue(bug_report)
    structures = build_repo_structure(repo_path)
    trace = trace_call_path(structures, seed_files=["login.py"], max_depth=3)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_path,
        capture_output=True, text=True
    ).stdout.strip()
    collection = get_or_build_collection(repo_path, "local-sample-repo", commit)
    rag_results = query_collection(collection, bug_report, n_results=5)

    diagnosis = analyze_root_cause(issue, call_graph_trace=trace, rag_chunks=rag_results)
    print(json.dumps(diagnosis, indent=2))

    print("\n--- Generating patch ---")
    target_file = diagnosis["affected_files"][0]
    patch_result = generate_patch(diagnosis, target_file, repo_path)
    print(patch_result["diff"])
    print(f"Scope check: {json.dumps(patch_result['scope_check'], indent=2)}")

    print("\n--- Validating diff applies cleanly (--check, non-destructive) ---")
    validation = validate_diff_applies(patch_result["diff"], repo_path)
    print(json.dumps(validation, indent=2))
