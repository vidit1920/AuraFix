"""
test_validator.py — Agent 6 in the pipeline.

This is where a generated patch meets reality. It does three things,
in order, each gated on the previous one succeeding:

  1. Apply the diff for real (not --check) using patch_agent.apply_diff.
  2. Run pytest, scoped to the test file matching the affected source
     file where possible (faster, cheaper than the full suite on every
     retry attempt).
  3. Return structured pass/fail data, including full tracebacks for
     failed tests — not just a pass/fail count. The retry prompt fed
     back to the Patch Agent needs the actual error text to have any
     chance of producing a better second attempt; "tests failed, try
     again" with no detail tends to reproduce the same mistake.

This module does NOT decide whether to retry — that's an orchestration
decision (LangGraph conditional edge), kept out of this module so the
retry cap and escalation logic live in one place (graph/workflow.py)
rather than being duplicated across every agent that might trigger a
retry.
"""

import os
import json
import shutil
import subprocess


def find_matching_test_file(repo_path: str, source_file: str) -> str | None:
    """
    Guesses the test file corresponding to a source file, using the
    common `test_<name>.py` convention. Returns None if no match is
    found — callers should fall back to running the full suite in
    that case rather than failing outright.

    Example: "auth/auth_service.py" -> "tests/test_auth_service.py"
    (searches the whole repo tree for this filename, since test
    directory layout varies between projects — some put tests/ at the
    root, some mirror the source tree under tests/).
    """
    basename = os.path.basename(source_file)  # "auth_service.py"
    name_without_ext = os.path.splitext(basename)[0]  # "auth_service"
    candidate_name = f"test_{name_without_ext}.py"

    skip_dirs = {".git", "__pycache__", "venv", ".venv", "node_modules"}
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        if candidate_name in files:
            full_path = os.path.join(root, candidate_name)
            rel_path = os.path.relpath(full_path, repo_path)
            return rel_path.replace(os.sep, "/")

    return None


def run_tests(repo_path: str, test_path: str = None, timeout: int = 120) -> dict:
    """
    Runs pytest in the repo, returns structured pass/fail data.

    Args:
        repo_path: local filesystem path to the repo.
        test_path: relative path to a specific test file/dir to run.
            If None, runs the full suite — slower, but a reasonable
            fallback when no matching test file was found.
        timeout: max seconds to let pytest run before giving up. A
            patched bug could in rare cases cause an infinite loop;
            without a timeout, one bad patch could hang the whole
            pipeline indefinitely.

    Returns:
        {
            "passed": bool,
            "total": int,
            "passed_count": int,
            "failed_count": int,
            "failed_tests": [{"test_name": str, "error_message": str}, ...],
            "error": str | None,   # set if pytest itself failed to run
                                    # (e.g. collection error, timeout)
        }
    """
    report_filename = ".pytest_report.json"
    report_path = os.path.join(repo_path, report_filename)

    # Clean up any stale report from a previous run before starting —
    # if pytest crashes before writing a new one, we don't want to
    # accidentally read old results and misreport them as current.
    if os.path.exists(report_path):
        os.remove(report_path)

    cmd = ["pytest", "--json-report", f"--json-report-file={report_filename}"]
    if test_path:
        cmd.extend(test_path.split(" "))

    try:
        subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "total": 0,
            "passed_count": 0,
            "failed_count": 0,
            "failed_tests": [],
            "error": f"pytest did not complete within {timeout}s — the patch may have introduced an infinite loop or hang",
        }

    if not os.path.exists(report_path):
        # pytest failed before producing a report at all — usually a
        # collection error (e.g. the patch introduced a syntax error
        # that prevents the test file from even importing).
        return {
            "passed": False,
            "total": 0,
            "passed_count": 0,
            "failed_count": 0,
            "failed_tests": [],
            "error": "pytest did not produce a report — likely a collection/import error in the patched file",
        }

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    os.remove(report_path)  # clean up so it doesn't get committed/confused with the next run

    failed_tests = [
        {
            "test_name": t["nodeid"],
            "error_message": _extract_failure_text(t),
        }
        for t in report.get("tests", [])
        if t["outcome"] == "failed"
    ]

    # Collection/import errors (e.g. the patch introduced a syntax error, so
    # the test module can't even be imported) show up as failed *collectors*,
    # NOT as failed tests, and pytest reports total=0 for them. Surface that
    # explicitly — otherwise a patch that breaks collection looks identical
    # to "0 tests, nothing wrong" and would get a false green light, which is
    # the worst possible failure mode for a tool whose job is proving a fix.
    collection_errors = [
        str(c.get("longrepr", "")).strip()
        for c in report.get("collectors", [])
        if c.get("outcome") == "failed"
    ]

    summary = report.get("summary", {})
    total = summary.get("total", 0)
    exitcode = report.get("exitcode", 1)

    # pytest's own exit code is the authoritative pass signal: 0 means every
    # collected test passed (1 = failures, 2 = collection error, 5 = nothing
    # collected). We additionally require total > 0 — "0 tests ran" must
    # never count as success, since it proves nothing about the fix.
    passed = (
        exitcode == 0
        and total > 0
        and summary.get("failed", 0) == 0
        and summary.get("error", 0) == 0
    )

    error = None
    if collection_errors:
        error = (
            "Test collection failed — the patched code likely has a syntax or "
            "import error:\n" + "\n".join(collection_errors)
        )
    elif total == 0:
        error = "No tests were collected, so the fix could not be validated."

    return {
        "passed": passed,
        "total": total,
        "passed_count": summary.get("passed", 0),
        "failed_count": summary.get("failed", 0),
        "failed_tests": failed_tests,
        "error": error,
    }


def _extract_failure_text(test_result: dict) -> str:
    """
    Pulls the most useful failure detail out of a pytest-json-report
    test entry. The 'longrepr' field (full traceback) is what the Fix
    Agent needs on retry — a bare pass/fail count gives it nothing to
    work with and tends to produce the same wrong patch again.
    """
    call_info = test_result.get("call", {})
    longrepr = call_info.get("longrepr")
    if longrepr:
        return longrepr
    # Fall back to setup-phase failures (e.g. fixture errors) if the
    # test never reached the call phase.
    setup_info = test_result.get("setup", {})
    return setup_info.get("longrepr", "no failure detail available")


_LANG_EXTS = {
    "Python": {".py"},
    "Java": {".java"},
    "C#": {".cs"},
    "C/C++": {".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"},
}
_SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "node_modules", ".pytest_cache",
              "target", "build", "bin", "obj", "dist"}


def _dominant_language(repo_path: str) -> str:
    """Picks the repo's primary language by counting source files per
    language. More reliable than build-marker order alone: many Python/JS
    projects ship a Makefile, which must NOT make them look like a C project
    (this exact mis-detection showed up testing against psf/requests)."""
    counts = {lang: 0 for lang in _LANG_EXTS}
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            for lang, exts in _LANG_EXTS.items():
                if ext in exts:
                    counts[lang] += 1
                    break
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else "Python"


def detect_runner(repo_path: str, affected_files: list | None = None) -> dict:
    """
    Decides how to run this repo's tests: first the dominant language (by
    source-file count), then that language's test command from its build
    markers. Returns {"language": str, "steps": list[list[str]] | None};
    steps is None for Python (which uses the rich pytest path).
    """
    def exists(*parts):
        return os.path.exists(os.path.join(repo_path, *parts))

    lang = _dominant_language(repo_path)

    if lang == "Python":
        return {"language": "Python", "steps": None}

    if lang == "Java":
        if exists("build.gradle") or exists("build.gradle.kts"):
            if exists("gradlew") or exists("gradlew.bat"):
                gw = os.path.join(repo_path, "gradlew.bat" if os.name == "nt" else "gradlew")
                return {"language": "Java", "steps": [[gw, "test"]]}
            return {"language": "Java", "steps": [["gradle", "test"]]}
        return {"language": "Java", "steps": [["mvn", "-q", "-B", "test"]]}

    if lang == "C#":
        return {"language": "C#", "steps": [["dotnet", "test"]]}

    # C / C++ — best effort; depends heavily on the project's build setup.
    if exists("CMakeLists.txt"):
        return {"language": "C/C++", "steps": [
            ["cmake", "-S", ".", "-B", "build"],
            ["cmake", "--build", "build"],
            ["ctest", "--test-dir", "build", "--output-on-failure"],
        ]}
    return {"language": "C/C++", "steps": [["make", "test"]]}


def _external_fail(error: str) -> dict:
    """A run_tests-shaped failure result for the non-Python path (e.g. the
    toolchain isn't installed)."""
    return {"passed": False, "total": 0, "passed_count": 0, "failed_count": 0,
            "failed_tests": [], "error": error}


def _run_external_steps(repo_path: str, steps: list, language: str, timeout: int = 900) -> dict:
    """
    Runs a non-Python test command (or sequence) and maps the outcome to the
    same shape run_tests() returns. Pass/fail is the process exit code (0 =
    pass), since we don't parse each ecosystem's per-test format. Captured
    output is fed back to the Fix Agent on failure (just like pytest
    tracebacks). If the toolchain isn't on PATH, fail with a clear message
    rather than a confusing crash.
    """
    exe = steps[0][0]
    if not os.path.isabs(exe) and shutil.which(exe) is None:
        return _external_fail(
            f"'{exe}' was not found on PATH — install the {language} toolchain to validate {language} fixes."
        )

    transcript = []
    for cmd in steps:
        try:
            proc = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return _external_fail(f"`{' '.join(cmd)}` timed out after {timeout}s.")
        except FileNotFoundError:
            return _external_fail(f"'{cmd[0]}' not found — install the {language} toolchain.")
        transcript.append(f"$ {' '.join(cmd)}\n{(proc.stdout or '')}{(proc.stderr or '')}")
        if proc.returncode != 0:
            tail = "\n".join(transcript)[-3000:]
            return {"passed": False, "total": 1, "passed_count": 0, "failed_count": 1,
                    "failed_tests": [{"test_name": f"{language} test suite", "error_message": tail}],
                    "error": None}

    return {"passed": True, "total": 1, "passed_count": 1, "failed_count": 0,
            "failed_tests": [], "error": None}


def validate_patch(repo_path: str, affected_files: list, run_full_suite_on_pass: bool = True) -> dict:
    """
    Language-aware entry point. Detects the project's toolchain and runs its
    tests. Python uses the rich pytest path (per-test detail); Java, C#, and
    C/C++ run their build tool and pass/fail on the exit code. Returns the
    same shape regardless: {"targeted_result", "full_suite_result",
    "overall_passed", "language"}.
    """
    runner = detect_runner(repo_path, affected_files)
    if runner["language"] != "Python":
        result = _run_external_steps(repo_path, runner["steps"], runner["language"])
        return {"targeted_result": result, "full_suite_result": None,
                "overall_passed": result["passed"], "language": runner["language"]}

    out = _validate_python(repo_path, affected_files, run_full_suite_on_pass)
    out["language"] = "Python"
    return out


def _validate_python(repo_path: str, affected_files: list, run_full_suite_on_pass: bool = True) -> dict:
    """
    Python path: figures out which test(s) to run for the
    given affected files, runs them, and optionally runs the full
    suite as a final sanity check if the targeted tests pass (catches
    regressions in unrelated areas that targeted testing would miss).

    Args:
        repo_path: local filesystem path to the repo (diff must
            already be applied before calling this).
        affected_files: from the Root Cause Agent's diagnosis — used
            to find the matching test file(s) via naming convention.
        run_full_suite_on_pass: if True and targeted tests pass, also
            run the complete suite once as a final check before
            considering validation successful. Set False to skip this
            for speed during early retry attempts, and True for the
            final attempt before PR creation.

    Returns:
        {
            "targeted_result": dict (output of run_tests, scoped),
            "full_suite_result": dict | None (only if run, see above),
            "overall_passed": bool,
        }
    """
    test_files = []
    for source_file in affected_files:
        match = find_matching_test_file(repo_path, source_file)
        if match:
            test_files.append(match)

    if test_files:
        # Run all matched test files together in one pytest invocation
        # rather than one call per file — faster, and pytest naturally
        # aggregates results across multiple paths in a single report.
        targeted_result = run_tests(repo_path, test_path=" ".join(test_files))
    else:
        # No naming-convention match found — fall back to the full
        # suite rather than skipping validation entirely.
        targeted_result = run_tests(repo_path, test_path=None)

    full_suite_result = None
    if targeted_result["passed"] and run_full_suite_on_pass and test_files:
        # Only worth running the full suite separately if we scoped
        # down earlier — if we already ran the full suite above
        # (no test_files match), running it again is redundant.
        full_suite_result = run_tests(repo_path, test_path=None)

    overall_passed = targeted_result["passed"] and (
        full_suite_result is None or full_suite_result["passed"]
    )

    return {
        "targeted_result": targeted_result,
        "full_suite_result": full_suite_result,
        "overall_passed": overall_passed,
    }


if __name__ == "__main__":
    # Quick manual test — run with: python -m validators.test_validator
    # Applies the real patch from the Patch Agent and validates it for
    # real this time (not --check), then reverts so the repo stays
    # clean for repeated testing.
    from agents.issue_agent import understand_issue
    from analyzers.call_graph import build_repo_structure, trace_call_path
    from rag.vector_store import get_or_build_collection, query_collection
    from agents.root_cause_agent import analyze_root_cause
    from agents.patch_agent import generate_patch, validate_diff_applies, apply_diff

    bug_report = "Login page crashes when password field is empty"
    repo_path = "sample-repo"

    print("--- Running upstream agents ---")
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
    print(f"Root cause: {diagnosis['root_cause']}\n")

    target_file = diagnosis["affected_files"][0]

    # Back up the original file content BEFORE patching. This is the
    # primary revert mechanism below — it has zero dependency on git
    # being a fully intact repository, unlike `git checkout`, which
    # failed on Windows in testing because the extracted project's
    # sample-repo/.git folder didn't survive a zip extraction (some
    # Windows zip tools silently skip dotfolders). A plain file-content
    # backup works regardless of git's state.
    original_target_path = os.path.join(repo_path, target_file)
    with open(original_target_path, "rb") as f:
        original_backup_bytes = f.read()

    patch_result = generate_patch(diagnosis, target_file, repo_path)
    print("--- Generated patch ---")
    print(patch_result["diff"])

    print("\n--- Running tests BEFORE applying patch (should show the known failure) ---")
    before = validate_patch(repo_path, diagnosis["affected_files"])
    print(json.dumps(before["targeted_result"], indent=2))

    print("\n--- Applying patch for real ---")
    check = validate_diff_applies(patch_result["diff"], repo_path)
    if not check["applies"]:
        print(f"Diff does not apply, aborting: {check['error']}")
    else:
        apply_result = apply_diff(patch_result["diff"], repo_path)
        print(f"Applied: {apply_result['applied']}")

        print("\n--- Running tests AFTER applying patch ---")
        after = validate_patch(repo_path, diagnosis["affected_files"])
        print(json.dumps(after["targeted_result"], indent=2))
        print(f"\nOverall passed: {after['overall_passed']}")

        print("\n--- Reverting patch so repo stays clean for repeated testing ---")
        with open(original_target_path, "wb") as f:
            f.write(original_backup_bytes)

        with open(original_target_path, "rb") as f:
            restored_bytes = f.read()
        reverted_correctly = restored_bytes == original_backup_bytes
        print(f"Reverted: {reverted_correctly}")
        if not reverted_correctly:
            print("WARNING: restored content does not match the original backup byte-for-byte")
