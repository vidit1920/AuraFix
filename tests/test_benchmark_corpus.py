"""
test_benchmark_corpus.py — the benchmark corpus is well-formed.

Fast structural checks (no toolchains): every case declares what the live
benchmark scores against, points at files that exist, and ships a reference fix
that actually changes the buggy file. The slow behavioural check (buggy fails,
fix passes, every language) is `python -m benchmarks.run_benchmark --offline`,
which CI runs in its own job.
"""

import json
import pathlib

import pytest

from benchmarks.run_benchmark import CASES_DIR, load_cases
from validators.test_validator import detect_runner

CASES = sorted(CASES_DIR.glob("*/case.json"))
LANG_RUNNER = {"Python": "Python", "Java": "Java", "C#": "C#", "C": "C/C++", "C++": "C/C++"}


def test_every_language_is_covered():
    langs = {json.loads(p.read_text(encoding="utf-8"))["language"] for p in CASES}
    assert {"Python", "Java", "C#", "C", "C++"} <= langs


@pytest.mark.parametrize("case_file", CASES, ids=lambda p: p.parent.name)
def test_case_is_well_formed(case_file: pathlib.Path):
    case = json.loads(case_file.read_text(encoding="utf-8"))
    base = case_file.parent
    assert case["id"] == base.name
    assert case["bug_report"].strip() and case["expected_function"].strip()
    assert case["expected_status"] in ("pr_ready", "needs_human_review")
    for rel in case["expected_files"]:
        assert (base / "repo" / rel).is_file(), rel

    solution = base / "solution"
    fixed = [p for p in solution.rglob("*") if p.is_file()]
    assert fixed, "a reference fix is required"
    for p in fixed:
        buggy = base / "repo" / p.relative_to(solution)
        assert buggy.is_file(), f"solution file has no buggy counterpart: {p}"
        assert buggy.read_bytes() != p.read_bytes(), f"solution identical to repo: {p}"

    # AuraFix must pick the right test runner for this repo.
    assert detect_runner(str(base / "repo"))["language"] == LANG_RUNNER[case["language"]]


def test_loader_includes_bundled_samples():
    ids = {c["id"] for c in load_cases([])}
    assert {"sample-login-crash", "sample-discount-py", "sample-discount-java"} <= ids
    assert len(ids) == len(CASES) + 3
