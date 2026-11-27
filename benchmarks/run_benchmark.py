"""
run_benchmark.py — end-to-end benchmark for AuraFix across every supported language.

The corpus (benchmarks/cases/<id>/) is a set of small repositories, each with one
real bug, tests that fail because of it, and a reference fix (solution/). The
bundled samples (sample-repo, sample-shop, sample-java) are included too.

Modes
-----
--offline   No LLM, no key. Proves the corpus *and* AuraFix's test runner for
            every language: each buggy repo must FAIL its tests through
            validators.test_validator, and the reference fix must PASS. Cheap
            and deterministic, so CI runs it on every push.

(default)   Live. Runs the whole AuraFix pipeline on every case, then scores it:
              - status matches the expected outcome,
              - the diagnosis blamed the right file (and function),
              - an independent re-run of the repo's tests passes on the patched
                code (AuraFix's own "tests passed" is not taken on trust).
            Rotates through --models when a model hits its quota, and --resume
            skips cases already solved in the last results file, so a free-tier
            key can finish the suite across several sessions.

Toolchains: Java needs Maven (or Gradle), C# the .NET SDK, C/C++ CMake (+ Ninja
or make) and a C/C++ compiler. Folders listed in AURAFIX_TOOL_PATHS are put in
front of PATH (see validators.test_validator.tool_env).

Examples
--------
    python -m benchmarks.run_benchmark --offline
    python -m benchmarks.run_benchmark --mode fast --embed local
    python -m benchmarks.run_benchmark --only java-,cs- --resume
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

PROJECT = pathlib.Path(__file__).resolve().parent.parent
CASES_DIR = PROJECT / "benchmarks" / "cases"
RESULTS_DIR = PROJECT / "benchmarks" / "results"
sys.path.insert(0, str(PROJECT))

# The bundled demo samples double as benchmark cases. They ship without a
# solution/ folder; offline mode only checks that their tests fail.
BUNDLED = [
    {"id": "sample-login-crash", "language": "Python", "path": "sample-repo",
     "bug_report": "Login page crashes when the password field is empty",
     "expected_files": ["auth/auth_service.py"], "expected_function": "authenticate",
     "expected_status": "pr_ready"},
    {"id": "sample-discount-py", "language": "Python", "path": "sample-shop",
     "bug_report": "Discounts are applied incorrectly. A 10% discount subtracts 10 from the price "
                   "instead of taking 10% off.",
     "expected_files": ["pricing/discount.py"], "expected_function": "apply_discount",
     "expected_status": "pr_ready"},
    {"id": "sample-discount-java", "language": "Java", "path": "sample-java",
     "bug_report": "Discounts are applied incorrectly. A 10% discount subtracts 10 from the price "
                   "instead of taking 10% off.",
     "expected_files": ["src/main/java/shop/Discount.java"], "expected_function": "applyDiscount",
     "expected_status": "pr_ready"},
]

QUOTA_MARKERS = ("quota", "rate limit", "overloaded")


def load_cases(only: list[str]) -> list[dict]:
    cases = []
    for cfile in sorted(CASES_DIR.glob("*/case.json")):
        c = json.loads(cfile.read_text(encoding="utf-8"))
        c["repo_dir"] = str(cfile.parent / "repo")
        sol = cfile.parent / "solution"
        c["solution_dir"] = str(sol) if sol.is_dir() else None
        cases.append(c)
    for b in BUNDLED:
        cases.append({**b, "repo_dir": str(PROJECT / b["path"]), "solution_dir": None})
    if only:
        cases = [c for c in cases if any(c["id"].startswith(o) or c["language"].lower() == o.lower()
                                         for o in only)]
    return cases


def materialise(case: dict) -> str:
    """Copy the case repo to a fresh temp dir as a one-commit git repo. Fixed
    author and dates make the commit hash stable, so the RAG index cache is
    reused across runs of the same case."""
    tmp = tempfile.mkdtemp(prefix=f"aurafix_bench_{case['id']}_")
    repo = os.path.join(tmp, "repo")
    shutil.copytree(case["repo_dir"], repo, ignore=shutil.ignore_patterns(
        "__pycache__", ".pytest_cache", "target", "build", "bin", "obj", ".git"))
    env = {**os.environ, "GIT_AUTHOR_NAME": "bench", "GIT_AUTHOR_EMAIL": "bench@example.invalid",
           "GIT_COMMITTER_NAME": "bench", "GIT_COMMITTER_EMAIL": "bench@example.invalid",
           "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z"}
    for cmd in (["git", "init", "-q", "-b", "main"], ["git", "add", "-A"],
                ["git", "-c", "core.autocrlf=false", "commit", "-q", "-m", "benchmark case"]):
        subprocess.run(cmd, cwd=repo, env=env, check=True, capture_output=True)
    return repo


def apply_solution(case: dict, repo: str) -> None:
    """Overlay the reference fix with *fresh* timestamps. copytree keeps the
    source files' old mtimes, so incremental builds (Maven, Ninja, make) that
    already compiled the buggy version would think nothing changed and re-run
    the stale binaries."""
    src = pathlib.Path(case["solution_dir"])
    for f in src.rglob("*"):
        if f.is_file():
            dest = pathlib.Path(repo) / f.relative_to(src)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dest)  # copyfile: contents only, mtime = now
            os.utime(dest, None)


def run_tests(repo: str, case: dict) -> dict:
    from validators.test_validator import validate_patch
    t = time.time()
    v = validate_patch(repo, case["expected_files"], run_full_suite_on_pass=True)
    r = v.get("targeted_result") or {}
    failed = r.get("failed_tests") or []
    detail = (failed[0].get("error_message") or "") if failed else (r.get("error") or "")
    return {"passed": bool(v.get("overall_passed")), "total": r.get("total"),
            "failed": r.get("failed_count"), "error": (r.get("error") or "")[:400] or None,
            "detail": str(detail)[-1500:] or None,
            "toolchain_missing": bool(r.get("toolchain_missing")), "no_tests": bool(r.get("no_tests")),
            "language": v.get("language"), "seconds": round(time.time() - t, 1)}


# --------------------------------------------------------------------- offline
def offline_case(case: dict) -> dict:
    out = {"id": case["id"], "language": case["language"], "mode": "offline"}
    try:
        repo = materialise(case)
        out["buggy"] = run_tests(repo, case)
        if case["solution_dir"]:
            apply_solution(case, repo)
            out["fixed"] = run_tests(repo, case)
        shutil.rmtree(os.path.dirname(repo), ignore_errors=True)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    b, f = out.get("buggy") or {}, out.get("fixed")
    out["ok"] = (not out.get("error") and not b.get("passed") and not b.get("toolchain_missing")
                 and not b.get("no_tests") and (f is None or f.get("passed")))
    return out


# ------------------------------------------------------------------------ live
def make_settings(args, model: str):
    from agents.runtime import LLMSettings
    return LLMSettings(chat_provider=args.provider, chat_model=model, chat_api_key=args.key,
                       chat_base_url=args.base_url, embed_provider=args.embed)


def live_case(case: dict, args) -> dict:
    from agents.runtime import set_settings
    from github_integration.credentials import set_github_token
    from graph import BugFixRequest, run_pipeline

    set_github_token("")  # benchmarks never touch GitHub credentials
    out = {"id": case["id"], "language": case["language"], "mode": args.mode, "embed": args.embed}
    attempts = []
    for model in [m for m in args.models if m not in args.exhausted]:
        repo = materialise(case)
        try:
            set_settings(make_settings(args, model))
            t = time.time()
            steps = []
            final = run_pipeline(BugFixRequest(
                bug_report=case["bug_report"], repo_path=repo, repo_id=f"bench-{case['id']}",
                create_pr=False, retrieval_mode=args.mode, max_retries=args.max_retries,
            ), on_step=lambda e: steps.append(e))
            seconds = round(time.time() - t, 1)
            reason = (final.get("outcome_reason") or "")
            attempts.append({"model": model, "status": final.get("status"), "seconds": seconds})
            if final.get("status") == "error" and any(m in reason.lower() for m in QUOTA_MARKERS):
                # Remember it for the rest of the run: retrying an exhausted
                # model on every later case only wastes time and requests.
                args.exhausted.add(model)
                print(f"      {model}: {reason[:90]} -> skipping this model from now on")
                continue

            rc = final.get("root_cause") or {}
            files = [f.replace("\\", "/") for f in rc.get("affected_files") or []]
            out.update({
                "model": model, "status": final.get("status"), "seconds": seconds,
                "reason": reason[:400], "attempts": final.get("attempt"),
                "root_cause": (rc.get("root_cause") or "")[:300], "confidence": rc.get("confidence"),
                "blamed_files": files, "blamed_function": rc.get("affected_function"),
                "right_file": any(e in files for e in case["expected_files"]),
                "right_function": (case["expected_function"].lower() in (rc.get("affected_function") or "").lower()),
                "diff": ((final.get("patch") or {}).get("diff") or "")[:2500],
                "steps": [f"{s['node']}: {s['summary']}" for s in steps],
                "rag_files": sorted({c["filename"] for c in final.get("rag_chunks") or []}),
            })
            # Independent check: re-run the tests on whatever is on disk now.
            out["verify"] = run_tests(repo, case) if final.get("status") in ("pr_ready",) else None
            if final.get("status") == "error":
                tb = [h for h in final.get("history") or [] if h.get("node") == "pipeline"]
                if tb:
                    out["traceback"] = ((tb[0].get("details") or {}).get("traceback") or "")[-2500:]
            break
        except Exception as e:  # noqa: BLE001
            out.update({"model": model, "status": "harness_error", "reason": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc()[-2500:]})
            break
        finally:
            shutil.rmtree(os.path.dirname(repo), ignore_errors=True)
    else:
        out.update({"status": "quota_exhausted", "reason": "every model hit its quota or was overloaded"})
    out["model_attempts"] = attempts

    expected = case.get("expected_status", "pr_ready")
    verified = expected != "pr_ready" or bool((out.get("verify") or {}).get("passed"))
    out["solved"] = out.get("status") == expected and verified and bool(out.get("right_file"))
    return out


# --------------------------------------------------------------------- report
def write_report(results: list[dict], mode: str) -> pathlib.Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = "offline" if mode == "offline" else "live"
    data = {"generated": stamp, "mode": mode, "results": results}
    (RESULTS_DIR / f"{name}-{stamp}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    (RESULTS_DIR / f"{name}-latest.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    lines = [f"# AuraFix benchmark ({mode}) - {stamp}", ""]
    if mode == "offline":
        ok = sum(r["ok"] for r in results)
        lines += [f"**{ok}/{len(results)} cases valid**: buggy code fails its tests, reference fix passes.", "",
                  "| Case | Language | Buggy tests | Reference fix | Result |", "|---|---|---|---|---|"]
        for r in results:
            b, f = r.get("buggy") or {}, r.get("fixed")
            bt = "error" if r.get("error") else ("pass (bug not caught!)" if b.get("passed") else
                                                 ("toolchain missing" if b.get("toolchain_missing") else
                                                  f"fail ({b.get('seconds')}s)"))
            ft = "n/a" if f is None else (f"pass ({f.get('seconds')}s)" if f.get("passed") else "FAIL")
            lines.append(f"| {r['id']} | {r['language']} | {bt} | {ft} | {'ok' if r['ok'] else 'PROBLEM'} |")
    else:
        solved = sum(r.get("solved", False) for r in results)
        lines += [f"**{solved}/{len(results)} solved** (expected outcome + right file + independently "
                  "re-verified tests).", "",
                  "| Case | Language | Status | Right file | Right function | Verified | Time | Model | Solved |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in results:
            v = r.get("verify")
            vt = "n/a" if v is None else ("pass" if v.get("passed") else "FAIL")
            lines.append(f"| {r['id']} | {r['language']} | {r.get('status')} | {r.get('right_file')} | "
                         f"{r.get('right_function')} | {vt} | {r.get('seconds', '-')}s | {r.get('model', '-')} | "
                         f"{'yes' if r.get('solved') else 'NO'} |")
        bad = [r for r in results if not r.get("solved")]
        if bad:
            lines += ["", "## Not solved", ""]
            for r in bad:
                lines += [f"### {r['id']} ({r.get('status')})", "", f"- Reason: {r.get('reason')}",
                          f"- Blamed: {r.get('blamed_files')} :: {r.get('blamed_function')}"]
                if r.get("verify") and r["verify"].get("error"):
                    lines.append(f"- Verify error: {r['verify']['error']}")
                lines.append("")
    path = RESULTS_DIR / f"{name}-latest.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="validate corpus + test runner, no LLM")
    ap.add_argument("--only", default="", help="comma-separated id prefixes or language names")
    ap.add_argument("--mode", default="multi_agent", choices=["multi_agent", "fast"])
    ap.add_argument("--embed", default="local", choices=["local", "sfr", "gemini", "openai", "auto"])
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--models", default="gemini-2.5-flash,gemini-2.5-flash-lite,gemini-flash-latest")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--env-file", default=str(PROJECT / ".env"))
    ap.add_argument("--resume", action="store_true", help="skip cases solved in the last live run")
    args = ap.parse_args()
    os.chdir(PROJECT)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    cases = load_cases([o.strip() for o in args.only.split(",") if o.strip()])
    from validators.test_validator import which
    print("Toolchains:", {t: bool(which(t)) for t in ("mvn", "dotnet", "cmake", "ninja", "make", "gcc", "g++")})

    if args.offline:
        results = []
        for c in cases:
            print(f"  {c['id']:<24} ", end="", flush=True)
            r = offline_case(c)
            b, f = r.get("buggy") or {}, r.get("fixed") or {}
            print(f"{'ok ' if r['ok'] else 'BAD'}  buggy_passed={b.get('passed')} fixed_passed={f.get('passed')} "
                  f"{r.get('error') or b.get('error') or ''}"[:200])
            if not r["ok"]:
                # Show why, so a CI failure is diagnosable from the log alone.
                bad = f if (f and not f.get("passed")) else b
                print("      " + str(bad.get("detail") or bad.get("error") or "").replace("\n", "\n      ")[-1500:])
            results.append(r)
        path = write_report(results, "offline")
        print(f"\n{sum(r['ok'] for r in results)}/{len(results)} valid. Report: {path}")
        return 0 if all(r["ok"] for r in results) else 1

    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    args.exhausted = set()
    args.key = os.environ.get({"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY",
                               "anthropic": "ANTHROPIC_API_KEY"}.get(args.provider, "LLM_API_KEY"), "")
    if not args.key and os.path.exists(args.env_file):
        from dotenv import dotenv_values
        env = dotenv_values(args.env_file)
        args.key = (env.get("GEMINI_API_KEY") if args.provider == "gemini" else
                    env.get(f"{args.provider.upper()}_API_KEY") or env.get("LLM_API_KEY") or "") or ""
    if not args.key:
        print("No API key: set GEMINI_API_KEY (or the provider's key) or use --offline.")
        return 2

    previous = {}
    latest = RESULTS_DIR / "live-latest.json"
    if args.resume and latest.exists():
        previous = {r["id"]: r for r in json.loads(latest.read_text(encoding="utf-8"))["results"]}

    results = []
    for c in cases:
        prev = previous.get(c["id"])
        if prev and prev.get("solved") and prev.get("mode") == args.mode:
            print(f"  {c['id']:<24} (solved earlier, skipped)")
            results.append(prev)
            continue
        print(f"  {c['id']:<24} ", end="", flush=True)
        r = live_case(c, args)
        print(f"{'SOLVED' if r.get('solved') else 'no    '} status={r.get('status')} "
              f"file={r.get('right_file')} fn={r.get('right_function')} "
              f"verify={(r.get('verify') or {}).get('passed')} {r.get('seconds', '-')}s [{r.get('model')}]")
        results.append(r)
        write_report(results + [p for i, p in previous.items() if i not in {x['id'] for x in results}
                                and i not in {x['id'] for x in cases}], args.mode)
    path = write_report(results, args.mode)
    print(f"\n{sum(r.get('solved', False) for r in results)}/{len(results)} solved. Report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
