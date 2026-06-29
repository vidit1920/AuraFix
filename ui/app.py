"""
app.py — the AuraFix dashboard (Agent 8).

A polished, hosting-ready Streamlit UI for the autonomous bug-fixing
pipeline. Design goals:
  - Make the flow obvious: choose a repo -> describe the bug -> run.
  - Make the result legible: a clear verdict, the diagnosis + reasoning,
    the diff, the test result, and a step-by-step timeline.
  - Make opening a PR an obvious, deliberate action — a button that
    appears right under a validated fix (not a hidden toggle).
  - Let the user keep interacting via a follow-up chat about the fix.

Run with:  streamlit run ui/app.py
"""

import os
import sys

# Make the project root importable regardless of how the app is launched.
# `streamlit run ui/app.py` only puts the ui/ folder on sys.path — unlike
# `python -m streamlit` — so without this the top-level packages
# (graph, agents, rag, ...) wouldn't be found.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Default to gemini-2.5-flash (its free-tier daily bucket is separate from
# flash-lite). Set GEMINI_MODEL in your shell to override. Must run before
# importing the pipeline, since llm_client captures the model at import.
os.environ.setdefault("GEMINI_MODEL", "gemini-2.5-flash")

import re

import streamlit as st

st.set_page_config(page_title="AuraFix — Autonomous Bug Fixer", page_icon="🛠️", layout="wide")

_IMPORT_ERROR = None
try:
    from graph import run_pipeline, BugFixRequest
    from graph.state import (
        STATUS_PR_CREATED,
        STATUS_PR_COMMITTED_LOCAL,
        STATUS_PR_READY,
        STATUS_NEEDS_HUMAN_REVIEW,
        STATUS_ERROR,
    )
    from github_integration.clone_repo import clone_repo
    from github_integration.pr_agent import (
        build_pr_title,
        build_pr_description,
        create_branch_and_commit,
        open_pull_request,
    )
    from agents.llm_client import call_llm
except Exception as e:  # noqa: BLE001
    _IMPORT_ERROR = e

# Bundled buggy repos shipped with the project, for zero-setup testing.
# Each is its own local git repo with a known bug and a failing test.
BUNDLED_SAMPLES = {
    "Login crash — empty password (auth)": {
        "path": "sample-repo", "id": "local-sample-repo",
        "bug": "Login page crashes when the password field is empty",
        "blurb": "a login flow that crashes on an empty password (missing None check)",
    },
    "Wrong discount math — Python (shopping cart)": {
        "path": "sample-shop", "id": "local-sample-shop",
        "bug": "Discounts are applied incorrectly — a 10% discount subtracts 10 from the price "
               "instead of taking 10% off.",
        "blurb": "a Python checkout that computes discounts with the wrong formula",
    },
    "Wrong discount math — Java (Maven)": {
        "path": "sample-java", "id": "local-sample-java",
        "bug": "Discounts are applied incorrectly — a 10% discount subtracts 10 from the price "
               "instead of taking 10% off.",
        "blurb": "a Java/Maven checkout with the wrong discount formula (needs Maven installed to run its tests)",
    },
}
_SUCCESS_STATUSES = {STATUS_PR_CREATED, STATUS_PR_COMMITTED_LOCAL, STATUS_PR_READY} if not _IMPORT_ERROR else set()


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], button, input, textarea, select {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Hide only the noisy chrome — keep the sidebar toggle usable */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stStatusWidget"] {display: none;}
[data-testid="stMainMenuButton"] {display: none;}
.stAppDeployButton, [data-testid="stAppDeployButton"], [data-testid="stBaseButton-header"] {display: none;}
[data-testid="stHeader"] {background: transparent;}
/* Keep the sidebar collapse / expand controls visible (so it can be reopened) */
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stExpandSidebarButton"] {
    display: flex !important; visibility: visible !important; opacity: 1 !important; z-index: 1000;
}

.block-container {padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1080px;}

/* 'Aurora' background — calm/minimal: dim, very slow drift */
.stApp::before {
    content: ""; position: fixed; inset: -25%; z-index: 0; pointer-events: none;
    background:
      radial-gradient(38% 38% at 18% 18%, rgba(59,130,246,0.12), transparent 60%),
      radial-gradient(34% 34% at 82% 24%, rgba(56,189,248,0.08), transparent 60%),
      radial-gradient(42% 42% at 62% 96%, rgba(99,102,241,0.09), transparent 60%);
    filter: blur(50px);
    animation: af-aurora 60s ease-in-out infinite alternate;
}
@keyframes af-aurora {
    0%   {transform: translate3d(0,0,0) scale(1);}
    100% {transform: translate3d(-1%, 1%, 0) scale(1.015);}
}
/* Keep all content above the background layer */
.block-container, section[data-testid="stSidebar"] {position: relative; z-index: 1;}

/* Hero */
.af-hero {padding: 6px 0 10px 0;}
.af-pill {
    display: inline-block; font-size: 0.72rem; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; color: #93c5fd;
    background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.35);
    padding: 4px 12px; border-radius: 999px; margin-bottom: 14px;
}
.af-title {
    font-size: 3rem; font-weight: 800; line-height: 1.05; margin: 0 0 8px 0;
    background: linear-gradient(92deg, #ffffff 0%, #93c5fd 55%, #3b82f6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.af-sub {color: #9aa3b8; font-size: 1.05rem; max-width: 720px; margin: 0;}
.af-langs {margin-top: 14px; color:#8b93a7; font-size:.85rem;}
.af-lchip {display:inline-block; background:rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.30);
    color:#bfdbfe; border-radius:999px; padding:2px 11px; margin:0 4px 4px 0; font-weight:600; font-size:.78rem;}

/* "Made by" credit, pinned top-right */
.af-credit {position: fixed; top: 14px; right: 22px; z-index: 1000; font-size: .8rem;
    color: #8b93a7; background: rgba(20,25,39,0.72); border: 1px solid #243150;
    padding: 5px 13px; border-radius: 999px; backdrop-filter: blur(6px);}
.af-credit b {color: #93c5fd; font-weight: 600;}

/* Step strip */
.af-steps {display: flex; gap: 14px; margin: 22px 0 6px 0; flex-wrap: wrap;}
.af-step {
    flex: 1; min-width: 200px; background: rgba(20,25,39,0.7); border: 1px solid #232a3d;
    border-radius: 14px; padding: 16px 18px; transition: transform .18s ease, border-color .18s ease;
}
.af-step:hover {transform: translateY(-3px); border-color: #2f4a78;}
.af-step .n {color: #60a5fa; font-weight: 800; font-size: .82rem; letter-spacing: .08em;}
.af-step .t {font-weight: 700; margin: 4px 0 2px 0;}
.af-step .d {color: #8b93a7; font-size: .86rem;}

/* Cards (bordered containers) — glassy, with a gentle hover lift */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px !important; border: 1px solid #232a3d !important;
    background: rgba(16,20,31,0.82); backdrop-filter: blur(7px);
    transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-2px); border-color: #2f4a78 !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.30);
}
[data-testid="stVerticalBlockBorderWrapper"] > div {padding: 2px 4px;}

/* Section headers inside cards */
.af-card-h {font-size: 1.02rem; font-weight: 700; margin: 2px 0 12px 0; letter-spacing: .01em;}

/* Primary button */
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {
    background: linear-gradient(92deg, #3b82f6, #60a5fa) !important;
    border: none !important; border-radius: 10px !important; font-weight: 600 !important;
    padding: 0.6rem 1rem !important; box-shadow: 0 6px 18px rgba(59,130,246,0.28) !important;
    transition: filter .15s ease, transform .15s ease;
}
.stButton > button[kind="primary"]:hover {filter: brightness(1.08); transform: translateY(-1px);}
.stButton > button {border-radius: 10px !important;}

/* Verdict banner (status colour is the one place an icon earns its place) */
.af-verdict {border-radius: 14px; padding: 16px 18px; margin: 4px 0 6px 0; font-weight: 600; border: 1px solid;}
.af-ok   {background: rgba(34,197,94,0.10); border-color: rgba(34,197,94,0.40); color: #86efac;}
.af-warn {background: rgba(234,179,8,0.10); border-color: rgba(234,179,8,0.40); color: #fde047;}
.af-err  {background: rgba(239,68,68,0.10); border-color: rgba(239,68,68,0.40); color: #fca5a5;}
.af-verdict .r {font-weight: 400; color: #c7cdda; font-size: .9rem; display:block; margin-top: 4px;}

/* Inline chips */
.af-chip {display:inline-block; background:#1b2233; border:1px solid #2a3346; color:#aab3c6;
    border-radius: 8px; padding: 2px 9px; font-size: .8rem; margin: 2px 4px 2px 0; font-family: ui-monospace, monospace;}
.af-dot {color:#60a5fa; margin-right:6px;}

/* Metrics */
[data-testid="stMetric"] {background:rgba(20,25,39,0.7); border:1px solid #232a3d; border-radius:12px; padding:12px 14px;}
[data-testid="stMetricLabel"] {color:#8b93a7 !important;}

a {color: #60a5fa !important;}
hr {border-color:#232a3d;}
</style>
"""


ABOUT_MD = """\
**AuraFix** turns a plain-English bug report into a validated fix and a pull request — autonomously,
with a human in the loop only at review time.

**The pipeline — 8 agents orchestrated with [LangGraph](https://github.com/langchain-ai/langgraph):**

1. **Issue understanding** — parses your report into structured fields (module, severity, suspected files).
2. **Repository indexing (RAG)** — embeds the repo into a vector store and retrieves the most relevant code.
3. **Code navigation** — builds an AST call graph to trace the *real* execution path — ground truth, not just keyword matches.
4. **Root-cause analysis** — weighs all the evidence into one diagnosis with a confidence score.
5. **Patch generation** — writes a *minimal* diff that fixes the root cause, not the symptom.
6. **Validation** — applies the patch and runs the tests; on failure it retries with the error context (up to 3 times), then escalates for human review.
7. **Pull request** — forks the repo (or pushes directly if you have access), commits, and opens a PR with the diagnosis, reasoning, and diff.

**Confidence gating:** a fix is only auto-eligible for a PR above the confidence threshold (sidebar) *and* with passing
tests — otherwise it's surfaced as "investigated, needs review" rather than forced into a PR.

**Supported languages** (structural call-graph navigation + automated test validation):

| Language | Navigation | Test validation |
|----------|------------|-----------------|
| Python | `ast` (full fidelity) | pytest |
| Java | tree-sitter | Maven / Gradle |
| C / C++ | tree-sitter | CMake + CTest / `make test` |
| C# | tree-sitter | `dotnet test` |

Running a non-Python language's tests needs that language's toolchain installed (JDK + Maven/Gradle, .NET SDK,
CMake + a compiler). The LLM can read and write other languages too, but these five are the ones with navigation
and validation wired in.

**Built with:** Python · LangGraph · Google Gemini · ChromaDB · tree-sitter · GitPython / PyGithub.

**Use responsibly:** every fix is AI-generated. Review it before merging, keep PRs as drafts until you've checked them,
and follow each project's contribution guidelines.
"""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _files_from_text(text: str) -> list[str]:
    if not text:
        return []
    return list(dict.fromkeys(re.findall(r"[\w./\\-]+\.py", text)))


def _has_token() -> bool:
    t = os.environ.get("GITHUB_TOKEN")
    return bool(t) and t != "your-github-token-here"


def _verdict(final: dict) -> None:
    status = final.get("status", "")
    reason = final.get("outcome_reason", "")
    label = {
        STATUS_PR_CREATED: "✅ Pull request opened",
        STATUS_PR_COMMITTED_LOCAL: "✅ Fix committed",
        STATUS_PR_READY: "✅ Fix found & validated",
        STATUS_NEEDS_HUMAN_REVIEW: "⚠️ Investigated — needs human review",
        STATUS_ERROR: "❌ Run failed",
    }.get(status, status)
    cls = "af-ok" if status in _SUCCESS_STATUSES else ("af-warn" if status == STATUS_NEEDS_HUMAN_REVIEW else "af-err")
    st.markdown(f'<div class="af-verdict {cls}">{label}<span class="r">{reason}</span></div>',
                unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Result renderers
# --------------------------------------------------------------------------
def _render_explainability(final: dict) -> None:
    issue = final.get("issue") or {}
    rc = final.get("root_cause") or {}
    trace = final.get("call_graph_trace") or {}
    targeted = (final.get("validation") or {}).get("targeted_result") or {}

    with st.container(border=True):
        st.markdown('<div class="af-card-h">Explainability report</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Severity", str(issue.get("severity", "—")).title())
        conf = rc.get("confidence")
        c2.metric("Confidence", f"{conf:.0%}" if isinstance(conf, (int, float)) else "—")
        c3.metric("Tests", f"{targeted.get('passed_count', 0)}/{targeted.get('total', 0)}" if targeted else "—")
        c4.metric("Fix attempts", final.get("attempt", 0))

        if isinstance(conf, (int, float)):
            st.progress(min(max(float(conf), 0.0), 1.0), text="Diagnosis confidence")

        investigated = trace.get("files_visited") or []
        if investigated:
            st.markdown("**Investigated files** (real execution path via AST call graph):")
            st.markdown("".join(f'<span class="af-chip">{f}</span>' for f in investigated),
                        unsafe_allow_html=True)

        if rc.get("root_cause"):
            st.markdown("**Root cause**")
            st.info(rc["root_cause"])
            if rc.get("affected_function"):
                st.caption(f"Location: `{', '.join(rc.get('affected_files', []))}` :: `{rc['affected_function']}()`")

        if rc.get("reasoning"):
            st.markdown("**Reasoning chain**")
            for i, step in enumerate(rc["reasoning"], 1):
                st.markdown(f"**{i}.** {step}")

        with st.expander("Raw structured outputs (JSON)"):
            st.json({"issue": issue, "root_cause": rc}, expanded=False)


def _render_fix(final: dict) -> None:
    patch = final.get("patch") or {}
    diff = patch.get("diff")
    if not diff:
        return
    with st.container(border=True):
        st.markdown('<div class="af-card-h">Proposed fix</div>', unsafe_allow_html=True)
        if final.get("target_file"):
            st.caption(f"File: `{final['target_file']}`")
        st.code(diff, language="diff")
        scope = patch.get("scope_check") or {}
        if scope:
            tag = "⚠️ larger than expected" if scope.get("suspiciously_large") else "minimal change ✓"
            st.caption(f"{scope.get('changed_lines')} of {scope.get('total_lines')} lines changed — {tag}")


def _render_validation(final: dict) -> None:
    validation = final.get("validation") or {}
    targeted = validation.get("targeted_result") or {}
    if not targeted and not validation.get("apply_error"):
        return
    with st.container(border=True):
        st.markdown('<div class="af-card-h">Validation</div>', unsafe_allow_html=True)
        if validation.get("apply_error"):
            st.error(f"The diff did not apply cleanly:\n\n```\n{validation['apply_error']}\n```")
            return
        if validation.get("overall_passed"):
            st.success(f"All tests passed ({targeted.get('passed_count', 0)}/{targeted.get('total', 0)}).")
        else:
            st.error(f"{targeted.get('failed_count', 0)} test(s) failing.")
        if targeted.get("error"):
            st.warning(targeted["error"])
        for t in targeted.get("failed_tests", []):
            with st.expander(f"❌ {t.get('test_name', 'test')}"):
                st.code(t.get("error_message", "(no detail)"))


def _render_timeline(final: dict) -> None:
    history = final.get("history") or []
    if not history:
        return
    with st.container(border=True):
        st.markdown('<div class="af-card-h">Pipeline timeline</div>', unsafe_allow_html=True)
        for step in history:
            st.markdown(f"<span class='af-dot'>●</span> **{step['node']}** — {step['summary']}",
                        unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Pull request (the obvious, deliberate action)
# --------------------------------------------------------------------------
def _open_pr_from_result(final: dict, draft: bool) -> dict:
    """Commits the already-applied fix on a branch and opens a PR, using the
    data from a completed analyze run."""
    rc = final["root_cause"]
    diff = final["patch"]["diff"]
    repo_path = final["repo_path"]
    base = final["base_branch"]
    targeted = (final.get("validation") or {}).get("targeted_result") or {}

    title = build_pr_title(rc)
    description = build_pr_description(rc, diff, targeted)
    slug = re.sub(r"[^a-z0-9]+", "-", (rc.get("affected_function") or "fix").lower()).strip("-") or "fix"
    branch = f"bugfix/aurafix-{slug}"

    commit = create_branch_and_commit(repo_path, branch, commit_message=title, base_branch=base)
    if not commit["success"]:
        return {"success": False, "error": commit["error"]}
    return open_pull_request(repo_path, final["repo_full_name"], branch, title, description,
                             base_branch=base, draft=draft)


def _render_pr_section(final: dict) -> None:
    status = final.get("status")
    repo_full = final.get("repo_full_name")

    with st.container(border=True):
        st.markdown('<div class="af-card-h">Pull request</div>', unsafe_allow_html=True)

        created = st.session_state.get("pr_created")
        if created and created.get("success") and created.get("url"):
            st.markdown(f'<div class="af-verdict af-ok">✅ Pull request opened'
                        f'<span class="r"><a href="{created["url"]}" target="_blank">{created["url"]}</a></span></div>',
                        unsafe_allow_html=True)
            if created.get("mode") == "fork":
                st.caption(f"Opened from your fork `{created.get('fork_full_name')}` "
                           "(no push access to the upstream repo).")
            return

        if status not in _SUCCESS_STATUSES:
            st.caption("A pull request can be opened once AuraFix has a validated fix.")
            return
        if not repo_full:
            st.info("This run was against a **local / sample** repo, so there's no GitHub repo to open a "
                    "PR against. Re-run with a **GitHub URL** to enable pull requests.")
            return
        if not _has_token():
            st.warning("Add a `GITHUB_TOKEN` (scope `public_repo`) to your `.env` to open pull requests, "
                       "then re-run.")
            return

        st.write("Reviewed the fix above? Open it as a pull request on "
                 f"**`{repo_full}`** — AuraFix forks if you can't push, otherwise pushes directly.")
        draft = st.checkbox("Open as draft (recommended)", value=True, key="pr_draft")
        if st.button("Open Pull Request", type="primary", key="pr_btn"):
            with st.spinner("Committing the fix and opening the pull request…"):
                res = _open_pr_from_result(final, draft)
            st.session_state["pr_created"] = res
            if res.get("success"):
                st.rerun()
            else:
                st.error(f"Couldn't open the PR: {res.get('error')}")


# --------------------------------------------------------------------------
# Follow-up chat
# --------------------------------------------------------------------------
def _answer_followup(final: dict, question: str, history: list) -> str:
    rc = final.get("root_cause") or {}
    diff = (final.get("patch") or {}).get("diff", "")
    targeted = (final.get("validation") or {}).get("targeted_result") or {}
    context = (
        f"Bug report: {final.get('bug_report')}\n"
        f"Root cause: {rc.get('root_cause')}\n"
        f"Affected: {rc.get('affected_files')} :: {rc.get('affected_function')}()\n"
        f"Confidence: {rc.get('confidence')}\n"
        f"Reasoning: {rc.get('reasoning')}\n"
        f"Proposed diff:\n{diff}\n"
        f"Tests: {targeted.get('passed_count')}/{targeted.get('total')} passed; "
        f"status={final.get('status')}\n"
    )
    sys_inst = (
        "You are AuraFix's assistant. The user just ran an autonomous bug-fixing pipeline and is asking "
        "follow-up questions about the diagnosis and fix. Answer using ONLY the context provided, concisely "
        "and technically. If they ask you to change the fix, explain conceptually what you'd change — you "
        "cannot re-run the pipeline from this chat; tell them to adjust the bug description and run again."
    )
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    prompt = f"Context about the run:\n{context}\n\nConversation so far:\n{convo}\n\nUser: {question}\n\nAnswer:"
    try:
        return call_llm(prompt, system_instruction=sys_inst)
    except Exception as e:  # noqa: BLE001
        return f"(Couldn't reach the model: {e})"


def _render_chat(final: dict) -> None:
    with st.container(border=True):
        st.markdown('<div class="af-card-h">Ask about this fix</div>', unsafe_allow_html=True)
        st.caption("Ask why this is the root cause, whether the fix is safe, what else to check, etc.")

        for m in st.session_state.get("chat", []):
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        q = st.chat_input("e.g. Why is this the root cause? Could this fix break anything?")
        if q:
            st.session_state.setdefault("chat", []).append({"role": "user", "content": q})
            with st.spinner("Thinking…"):
                ans = _answer_followup(final, q, st.session_state["chat"])
            st.session_state["chat"].append({"role": "assistant", "content": ans})
            st.rerun()


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="af-credit">Made by <b>Vidit Gupta</b></div>', unsafe_allow_html=True)

    # Hero
    st.markdown(
        '<div class="af-hero">'
        '<span class="af-pill">Autonomous AI Agent</span>'
        '<div class="af-title">AuraFix</div>'
        '<p class="af-sub">Describe a bug in plain English. A pipeline of AI agents investigates the '
        'repository, pinpoints the root cause, writes a minimal fix, proves it with tests, and opens a '
        'pull request — autonomously.</p>'
        '<div class="af-langs">Supported languages: '
        '<span class="af-lchip">Python</span><span class="af-lchip">Java</span>'
        '<span class="af-lchip">C</span><span class="af-lchip">C++</span>'
        '<span class="af-lchip">C#</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if _IMPORT_ERROR is not None:
        st.error("AuraFix couldn't start — see the error below.")
        st.code(str(_IMPORT_ERROR))
        msg = str(_IMPORT_ERROR)
        if "GEMINI_API_KEY" in msg or "api key" in msg.lower():
            st.markdown("**Looks like a missing API key.** Set `GEMINI_API_KEY` in `.env` "
                        "(free key at https://aistudio.google.com/apikey) and restart.")
        else:
            st.markdown("**Setup/import issue.** Launch from the project root with the venv active "
                        "(`streamlit run ui/app.py`) and ensure `pip install -r requirements.txt` ran.")
        st.stop()

    # "How it works" strip
    st.markdown(
        '<div class="af-steps">'
        '<div class="af-step"><div class="n">STEP 1</div><div class="t">Point it at a repo</div>'
        '<div class="d">The bundled sample, a local folder, or any GitHub URL.</div></div>'
        '<div class="af-step"><div class="n">STEP 2</div><div class="t">Describe the bug</div>'
        '<div class="d">Plain English. Add a stack trace for extra accuracy.</div></div>'
        '<div class="af-step"><div class="n">STEP 3</div><div class="t">Review & open a PR</div>'
        '<div class="d">See the diagnosis + fix, then open a pull request in one click.</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Sidebar settings
    with st.sidebar:
        st.markdown("### Settings")
        st.caption(f"Model: `{os.environ.get('GEMINI_MODEL')}`")
        st.markdown("&nbsp;", unsafe_allow_html=True)
        confidence_threshold = st.slider("Auto-PR confidence threshold", 0.0, 1.0, 0.7, 0.05,
                                         help="Fixes below this confidence are flagged for human review.")
        max_retries = st.number_input("Max fix attempts", 1, 5, 3)
        max_rag_results = st.number_input("RAG chunks to retrieve", 1, 15, 5)
        st.divider()
        st.markdown("**Credentials**")
        gem_ok = bool(os.environ.get("GEMINI_API_KEY")) and os.environ.get("GEMINI_API_KEY") != "your-key-here"
        st.caption(("✅ Gemini key detected" if gem_ok else "⚠️ GEMINI_API_KEY not set"))
        st.caption(("✅ GitHub token detected" if _has_token() else "ℹ️ GITHUB_TOKEN not set (needed for PRs)"))
        st.divider()
        st.caption("AuraFix · autonomous bug-fixing agent")

    # ---- Inputs ----
    with st.container(border=True):
        st.markdown('<div class="af-card-h">1 · Choose a repository</div>', unsafe_allow_html=True)
        source = st.radio("Repository source",
                          ["Bundled sample repo", "Local path", "GitHub URL"],
                          horizontal=True, label_visibility="collapsed")
        repo_path_input, repo_url_input = "", ""
        sample = None
        if source == "Bundled sample repo":
            sample_name = st.selectbox("Sample", list(BUNDLED_SAMPLES.keys()), label_visibility="collapsed")
            sample = BUNDLED_SAMPLES[sample_name]
            st.caption(f"Using bundled `{sample['path']}` — {sample['blurb']} (no GitHub needed).")
        elif source == "Local path":
            repo_path_input = st.text_input("Local path", placeholder="C:\\path\\to\\repo",
                                            label_visibility="collapsed")
        else:
            repo_url_input = st.text_input("GitHub URL", placeholder="https://github.com/owner/repo",
                                           label_visibility="collapsed")
            st.caption("Pulls the real code and lets you open a pull request from the result.")

    with st.container(border=True):
        st.markdown('<div class="af-card-h">2 · Describe the bug</div>', unsafe_allow_html=True)
        default_bug = sample["bug"] if (source == "Bundled sample repo" and sample) else ""
        bug_report = st.text_area("Bug description", value=default_bug, height=90,
                                  label_visibility="collapsed",
                                  placeholder="e.g. Login page crashes when the password field is empty")
        with st.expander("Advanced (optional) — sharpen accuracy"):
            stack_trace = st.text_area("Stack trace", height=110,
                                       help="File paths here seed the code navigation with exact entry points.")
            failing_test = st.text_input("Failing test name")
            branch_override = st.text_input("Base branch override", placeholder="main / master / develop")

    run = st.button("Run AuraFix", type="primary", use_container_width=True)

    # ---- Run ----
    if run:
        if not bug_report.strip():
            st.error("Please describe the bug first.")
            st.stop()

        repo_path, repo_id, repo_full_name = None, None, None
        if source == "Bundled sample repo":
            repo_path, repo_id = sample["path"], sample["id"]
        elif source == "Local path":
            if not repo_path_input.strip() or not os.path.isdir(repo_path_input.strip()):
                st.error("That local path doesn't exist.")
                st.stop()
            repo_path, repo_id = repo_path_input.strip(), repo_path_input.strip()
        else:
            if not repo_url_input.strip():
                st.error("Please paste a GitHub URL.")
                st.stop()
            with st.spinner(f"Cloning {repo_url_input}…"):
                cloned = clone_repo(repo_url_input.strip())
            if not cloned["success"]:
                st.error(f"Clone failed: {cloned['error']}")
                st.stop()
            repo_path, repo_id, repo_full_name = cloned["repo_path"], repo_url_input.strip(), cloned["repo_full_name"]

        seeds = _files_from_text(stack_trace) + _files_from_text(failing_test)
        request = BugFixRequest(
            bug_report=bug_report, repo_path=repo_path, repo_id=repo_id,
            repo_full_name=repo_full_name, base_branch=branch_override.strip() or None,
            create_pr=False,  # analyze first; the PR is a deliberate one-click action afterward
            seed_files=seeds or None, confidence_threshold=confidence_threshold,
            max_retries=int(max_retries), max_rag_results=int(max_rag_results),
        )
        # New run -> clear any prior PR / chat state.
        st.session_state.pop("pr_created", None)
        st.session_state["chat"] = []
        with st.spinner("Running the pipeline: issue → indexing → navigation → root cause → patch → validation…"):
            st.session_state["result"] = run_pipeline(request)

    # ---- Results ----
    final = st.session_state.get("result")
    if final:
        st.markdown("---")
        _verdict(final)
        _render_explainability(final)
        _render_fix(final)
        _render_validation(final)
        _render_pr_section(final)
        _render_chat(final)
        _render_timeline(final)
    else:
        st.caption("Results will appear here after you run AuraFix.")

    # About / how it works — always available (handy for the hosted version)
    st.markdown("---")
    with st.expander("About AuraFix — how it works"):
        st.markdown(ABOUT_MD)
    st.caption("AuraFix · LangGraph · Google Gemini · ChromaDB · PyGithub — every fix is AI-generated; review before merging.")


if __name__ == "__main__":
    main()
