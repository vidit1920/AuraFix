"""
app.py - the AuraFix dashboard.

A hosted, multi-user Streamlit UI for the autonomous bug-fixing pipeline:

  * Sign in with Firebase (Google / GitHub), verified on the server.
  * Bring your own LLM key (Gemini / OpenAI / Anthropic / any
    OpenAI-compatible endpoint). No shared server key, so no shared quota or
    cost, and each user's credentials stay in their own session only.
  * Describe a bug, run the pipeline, review the diagnosis and fix, and open
    a pull request under your own GitHub token.

Layout: three pages under a top navigation bar.
  * Fix a bug     - the guided workspace (repo -> bug -> run -> results).
  * Setup         - connect a model (with step-by-step key instructions per
                    provider), optional GitHub token, advanced run settings.
  * How it works  - what the pipeline does, what results mean, FAQ.

Security notes live next to the code that needs them (auth gate, rate
limiting, input validation, safe rendering). Run in production via
``python server.py`` (adds security headers); local dev can use
``streamlit run ui/app.py``.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys

# Make the project root importable regardless of how the app is launched.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

st.set_page_config(page_title="AuraFix", page_icon=":material/auto_fix_high:", layout="wide")

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
    from github_integration.credentials import set_github_token
    from github_integration.pr_agent import (
        build_pr_title,
        build_pr_description,
        create_branch_and_commit,
        open_pull_request,
    )
    from agents.llm_client import call_llm
    from agents.runtime import LLMSettings, set_settings, clear_settings, DEFAULT_CHAT_MODELS, sfr_available
    from security import auth, sanitize, ratelimit
    from security.sanitize import ValidationError
    from ui.firebase_auth import firebase_login
    from validators.test_validator import which as tool_which
except Exception as e:  # noqa: BLE001
    _IMPORT_ERROR = e

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
INSTAGRAM_URL = "https://www.instagram.com/_viditgupta/"
LINKEDIN_URL = "https://www.linkedin.com/in/gupta-vidit/"

BUNDLED_SAMPLES = {
    "Login crash on empty password (Python)": {
        "path": "sample-repo", "id": "local-sample-repo",
        "bug": "Login page crashes when the password field is empty",
        "blurb": "A login flow that crashes on an empty password (missing None check).",
    },
    "Wrong discount math (Python)": {
        "path": "sample-shop", "id": "local-sample-shop",
        "bug": "Discounts are applied incorrectly. A 10% discount subtracts 10 from the price "
               "instead of taking 10% off.",
        "blurb": "A Python checkout that computes discounts with the wrong formula.",
    },
    "Wrong discount math (Java / Maven)": {
        "path": "sample-java", "id": "local-sample-java",
        "bug": "Discounts are applied incorrectly. A 10% discount subtracts 10 from the price "
               "instead of taking 10% off.",
        "blurb": "A Java checkout with the wrong discount formula.",
    },
}
_SAMPLE_BUGS = {s["bug"] for s in BUNDLED_SAMPLES.values()}

SRC_SAMPLE, SRC_GITHUB, SRC_LOCAL = "Sample project", "GitHub repository", "Local folder"

# Provider metadata, including the step-by-step "get a key" guide shown on the
# Setup page. Models are editable text, not locked dropdowns, since provider
# catalogues change and users know their own models.
PROVIDERS = {
    "Google Gemini": {
        "id": "gemini", "needs_base": False,
        "model_help": "e.g. gemini-flash-latest, gemini-2.5-flash-lite",
        "guide": {
            "badge": "Free tier", "badge_kind": "ok",
            "site": "Google AI Studio", "url": "https://aistudio.google.com/apikey",
            "steps": [
                "Open <b>Google AI Studio</b> and sign in with your Google account.",
                "Click <b>Create API key</b>. If asked, choose or create a Google Cloud project.",
                "Copy the key. It starts with <code>AIza</code>.",
                "Paste it into the <b>API key</b> field on this page, then click <b>Test connection</b>.",
            ],
            "note": "No card needed. The free tier has a daily request limit, and each AuraFix "
                    "run uses a handful of requests.",
        },
    },
    "OpenAI": {
        "id": "openai", "needs_base": False,
        "model_help": "e.g. gpt-4o-mini, gpt-4o",
        "guide": {
            "badge": "Paid", "badge_kind": "muted",
            "site": "OpenAI Platform", "url": "https://platform.openai.com/api-keys",
            "steps": [
                "Sign in at <b>platform.openai.com</b>.",
                "Add credit under <b>Settings &rsaquo; Billing</b>. API usage is pay as you go.",
                "Open <b>API keys</b> and click <b>Create new secret key</b>.",
                "Copy the key right away (it is shown once). It starts with <code>sk-</code>.",
                "Paste it into the <b>API key</b> field on this page, then click <b>Test connection</b>.",
            ],
            "note": "A small model such as gpt-4o-mini keeps each run very cheap.",
        },
    },
    "Anthropic (Claude)": {
        "id": "anthropic", "needs_base": False, "default_model": "claude-3-5-haiku-latest",
        "model_help": "e.g. claude-3-5-haiku-latest, claude-3-5-sonnet-latest",
        "guide": {
            "badge": "Paid", "badge_kind": "muted",
            "site": "Anthropic Console", "url": "https://console.anthropic.com/settings/keys",
            "steps": [
                "Sign in to the <b>Anthropic Console</b>.",
                "Add credit under <b>Billing</b>.",
                "Open <b>API keys</b> and click <b>Create key</b>.",
                "Copy the key. It starts with <code>sk-ant-</code>.",
                "Paste it into the <b>API key</b> field on this page, then click <b>Test connection</b>.",
            ],
            "note": "Use a fast, low-cost model for routine bugs.",
        },
    },
    "OpenAI-compatible (Groq, OpenRouter, ...)": {
        "id": "openai_compatible", "needs_base": True,
        "default_base": "https://api.groq.com/openai/v1",
        "model_help": "Model id from your provider, e.g. llama-3.3-70b-versatile",
        "guide": {
            "badge": "Free tier (Groq)", "badge_kind": "ok",
            "site": "Groq Console", "url": "https://console.groq.com/keys",
            "steps": [
                "Sign in to the <b>Groq Console</b> (free).",
                "Open <b>API Keys</b> and click <b>Create API Key</b>.",
                "Copy the key. It starts with <code>gsk_</code>.",
                "Keep the Base URL as <code>https://api.groq.com/openai/v1</code> and the model as "
                "<code>llama-3.3-70b-versatile</code>.",
                "Paste the key into the <b>API key</b> field, then click <b>Test connection</b>.",
            ],
            "note": "Using OpenRouter instead? Get a key at openrouter.ai/keys and set the Base URL "
                    "to <code>https://openrouter.ai/api/v1</code>.",
        },
    },
}

GITHUB_TOKEN_URL = "https://github.com/settings/tokens/new?scopes=public_repo&description=AuraFix"

EMBED_SFR = "SFR-Embedding-Code 400M (local, best for code)"
EMBED_MINILM = "MiniLM (local, small and fast)"
EMBED_CHOICES = {
    EMBED_SFR: "sfr",
    EMBED_MINILM: "local",
    "Automatic (match provider)": "auto",
    "Google Gemini": "gemini",
    "OpenAI": "openai",
}

RETRIEVAL_MULTI = "Multi-agent RAG (planner, 3 retrievers, critic)"
RETRIEVAL_FAST = "Fast (one search round, no extra AI calls)"
RETRIEVAL_CHOICES = {RETRIEVAL_MULTI: "multi_agent", RETRIEVAL_FAST: "fast"}

# Friendly names for pipeline nodes (history entries) in progress + timeline.
NODE_LABELS = {
    "issue_agent": "Understood the bug report",
    "repo_indexing": "Indexed the repository",
    "query_planner": "Planned the code search",
    "retrievers": "Searched the code",
    "relevance_critic": "Checked what was found",
    "code_navigation": "Traced the code path",
    "root_cause_agent": "Found the root cause",
    "patch_agent": "Wrote a fix",
    "test_validator": "Ran the tests",
    "pr_agent": "Prepared the pull request",
    "human_review": "Flagged for human review",
    "pipeline": "Pipeline stopped",
}

_SUCCESS_STATUSES = (
    {STATUS_PR_CREATED, STATUS_PR_COMMITTED_LOCAL, STATUS_PR_READY} if not _IMPORT_ERROR else set()
)

# Widget keys whose values must survive switching pages. Streamlit drops a
# widget's state when it is not rendered on the current page, so these are
# re-assigned at the top of every run (the documented multipage pattern).
_PERSIST_KEYS = (
    ["provider_label", "embed_label", "opt_retrieval", "embedkey_gemini", "embedkey_openai", "gh_token",
     "opt_conf", "opt_retries", "opt_chunks", "src", "sample_name", "repo_url", "repo_local",
     "bug_report", "stack_trace", "failing_test", "branch_override", "pr_draft"]
    + [f"{p}_{m['id']}" for m in PROVIDERS.values() for p in ("model", "base", "key")]
)

# --------------------------------------------------------------------------
# Styling. Warm charcoal theme, one amber accent, 8px radius,
# AA contrast throughout. System fonts only (CSP allows self-hosted fonts).
# --------------------------------------------------------------------------
CSS = """
<style>
:root {
  --af-bg: #151311; --af-card: #1E1B18; --af-raised: #29251F; --af-border: #3A332C;
  --af-text: #FAFAF9; --af-body: #D6D3D1; --af-sub: #A8A29E;
  --af-accent: #F59E0B; --af-accent-text: #FBBF24; --af-accent-soft: rgba(245,158,11,.14);
  --af-on-accent: #2A1700;
  --af-ok: #4ADE80; --af-info: #93C5FD; --af-warn: #FB923C; --af-err: #F87171;
  --af-mono: ui-monospace, "JetBrains Mono", "Cascadia Code", "SF Mono", Consolas, monospace;
}
.block-container {max-width: 1120px; padding-top: 4.25rem; padding-bottom: 2rem;}
a {color: var(--af-accent-text);}
a:focus-visible, button:focus-visible {outline: 2px solid var(--af-accent); outline-offset: 2px;}

/* Primary buttons: dark text on amber for 9:1 contrast (white would be 2.1:1). */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"] p {
  color: var(--af-on-accent) !important; font-weight: 600;}
[data-testid="stBaseButton-primary"]:disabled, [data-testid="stBaseButton-primary"]:disabled p {
  color: var(--af-sub) !important;}

/* Page header */
.af-kicker {font-family: var(--af-mono); font-size: .75rem; font-weight: 600; letter-spacing: .08em;
  text-transform: uppercase; color: var(--af-accent-text); margin-bottom: 6px;}
.af-h1 {font-size: 2rem; font-weight: 700; line-height: 1.15; color: var(--af-text); margin: 0 0 8px;}
.af-lead {color: var(--af-body); font-size: 1rem; line-height: 1.55; max-width: 720px; margin: 0 0 18px;}

/* Progress stepper */
.af-stepper {display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 0 0 18px;
  padding: 0; list-style: none;}
.af-step {display: flex; gap: 10px; align-items: flex-start; padding: 12px 14px;
  border: 1px solid var(--af-border); border-radius: 8px; background: var(--af-card);}
.af-step .dot {flex: none; width: 24px; height: 24px; border-radius: 50%; display: flex;
  align-items: center; justify-content: center; font-size: .78rem; font-weight: 700;
  border: 1.5px solid var(--af-border); color: var(--af-sub);}
.af-step .dot svg {width: 14px; height: 14px;}
.af-step.done .dot {background: var(--af-accent); border-color: var(--af-accent); color: var(--af-on-accent);}
.af-step.current {border-color: var(--af-accent);}
.af-step.current .dot {border-color: var(--af-accent); color: var(--af-accent-text);}
.af-step .t {font-weight: 600; color: var(--af-text); font-size: .9rem; line-height: 1.3;}
.af-step .d {color: var(--af-sub); font-size: .8rem; line-height: 1.4; margin-top: 2px;}
@media (max-width: 760px) {.af-stepper {grid-template-columns: 1fr 1fr;}}

/* Card headings */
.af-card-h {display: flex; align-items: center; gap: 10px; font-size: 1.02rem; font-weight: 600;
  color: var(--af-text); margin: 2px 0 4px;}
.af-num {flex: none; width: 24px; height: 24px; border-radius: 6px; display: inline-flex;
  align-items: center; justify-content: center; font-size: .8rem; font-weight: 700;
  background: var(--af-accent-soft); color: var(--af-accent-text);}
.af-card-sub {color: var(--af-sub); font-size: .87rem; line-height: 1.5; margin: 0 0 10px;}

/* Pills */
.af-pill {display: inline-flex; align-items: center; gap: 6px; padding: 2px 10px; border-radius: 999px;
  font-size: .78rem; font-weight: 600; border: 1px solid; white-space: nowrap;}
.af-pill.ok {color: var(--af-ok); border-color: rgba(74,222,128,.45); background: rgba(74,222,128,.1);}
.af-pill.warn {color: var(--af-warn); border-color: rgba(251,146,60,.45); background: rgba(251,146,60,.1);}
.af-pill.err {color: var(--af-err); border-color: rgba(248,113,113,.45); background: rgba(248,113,113,.1);}
.af-pill.muted {color: var(--af-sub); border-color: var(--af-border); background: transparent;}

/* Key/value rows in side panels */
.af-kv {display: flex; justify-content: space-between; align-items: center; gap: 12px;
  padding: 9px 0; border-top: 1px solid var(--af-border); font-size: .88rem;}
.af-kv:first-of-type {border-top: none;}
.af-kv .k {color: var(--af-sub);}
.af-kv .v {color: var(--af-text); font-weight: 500; text-align: right; overflow-wrap: anywhere;}

/* Numbered instructions */
.af-ol {counter-reset: s; list-style: none; padding: 0; margin: 6px 0 4px;}
.af-ol li {counter-increment: s; position: relative; padding: 2px 0 12px 36px; color: var(--af-body);
  font-size: .92rem; line-height: 1.5;}
.af-ol li::before {content: counter(s); position: absolute; left: 0; top: 0; width: 24px; height: 24px;
  border-radius: 50%; background: var(--af-accent-soft); color: var(--af-accent-text); font-weight: 700;
  font-size: .78rem; display: flex; align-items: center; justify-content: center;}
.af-ol b {color: var(--af-text);}
.af-ol code, .af-note code, .af-card-sub code {font-family: var(--af-mono); font-size: .82em;
  background: var(--af-raised); padding: 1px 5px; border-radius: 4px; color: var(--af-text);}
.af-ul {margin: 4px 0 0; padding-left: 18px; color: var(--af-body); font-size: .9rem; line-height: 1.55;}
.af-ul li {margin-bottom: 6px;}
.af-note {color: var(--af-sub); font-size: .84rem; line-height: 1.5;}

/* Result banner */
.af-banner {border: 1px solid var(--af-border); border-left: 4px solid var(--c); background: var(--af-card);
  border-radius: 8px; padding: 14px 18px; margin: 6px 0 14px;}
.af-banner .h {font-weight: 700; color: var(--c); font-size: 1.05rem;}
.af-banner .r {color: var(--af-body); font-size: .92rem; line-height: 1.5; margin-top: 4px;}
.af-banner .n {color: var(--af-sub); font-size: .86rem; line-height: 1.5; margin-top: 8px;}
.af-banner .n b {color: var(--af-text);}

/* Pipeline stages (How it works) */
.af-stage {display: flex; gap: 14px; padding: 14px 0; border-top: 1px solid var(--af-border);}
.af-stage:first-child {border-top: none; padding-top: 4px;}
.af-stage .t {font-weight: 600; color: var(--af-text); font-size: .95rem;}
.af-stage .d {color: var(--af-body); font-size: .9rem; line-height: 1.5; margin-top: 2px;}

/* Status badge (used inside result tabs) */
.af-badge {display: inline-block; border: 1px solid; border-radius: 8px; padding: 8px 12px;
  font-weight: 600; font-size: .92rem;}
.af-badge .af-detail {display: block; font-weight: 400; color: var(--af-body); font-size: .85rem; margin-top: 4px;}

/* Footer */
.af-foot {margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--af-border); display: flex;
  justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
  color: var(--af-sub); font-size: .82rem;}
.af-foot .links {display: flex; gap: 6px; align-items: center;}
.af-foot a {color: var(--af-sub); display: inline-flex; padding: 6px; border-radius: 6px;}
.af-foot a:hover {color: var(--af-text); background: var(--af-card);}
.af-foot svg {width: 16px; height: 16px;}

/* Sign-in screen */
.af-auth-brand {text-align: center; margin: 7vh 0 20px 0;}
.af-logo-badge {display: inline-flex; align-items: center; justify-content: center; width: 48px; height: 48px;
  border-radius: 12px; margin-bottom: 14px; background: var(--af-accent-soft);
  border: 1px solid rgba(245,158,11,.5); color: var(--af-accent-text);}
.af-logo-badge svg {width: 24px; height: 24px;}
.af-logo-word {font-size: 1.75rem; font-weight: 700; color: var(--af-text); letter-spacing: -.01em; line-height: 1;}
.af-auth-tag {color: var(--af-body); font-size: .95rem; margin-top: 8px;}
.af-auth-head {font-size: 1.12rem; font-weight: 600; color: var(--af-text); text-align: center; margin: 2px 0 4px;}
.af-auth-sub {color: var(--af-sub); font-size: .87rem; text-align: center; margin: 0 auto 16px; max-width: 300px; line-height: 1.45;}
.af-auth-foot {text-align: center; color: var(--af-sub); font-size: .8rem; margin-top: 16px; line-height: 1.5;}
.af-auth-card [data-testid="stVerticalBlockBorderWrapper"] {max-width: 360px; margin: 0 auto;}
</style>
"""

# Inline SVG marks (no emoji icons). Rendered inline so they inherit colour.
_IG_SVG = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
           'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="5"/>'
           '<circle cx="12" cy="12" r="4"/><circle cx="17.3" cy="6.7" r="1" fill="currentColor" stroke="none"/></svg>')
_LI_SVG = ('<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M4.98 3.5A2.5 2.5 0 1 0 5 8.5a2.5 2.5 0 0 0 0-5zM3 9h4v12H3zM9 9h3.8v1.7h.05c.53-.95 1.83-1.95 3.76-1.95 4.02 0 4.76 2.5 4.76 5.76V21h-4v-5.3c0-1.26-.02-2.9-1.76-2.9-1.77 0-2.04 1.38-2.04 2.8V21H9z"/></svg>')
_CHECK_SVG = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" '
              'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>')
_LOGO_SVG = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
             'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18v3h3l6.3-6.3a4 4 0 0 0 5.4-5.4l-2.6 2.6-2.4-.6-.6-2.4z"/></svg>')

# Filled in by main() once the pages exist, so any page can link to another.
PAGES: dict = {}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _html(markup: str) -> None:
    """Render trusted, pre-escaped markup. Any user/model text must be passed
    through sanitize.escape before it reaches here."""
    st.markdown(markup, unsafe_allow_html=True)


def _esc(text) -> str:
    return sanitize.escape(str(text))


def _page_header(kicker: str, title: str, lead: str) -> None:
    text, account = st.columns([5, 1.3], gap="medium")
    with text:
        _html(f'<div class="af-kicker">{kicker}</div>'
              f'<div class="af-h1" role="heading" aria-level="1">{title}</div>'
              f'<p class="af-lead">{lead}</p>')
    with account:
        _render_account_menu()


def _card_head(title: str, sub: str = "", num: str | None = None) -> None:
    num_html = f'<span class="af-num">{num}</span>' if num else ""
    sub_html = f'<p class="af-card-sub">{sub}</p>' if sub else ""
    _html(f'<div class="af-card-h">{num_html}{title}</div>{sub_html}')


def _pill(kind: str, text: str) -> str:
    return f'<span class="af-pill {kind}">{_esc(text)}</span>'


def _identity() -> str:
    """Rate-limit key: the signed-in uid, else the client IP, else a session
    fallback. Never personally identifying beyond what we already hold."""
    user = st.session_state.get("auth_user") or {}
    if user.get("uid"):
        return "uid:" + user["uid"]
    try:
        ip = st.context.ip_address
    except Exception:  # noqa: BLE001
        ip = None
    return "ip:" + (ip or "anon")


def _status(kind: str, label: str, detail: str = "") -> None:
    """Flat, escaped status badge. ``label`` is from a fixed internal set;
    ``detail`` may be model/user text, so it is HTML-escaped."""
    colors = {"ok": "#4ADE80", "warn": "#FB923C", "err": "#F87171", "muted": "#A8A29E"}
    c = colors.get(kind, "#A8A29E")
    detail_html = f'<span class="af-detail">{_esc(detail)}</span>' if detail else ""
    _html(f'<div class="af-badge" style="border-color:{c};color:{c}">{_esc(label)}{detail_html}</div>')


def _keep_widget_state() -> None:
    for k in _PERSIST_KEYS:
        if k in st.session_state:
            st.session_state[k] = st.session_state[k]


def _init_defaults() -> None:
    ss = st.session_state
    ss.setdefault("provider_label", next(iter(PROVIDERS)))
    for meta in PROVIDERS.values():
        pid = meta["id"]
        ss.setdefault(f"model_{pid}", meta.get("default_model") or DEFAULT_CHAT_MODELS.get(pid, ""))
        if meta["needs_base"]:
            ss.setdefault(f"base_{pid}", meta.get("default_base", ""))
    ss.setdefault("embed_label", EMBED_SFR if sfr_available() else EMBED_MINILM)
    ss.setdefault("opt_retrieval", RETRIEVAL_MULTI)
    ss.setdefault("opt_conf", 0.7)
    ss.setdefault("opt_retries", 3)
    ss.setdefault("opt_chunks", 5)
    ss.setdefault("src", SRC_SAMPLE)
    first = next(iter(BUNDLED_SAMPLES))
    ss.setdefault("sample_name", first)
    ss.setdefault("bug_report", BUNDLED_SAMPLES[first]["bug"])
    ss.setdefault("pr_draft", True)


def _key_fingerprint(pid: str, model: str, key: str) -> str:
    """Identifies the exact (provider, model, key) combo that passed Test
    connection, without keeping a second copy of the key itself."""
    return hashlib.sha256(f"{pid}|{model}|{key}".encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# Authentication gate
# --------------------------------------------------------------------------
def _current_user() -> dict | None:
    return st.session_state.get("auth_user")


def _render_login() -> None:
    _, mid, _ = st.columns([1, 1.5, 1])
    with mid:
        _html('<div class="af-auth-brand">'
              f'<div class="af-logo-badge">{_LOGO_SVG}</div>'
              '<div class="af-logo-word">AuraFix</div>'
              '<div class="af-auth-tag">From bug report to tested fix and pull request.</div>'
              '</div>')

        _html('<div class="af-auth-card">')
        with st.container(border=True):
            _html('<div class="af-auth-head">Sign in to continue</div>'
                  '<div class="af-auth-sub">Use your Google or GitHub account. '
                  'First time here? Your account is created automatically.</div>')
            signout = st.session_state.get("_pending_signout", False)
            persist = False
            if not signout:
                persist = st.checkbox(f"Keep me signed in for {auth.session_max_days()} days",
                                      value=False, key="stay_signed_in")
            result = firebase_login(auth.client_config(), auth.enabled_providers(),
                                    persist=persist, signout=signout, key="login")
            if result and isinstance(result, dict):
                nonce = result.get("nonce")
                if nonce and nonce != st.session_state.get("_consumed_nonce"):
                    st.session_state["_consumed_nonce"] = nonce
                    if result.get("signedOut"):
                        st.session_state.pop("_pending_signout", None)
                        st.session_state.pop("auth_user", None)
                        st.rerun()
                    else:
                        try:
                            claims = auth.verify_token(result.get("idToken", ""))
                        except auth.AuthError as e:
                            st.error(str(e))
                            return
                        st.session_state["auth_user"] = claims
                        st.rerun()
        _html('</div>')

        _html('<div class="af-auth-foot">Secured by Firebase. We only receive your name and '
              'email to identify your session. Your model key is never shared.</div>')


def _require_auth() -> bool:
    """Returns True if the request may proceed. Renders the login screen and
    stops otherwise. Fails closed when auth is required but misconfigured."""
    if not auth.auth_enabled():
        # Operator explicitly disabled the gate (local/dev). Proceed with a
        # synthetic local user so the rest of the app has an identity.
        if not _current_user():
            st.session_state["auth_user"] = {
                "uid": "local", "email": "", "name": "Local user", "provider": "none",
            }
        return True

    err = auth.config_error()
    if err:
        st.error(err)
        st.stop()

    if _current_user():
        return True

    _render_login()
    st.stop()


def _sign_out() -> None:
    # Drop the local session (including any keys the user pasted) and ask the
    # sign-in component to also clear Firebase's persisted session, so "keep me
    # signed in" doesn't just log the user straight back in.
    st.session_state["_pending_signout"] = True
    for k in ["auth_user", "_consumed_nonce", "result", "pr_created", "chat", "key_verified", "gh_token",
              "embedkey_gemini", "embedkey_openai"] + [f"key_{m['id']}" for m in PROVIDERS.values()]:
        st.session_state.pop(k, None)


# --------------------------------------------------------------------------
# Credentials (read from session state, so they apply on every page)
# --------------------------------------------------------------------------
def _provider_meta() -> dict:
    return PROVIDERS[st.session_state.get("provider_label", next(iter(PROVIDERS)))]


def _build_settings() -> LLMSettings:
    """Construct validated LLMSettings from the Setup page values. Raises
    ValidationError (shown to the user) if the key/base URL are missing or
    malformed."""
    meta = _provider_meta()
    pid = meta["id"]
    default_model = meta.get("default_model") or DEFAULT_CHAT_MODELS.get(pid, "")

    api_key = sanitize.clean_api_key(st.session_state.get(f"key_{pid}", ""))
    model = (st.session_state.get(f"model_{pid}") or default_model).strip()
    if not model:
        raise ValidationError("Please enter a model name.")

    base = None
    if meta["needs_base"]:
        base = sanitize.validate_base_url(st.session_state.get(f"base_{pid}", ""))

    embed_provider = EMBED_CHOICES.get(st.session_state.get("embed_label"), "local")
    if embed_provider == "sfr" and not sfr_available():
        embed_provider = "local"  # extras not installed on this server: use MiniLM
    embed_key = ""
    if embed_provider in ("gemini", "openai"):
        raw = st.session_state.get(f"embedkey_{embed_provider}", "")
        if raw.strip():
            embed_key = sanitize.clean_api_key(raw)

    return LLMSettings(
        chat_provider=pid, chat_model=model, chat_api_key=api_key, chat_base_url=base,
        embed_provider=embed_provider, embed_api_key=embed_key,
    )


def _apply_session_credentials() -> bool:
    """Install this session's LLM + GitHub credentials into the request context
    so any code path (run or chat) uses them. Returns True if an LLM key is
    configured."""
    set_github_token(st.session_state.get("gh_token", ""))
    try:
        settings = _build_settings()
    except ValidationError:
        # Always clear, so a previous session's key can never linger on a
        # reused worker thread and be used by this one.
        clear_settings()
        return False
    set_settings(settings)
    return True


def _key_state() -> tuple[str, str]:
    """('ok'|'warn'|'muted', label) describing the model connection."""
    meta = _provider_meta()
    pid = meta["id"]
    key = st.session_state.get(f"key_{pid}", "").strip()
    if not key:
        return "muted", "Not connected"
    model = st.session_state.get(f"model_{pid}", "")
    if st.session_state.get("key_verified") == _key_fingerprint(pid, model, key):
        return "ok", "Connected, verified"
    return "warn", "Key added, not tested"


# --------------------------------------------------------------------------
# Page: Setup
# --------------------------------------------------------------------------
def page_setup() -> None:
    _page_header(
        "Setup",
        "Connect your AI model",
        "AuraFix runs on your own API key, so you control the cost and nothing is shared. "
        "Your keys stay in this browser session only. They are never stored on the server "
        "or written to logs.",
    )
    _html('<p class="af-note" style="margin-top:-8px">New here? Choose <b>Google Gemini</b>: '
          'it has a free tier and takes about a minute to set up.</p>')

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        with st.container(border=True):
            _card_head("Model provider", "Pick a provider, then paste your key.", num="1")
            st.selectbox("Provider", list(PROVIDERS.keys()), key="provider_label",
                         on_change=lambda: st.session_state.pop("key_verified", None))
            meta = _provider_meta()
            pid = meta["id"]
            st.text_input("Model", key=f"model_{pid}", help=meta["model_help"])
            if meta["needs_base"]:
                st.text_input("Base URL", key=f"base_{pid}",
                              help="Your provider's OpenAI-compatible endpoint.")
            st.text_input("API key", type="password", key=f"key_{pid}",
                          placeholder="Paste your key here",
                          help="Kept only in this browser session.")

            kind, label = _key_state()
            test = st.button("Test connection", icon=":material/bolt:", use_container_width=True,
                             disabled=kind == "muted")
            _html(f'<div class="af-kv"><span class="k">Status</span><span class="v">{_pill(kind, label)}</span></div>')
            if test:
                _test_connection()

    with right:
        with st.container(border=True):
            g = meta["guide"]
            _html(f'<div class="af-card-h">How to get a {_esc(g["site"])} key '
                  f'{_pill(g["badge_kind"], g["badge"])}</div>')
            steps = "".join(f"<li>{s}</li>" for s in g["steps"])
            _html(f'<ol class="af-ol">{steps}</ol>')
            st.link_button(f"Open {g['site']}", g["url"], icon=":material/open_in_new:",
                           use_container_width=True)
            _html(f'<p class="af-note" style="margin-top:10px">{g["note"]}</p>')

    st.write("")
    left2, right2 = st.columns([1.05, 1], gap="large")
    with left2:
        with st.container(border=True):
            _card_head("GitHub token", "Optional. Only needed to open a pull request with the fix, "
                       "under your own GitHub account.", num="2")
            st.text_input("GitHub token", type="password", key="gh_token",
                          placeholder="ghp_...", help="Classic token with the public_repo scope "
                          "(or repo for private repositories).")
            has_gh = bool(st.session_state.get("gh_token", "").strip())
            _html(_pill("ok" if has_gh else "muted", "Token added" if has_gh else "No token"))
    with right2:
        with st.container(border=True):
            _html('<div class="af-card-h">How to create a GitHub token</div>')
            _html('<ol class="af-ol">'
                  '<li>Sign in to GitHub and open <b>Settings &rsaquo; Developer settings &rsaquo; '
                  'Personal access tokens &rsaquo; Tokens (classic)</b>. The button below opens it '
                  'with the right scope already ticked.</li>'
                  '<li>Set an expiry date. Keep <code>public_repo</code> ticked '
                  '(tick <code>repo</code> instead for private repositories).</li>'
                  '<li>Click <b>Generate token</b> and copy it. It starts with <code>ghp_</code>.</li>'
                  '<li>Paste it into the <b>GitHub token</b> field on this page.</li></ol>')
            st.link_button("Create a GitHub token", GITHUB_TOKEN_URL, icon=":material/open_in_new:",
                           use_container_width=True)
            _html('<p class="af-note" style="margin-top:10px">If you cannot push to the repository, '
                  'AuraFix forks it and opens the pull request from your fork.</p>')

    st.write("")
    with st.expander("Advanced settings", icon=":material/tune:"):
        a, b = st.columns(2, gap="large")
        with a:
            st.selectbox("Code search (embeddings)", list(EMBED_CHOICES.keys()), key="embed_label",
                         help="Used to index the repository. The local models need no key and no quota.")
            embed_provider = EMBED_CHOICES.get(st.session_state.get("embed_label"), "local")
            if embed_provider == "sfr":
                if sfr_available():
                    st.caption("Salesforce's code-search model (400M parameters). Downloads about "
                               "870 MB the first time and needs about 2 GB of memory. Research and "
                               "non-commercial use only (CC-BY-NC-4.0).")
                else:
                    st.warning("SFR isn't installed on this server, so MiniLM is used instead. "
                               "Install it with requirements-sfr.txt (needs about 2 GB of memory).")
            st.selectbox("Code retrieval", list(RETRIEVAL_CHOICES.keys()), key="opt_retrieval",
                         help="Multi-agent RAG plans several searches, runs semantic, keyword and "
                              "call-graph search, and has a critic check the results (2 to 3 extra "
                              "AI calls). Fast runs one search round with no extra AI calls.")
            if embed_provider in ("gemini", "openai"):
                st.text_input("Embedding key (leave empty to reuse the model key)",
                              type="password", key=f"embedkey_{embed_provider}")
        with b:
            st.slider("Auto pull request confidence", 0.0, 1.0, step=0.05, key="opt_conf",
                      help="Fixes below this confidence are flagged for human review.")
            st.number_input("Max fix attempts", 1, 5, key="opt_retries",
                            help="How many times AuraFix may revise a fix whose tests fail.")
            st.number_input("Code chunks to retrieve", 1, 15, key="opt_chunks",
                            help="More context can help on large repositories, but uses more tokens.")

    st.write("")
    if _key_state()[0] != "muted":
        st.page_link(PAGES["fix"], label="Continue to Fix a bug", icon=":material/arrow_forward:")


def _test_connection() -> None:
    decision = ratelimit.check_chat(_identity())
    if not decision.allowed:
        st.warning(decision.reason)
        return
    try:
        settings = _build_settings()
    except ValidationError as e:
        st.error(str(e))
        return
    set_settings(settings)
    with st.spinner("Contacting the provider..."):
        try:
            reply = call_llm("Reply with the single word OK.", max_retries=1)
        except Exception as e:  # noqa: BLE001 — message already scrubbed of keys
            st.session_state.pop("key_verified", None)
            msg = str(e).strip() or type(e).__name__
            st.error(f"Could not connect: {msg[:300]}")
            return
    if not (reply or "").strip():
        st.error("The provider answered with an empty reply. Check the model name.")
        return
    st.session_state["key_verified"] = _key_fingerprint(
        settings.chat_provider, settings.chat_model, settings.chat_api_key)
    st.rerun()


# --------------------------------------------------------------------------
# Page: How it works
# --------------------------------------------------------------------------
_STAGES = [
    ("Understand", "Reads your bug report and turns it into a structured issue: which module, how "
                   "severe, and which files are likely involved."),
    ("Index", "Splits the repository into chunks and embeds them with a code-search model "
              "(SFR-Embedding-Code 400M when available) so code can be found by meaning."),
    ("Search with a team of agents", "A planner agent writes several targeted searches. Three "
              "retriever agents run them: semantic search, keyword search, and a call-graph lookup. "
              "Their results are fused, then a critic agent keeps only what bears on the bug and asks "
              "for follow-up searches if something is missing."),
    ("Navigate", "Builds a call graph and follows the real execution path from the likely entry "
                 "points, so the diagnosis rests on how the code actually runs."),
    ("Diagnose", "Combines the issue, the call path and the retrieved code into a root cause, the "
                 "exact function at fault, and a confidence score."),
    ("Fix", "Writes the smallest change that fixes the root cause, as a diff."),
    ("Validate", "Applies the diff and runs the project's own tests. If they fail, AuraFix revises "
                 "the fix using the failure output, up to the attempt limit."),
]

_OUTCOMES = [
    ("ok", "Fix found and validated", "The tests pass and confidence is high. You can open a pull "
                                      "request in one click."),
    ("warn", "Needs human review", "AuraFix found something but could not prove it: tests kept "
                                   "failing, confidence was low, or the repository has no tests. "
                                   "The proposed fix is still shown for you to judge."),
    ("err", "Run did not complete", "Usually a key, quota or network problem. Check Setup and run "
                                    "again."),
]

_FAQ = [
    ("Which provider should I choose?",
     "Start with Google Gemini. It has a free tier and works well for typical bugs. Groq is another "
     "free option. OpenAI and Anthropic are paid but very capable."),
    ("Why did my run end in \"Needs human review\"?",
     "The most common reason is that the repository has no automated tests, so AuraFix cannot prove "
     "the fix works. Add even one test that reproduces the bug and run again. Other reasons: the "
     "tests still failed after every attempt, or the diagnosis confidence was below your threshold."),
    ("How long does a run take?",
     "Usually one to three minutes. Free-tier keys are rate limited, so a run can take longer when "
     "the provider asks AuraFix to slow down. The progress panel shows each stage as it finishes."),
    ("Which repositories work?",
     "Public GitHub repositories. Python projects with pytest tests work best. Java (Maven or Gradle), "
     "C# and C/C++ are supported when their build tools are available on the server."),
    ("Is my API key safe?",
     "Your key lives only in your browser session for as long as it is open. It is never saved on "
     "the server, never logged, and it is cleared when you sign out."),
    ("Will AuraFix change my repository?",
     "Not on its own. It works on a temporary copy. A pull request is opened only when you click "
     "Open pull request, and you can open it as a draft."),
]


def page_guide() -> None:
    _page_header(
        "Guide",
        "How AuraFix works",
        "AuraFix is an autonomous agent that takes a plain-English bug report, finds the root cause in "
        "your code, writes a minimal fix, proves it with your tests, and prepares a pull request.",
    )

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        with st.container(border=True):
            _card_head("What happens when you click Run")
            rows = "".join(
                f'<div class="af-stage"><span class="af-num">{i}</span><div>'
                f'<div class="t">{t}</div><div class="d">{d}</div></div></div>'
                for i, (t, d) in enumerate(_STAGES, 1)
            )
            _html(rows)
    with right:
        with st.container(border=True):
            _card_head("Using AuraFix in 4 steps")
            _html('<ol class="af-ol">'
                  '<li><b>Connect a model</b> on the Setup page with your own API key.</li>'
                  '<li><b>Choose a repository</b>: a sample project, or any public GitHub repository.</li>'
                  '<li><b>Describe the bug</b> like you would in an issue: what you did, what happened, '
                  'what you expected.</li>'
                  '<li><b>Run AuraFix</b> and review the root cause, the fix and the test results. '
                  'Open a pull request if you are happy with it.</li></ol>')
            st.page_link(PAGES["setup"], label="Go to Setup", icon=":material/key:")
            st.page_link(PAGES["fix"], label="Go to Fix a bug", icon=":material/build:")
        st.write("")
        with st.container(border=True):
            _card_head("What the result means")
            for kind, title, desc in _OUTCOMES:
                _html(f'<div style="margin:8px 0 12px">{_pill(kind, title)}'
                      f'<div class="af-note" style="margin-top:6px">{desc}</div></div>')

    st.write("")
    _html('<div class="af-card-h" style="margin-top:8px">Frequently asked questions</div>')
    for q, a in _FAQ:
        with st.expander(q):
            st.write(a)


# --------------------------------------------------------------------------
# Page: Fix a bug (workspace)
# --------------------------------------------------------------------------
def _on_source_change() -> None:
    src = st.session_state.get("src")
    bug = st.session_state.get("bug_report", "")
    if src == SRC_SAMPLE:
        st.session_state["bug_report"] = BUNDLED_SAMPLES[st.session_state["sample_name"]]["bug"]
    elif bug in _SAMPLE_BUGS:
        st.session_state["bug_report"] = ""


def _on_sample_change() -> None:
    st.session_state["bug_report"] = BUNDLED_SAMPLES[st.session_state["sample_name"]]["bug"]


def _render_stepper(have_key: bool, repo_ready: bool, bug_ready: bool, has_result: bool) -> None:
    steps = [
        ("Connect a model", "Your own API key, on Setup", have_key),
        ("Choose a repository", "Sample or public GitHub repo", repo_ready),
        ("Describe the bug", "What happens vs. what you expect", bug_ready),
        ("Run and review", "Root cause, fix, tests, PR", has_result),
    ]
    current = next((i for i, s in enumerate(steps) if not s[2]), None)
    items = []
    for i, (t, d, done) in enumerate(steps):
        cls = "done" if done else ("current" if i == current else "")
        dot = _CHECK_SVG if done else str(i + 1)
        state = "completed" if done else ("current step" if i == current else "not started")
        items.append(f'<li class="af-step {cls}" aria-label="Step {i + 1}: {t}, {state}">'
                     f'<span class="dot">{dot}</span><div><div class="t">{t}</div>'
                     f'<div class="d">{d}</div></div></li>')
    _html(f'<ol class="af-stepper">{"".join(items)}</ol>')


def page_fix() -> None:
    have_key = st.session_state.get("_have_key", False)
    _page_header(
        "Autonomous bug fixing",
        "Fix a bug",
        "Point AuraFix at a repository and describe what is broken. It finds the root cause, writes a "
        "fix, runs your tests to prove it, and prepares a pull request.",
    )

    src = st.session_state.get("src", SRC_SAMPLE)
    repo_ready = src == SRC_SAMPLE or bool(
        st.session_state.get("repo_url" if src == SRC_GITHUB else "repo_local", "").strip())
    bug_ready = bool(st.session_state.get("bug_report", "").strip())
    _render_stepper(have_key, repo_ready, bug_ready, bool(st.session_state.get("result")))

    main_col, side_col = st.columns([1.75, 1], gap="large")

    with side_col:
        _render_session_panel(have_key)

    with main_col:
        sources = [SRC_SAMPLE, SRC_GITHUB]
        if sanitize.local_path_enabled():
            sources.append(SRC_LOCAL)
        if st.session_state.get("src") not in sources:
            st.session_state["src"] = SRC_SAMPLE

        with st.container(border=True):
            _card_head("Choose a repository",
                       "Try a sample project first, or paste any public GitHub repository.", num="1")
            st.segmented_control("Repository source", sources, key="src", required=True,
                                 on_change=_on_source_change, label_visibility="collapsed")
            source = st.session_state.get("src", SRC_SAMPLE)
            if source == SRC_SAMPLE:
                st.selectbox("Sample project", list(BUNDLED_SAMPLES.keys()), key="sample_name",
                              on_change=_on_sample_change)
                sample = BUNDLED_SAMPLES[st.session_state["sample_name"]]
                st.caption(sample["blurb"] + " The bug description below is filled in for you.")
                if sample["path"] == "sample-java" and not tool_which("mvn"):
                    st.warning("Maven isn't installed on this server, so AuraFix can find and fix "
                               "this bug but can't run the Java tests to verify the fix.")
            elif source == SRC_LOCAL:
                st.text_input("Local path", key="repo_local", placeholder="/path/to/repo")
            else:
                st.text_input("GitHub repository URL", key="repo_url",
                              placeholder="https://github.com/owner/repo")
                st.caption("Public repositories only. AuraFix needs the repository's tests to prove "
                           "a fix; without tests the result is flagged for human review.")

        with st.container(border=True):
            _card_head("Describe the bug",
                       "Say what you did, what happened, and what you expected. Mention a file or "
                       "function if you know it.", num="2")
            st.text_area("Bug description", key="bug_report", height=110,
                         placeholder="e.g. Logging in as admin with a wrong password still succeeds. "
                                     "Expected: the login is rejected.")
            with st.expander("Add more detail (optional)", icon=":material/add:"):
                st.text_area("Stack trace", key="stack_trace", height=110,
                             help="File paths in a stack trace point AuraFix straight at the right code.")
                st.text_input("Failing test name", key="failing_test")
                st.text_input("Base branch", key="branch_override",
                              placeholder="Defaults to the repository's main branch")

        run = st.button("Run AuraFix", type="primary", icon=":material/play_arrow:",
                        use_container_width=True, disabled=not have_key)
        if not have_key:
            st.caption("Connect a model on the Setup page to enable this button.")
        else:
            st.caption("A run usually takes 1 to 3 minutes. You will see each stage as it finishes.")

    if run:
        _handle_run()

    final = st.session_state.get("result")
    if final:
        _render_results(final)


def _render_session_panel(have_key: bool) -> None:
    with st.container(border=True):
        _card_head("Your setup")
        meta = _provider_meta()
        kind, label = _key_state()
        has_gh = bool(st.session_state.get("gh_token", "").strip())
        model = st.session_state.get(f"model_{meta['id']}", "")
        _html(f'<div class="af-kv"><span class="k">Model</span><span class="v">{_pill(kind, label)}</span></div>'
              f'<div class="af-kv"><span class="k">Provider</span><span class="v">{_esc(st.session_state.get("provider_label"))}</span></div>'
              f'<div class="af-kv"><span class="k">Model name</span><span class="v">{_esc(model or "-")}</span></div>'
              f'<div class="af-kv"><span class="k">Pull requests</span><span class="v">'
              f'{_pill("ok", "Enabled") if has_gh else _pill("muted", "Add a GitHub token")}</span></div>')
        st.page_link(PAGES["setup"], label="Connect a model" if not have_key else "Change setup",
                     icon=":material/key:")

    st.write("")
    with st.container(border=True):
        _card_head("Tips for a good result")
        _html('<ul class="af-ul">'
              '<li>Describe one bug per run.</li>'
              '<li>Include the exact error message or a stack trace if you have one.</li>'
              '<li>Repositories with tests give a verified fix. Without tests the fix is flagged for '
              'review.</li></ul>')
        st.page_link(PAGES["guide"], label="How AuraFix works", icon=":material/menu_book:")


# --------------------------------------------------------------------------
# Running the pipeline
# --------------------------------------------------------------------------
def _handle_run() -> None:
    ss = st.session_state
    try:
        bug_report = sanitize.clamp(ss.get("bug_report", ""), sanitize.MAX_BUG_REPORT, "Bug description")
        if not bug_report:
            st.error("Please describe the bug first.")
            return
        stack_trace = sanitize.clamp(ss.get("stack_trace", ""), sanitize.MAX_STACK_TRACE, "Stack trace")
        failing_test = sanitize.clamp(ss.get("failing_test", ""), sanitize.MAX_SHORT_FIELD, "Failing test")
        branch_override = sanitize.clamp(ss.get("branch_override", ""), sanitize.MAX_SHORT_FIELD, "Branch")
    except ValidationError as e:
        st.error(str(e))
        return

    # Rate-limit the (expensive) run per identity before doing any work.
    decision = ratelimit.check_run(_identity())
    if not decision.allowed:
        wait = f" Try again in about {decision.retry_after}s." if decision.retry_after else ""
        st.warning(decision.reason + wait)
        return

    source = ss.get("src", SRC_SAMPLE)
    repo_full_name = None
    with st.status("Running AuraFix...", expanded=True) as status:
        try:
            if source == SRC_SAMPLE:
                sample = BUNDLED_SAMPLES[ss["sample_name"]]
                repo_path, repo_id = sample["path"], sample["id"]
            elif source == SRC_LOCAL:
                repo_path = sanitize.validate_local_path(ss.get("repo_local", ""))
                repo_id = repo_path
            else:
                target = sanitize.validate_github_url(ss.get("repo_url", ""))
                st.write(f"Cloning {target.full_name}...")
                cloned = clone_repo(target.url)
                if not cloned["success"]:
                    status.update(label="Could not clone the repository", state="error")
                    _status("err", "Clone failed", str(cloned.get("error", "")))
                    st.caption("Check that the URL is right and the repository is public.")
                    return
                repo_path, repo_id, repo_full_name = cloned["repo_path"], target.url, cloned["repo_full_name"]
                st.markdown(":material/check_circle: Cloned the repository")
        except ValidationError as e:
            status.update(label="Check the repository", state="error")
            st.error(str(e))
            return

        seeds = _files_from_text(stack_trace) + _files_from_text(failing_test)
        request = BugFixRequest(
            bug_report=bug_report, repo_path=repo_path, repo_id=repo_id,
            repo_full_name=repo_full_name, base_branch=branch_override or None,
            create_pr=False,
            seed_files=seeds or None, confidence_threshold=float(ss.get("opt_conf", 0.7)),
            max_retries=int(ss.get("opt_retries", 3)), max_rag_results=int(ss.get("opt_chunks", 5)),
            retrieval_mode=RETRIEVAL_CHOICES.get(ss.get("opt_retrieval"), "multi_agent"),
        )
        ss.pop("pr_created", None)
        ss["chat"] = []

        def on_step(entry: dict) -> None:
            node = entry.get("node", "")
            label = NODE_LABELS.get(node, node)
            status.update(label=f"Running AuraFix: {label.lower()}...")
            st.markdown(f":material/check_circle: **{label}**: {entry.get('summary', '')}")

        final = run_pipeline(request, on_step=on_step)
        ss["result"] = final

        state = final.get("status")
        if state in _SUCCESS_STATUSES:
            status.update(label="Done: fix found and validated", state="complete", expanded=False)
        elif state == STATUS_NEEDS_HUMAN_REVIEW:
            status.update(label="Done: needs human review", state="complete", expanded=False)
        else:
            status.update(label="The run did not complete", state="error", expanded=False)


def _files_from_text(text: str) -> list:
    if not text:
        return []
    return list(dict.fromkeys(re.findall(r"[\w./\\-]+\.py", text)))


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
def _render_results(final: dict) -> None:
    st.write("")
    head, clear = st.columns([4, 1], vertical_alignment="bottom")
    with head:
        _html('<div class="af-card-h" style="font-size:1.25rem;margin-top:8px">Results</div>')
    with clear:
        if st.button("Clear results", icon=":material/close:", use_container_width=True):
            for k in ("result", "pr_created", "chat"):
                st.session_state.pop(k, None)
            st.rerun()

    _verdict(final)
    tabs = st.tabs(["Summary", "Proposed fix", "Tests", "Pull request", "Ask AuraFix", "Timeline"])
    with tabs[0]:
        _render_explainability(final)
    with tabs[1]:
        _render_fix(final)
    with tabs[2]:
        _render_validation(final)
    with tabs[3]:
        _render_pr_section(final)
    with tabs[4]:
        _render_chat(final)
    with tabs[5]:
        _render_timeline(final)


def _verdict(final: dict) -> None:
    status = final.get("status", "")
    reason = final.get("outcome_reason", "")
    _targeted = (final.get("validation") or {}).get("targeted_result") or {}
    if status in _SUCCESS_STATUSES:
        color, title = "#4ADE80", "Fix found and validated"
        nxt = "<b>Next:</b> review the proposed fix, then open a pull request from the Pull request tab."
    elif status == STATUS_NEEDS_HUMAN_REVIEW:
        color, title = "#FB923C", "Needs human review"
        if _targeted.get("toolchain_missing"):
            nxt = ("<b>Next:</b> check the proposed fix yourself. This server can't run this "
                   "language's tests, so the fix was not verified.")
        elif _targeted.get("no_tests"):
            nxt = ("<b>Next:</b> check the proposed fix yourself. To let AuraFix verify fixes, add a "
                   "test that reproduces the bug and run again.")
        else:
            nxt = ("<b>Next:</b> read the root cause and proposed fix. You can add a stack trace or "
                   "more detail to the bug description and run again.")
    else:
        color, title = "#F87171", "Run did not complete"
        nxt = "<b>Next:</b> use Test connection on the Setup page to check your key, then run again."
        # On error, show a generic reason to users (debug off); detail stays in logs.
        if final.get("error_kind") == "provider":
            nxt = ("<b>Next:</b> fix the key or model on the Setup page, or wait a minute if the "
                   "provider is busy or the quota is used up, then run again.")
        # Provider errors are plain-language and key-free, so they are shown;
        # anything else stays generic for users (detail is in the logs).
        elif status == STATUS_ERROR and not auth.dev_mode():
            reason = "Something went wrong while running the pipeline."
    _html(f'<div class="af-banner" style="--c:{color}" role="status"><div class="h">{title}</div>'
          f'<div class="r">{_esc(reason)}</div><div class="n">{nxt}</div></div>')


def _render_explainability(final: dict) -> None:
    issue = final.get("issue") or {}
    rc = final.get("root_cause") or {}
    trace = final.get("call_graph_trace") or {}
    targeted = (final.get("validation") or {}).get("targeted_result") or {}

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Severity", str(issue.get("severity", "-")).title())
        conf = rc.get("confidence")
        c2.metric("Confidence", f"{conf:.0%}" if isinstance(conf, (int, float)) else "-")
        c3.metric("Tests passed", f"{targeted.get('passed_count', 0)}/{targeted.get('total', 0)}" if targeted else "-")
        c4.metric("Fix attempts", final.get("attempt", 0))

    if rc.get("root_cause"):
        with st.container(border=True):
            _card_head("Root cause")
            st.write(rc["root_cause"])
            if rc.get("affected_function"):
                files = ", ".join(rc.get("affected_files", []))
                st.caption(f"Location: {files} :: {rc['affected_function']}()")
            if rc.get("reasoning"):
                st.markdown("**How AuraFix got there**")
                for i, step in enumerate(rc["reasoning"], 1):
                    st.markdown(f"{i}. {step}")

    _render_retrieval(final)

    investigated = trace.get("files_visited") or []
    if investigated:
        with st.expander(f"Files investigated ({len(investigated)})", icon=":material/folder_open:"):
            st.caption("The real execution path, traced through the call graph.")
            st.code("\n".join(str(f) for f in investigated), language="text")

    with st.expander("Raw structured output (JSON)", icon=":material/data_object:"):
        st.json({"issue": issue, "root_cause": rc}, expanded=False)


_AGENT_NAMES = {"semantic": "Semantic", "lexical": "Keyword", "structural": "Call graph"}


def _render_retrieval(final: dict) -> None:
    rounds = final.get("retrieval_trace") or []
    chunks = final.get("rag_chunks") or []
    if not rounds and not chunks:
        return
    plan = final.get("retrieval_plan") or {}
    info = final.get("index_info") or {}
    mode = "multi-agent" if final.get("retrieval_mode") == "multi_agent" else "fast"
    with st.expander(f"How the code was found ({mode} RAG, {len(rounds)} round(s))",
                     icon=":material/travel_explore:"):
        if info:
            st.caption(f"Index: {info.get('chunks')} chunks embedded with {info.get('embedding')}.")
        planner = "AI query planner" if plan.get("source") == "llm" else "rule-based plan"
        if plan.get("fallback_reason"):
            planner += " (the AI planner was unavailable)"
        st.markdown(f"**Searches** ({planner})")
        for q in plan.get("queries") or []:
            st.markdown(f"- {q.get('text')} ({q.get('intent')})")
        if plan.get("keywords"):
            st.caption("Keywords: " + ", ".join(plan["keywords"]))
        for r in rounds:
            found = r.get("found") or {}
            st.markdown(f"**Round {r.get('round')}**: semantic {found.get('semantic', 0)}, keyword "
                        f"{found.get('lexical', 0)}, call graph {found.get('structural', 0)} hits; "
                        f"{r.get('candidates', 0)} unique chunks after fusion.")
            if r.get("round", 1) > 1 and r.get("queries"):
                st.caption("Follow-up searches: " + "; ".join(r["queries"]))
            for agent, err in (r.get("errors") or {}).items():
                st.caption(f"{_AGENT_NAMES.get(agent, agent)} search failed: {err}")
            critic = r.get("critic") or {}
            if critic.get("source") == "llm":
                verdict = "enough context" if critic.get("sufficient") else "needs more context"
                line = f"Critic kept {len(critic.get('kept') or [])} chunk(s): {verdict}."
                if critic.get("missing"):
                    line += f" Missing: {critic['missing']}"
                st.caption(line)
            elif critic:
                st.caption("Critic: rule-based ranking" + (" (the AI critic was unavailable)."
                                                          if critic.get("fallback_reason") else "."))
        if chunks:
            st.markdown("**Context given to the root cause agent**")
            st.dataframe(
                [{"File": c["filename"], "Chunk": c["chunk_index"],
                  "Found by": ", ".join(_AGENT_NAMES.get(x, x) for x in c.get("sources") or []) or "-",
                  "Critic": "Relevant" if c.get("kept_by_critic") else "-"}
                 for c in chunks],
                hide_index=True, use_container_width=True,
            )


def _render_fix(final: dict) -> None:
    patch = final.get("patch") or {}
    diff = patch.get("diff")
    if not diff:
        st.caption("No fix was generated in this run.")
        return
    if final.get("target_file"):
        st.caption(f"File: {final['target_file']}")
    st.code(diff, language="diff")
    scope = patch.get("scope_check") or {}
    if scope:
        tag = "larger than expected" if scope.get("suspiciously_large") else "minimal change"
        st.caption(f"{scope.get('changed_lines')} of {scope.get('total_lines')} lines changed ({tag}).")


def _render_validation(final: dict) -> None:
    validation = final.get("validation") or {}
    targeted = validation.get("targeted_result") or {}
    if not targeted and not validation.get("apply_error"):
        st.caption("Tests were not run in this run.")
        return
    if validation.get("apply_error"):
        _status("err", "The fix could not be applied to the code")
        st.code(str(validation["apply_error"]), language="text")
        return
    if targeted.get("no_tests"):
        _status("warn", "No tests found",
                "This repository has no tests, so the fix could not be verified automatically.")
        return
    if targeted.get("toolchain_missing"):
        _status("warn", "Tests could not run", str(targeted.get("error", "")))
        return
    if validation.get("overall_passed"):
        _status("ok", "All tests passed", f"{targeted.get('passed_count', 0)}/{targeted.get('total', 0)}")
    else:
        _status("err", "Tests failing", f"{targeted.get('failed_count', 0)} failing")
    if targeted.get("error"):
        st.warning(str(targeted["error"]))
    for t in targeted.get("failed_tests", []):
        with st.expander(str(t.get("test_name", "test"))):
            st.code(str(t.get("error_message", "(no detail)")), language="text")


def _render_timeline(final: dict) -> None:
    history = final.get("history") or []
    if not history:
        st.caption("No steps were recorded.")
        return
    for i, step in enumerate(history, 1):
        node = step.get("node", "")
        # st.markdown escapes HTML by default, so these strings render safely.
        st.markdown(f"{i}. **{NODE_LABELS.get(node, node)}**: {step.get('summary', '')}")


# --------------------------------------------------------------------------
# Pull request
# --------------------------------------------------------------------------
def _open_pr_from_result(final: dict, draft: bool) -> dict:
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

    created = st.session_state.get("pr_created")
    if created and created.get("success") and created.get("url"):
        safe_url = sanitize.safe_external_url(created.get("url"))
        _status("ok", "Pull request opened")
        if safe_url:
            st.link_button("Open the pull request", safe_url, icon=":material/open_in_new:")
        else:
            st.caption("The pull request was created.")
        if created.get("mode") == "fork":
            st.caption(f"Opened from your fork {created.get('fork_full_name')} "
                       "(you do not have push access to the original repository).")
        return

    if status not in _SUCCESS_STATUSES:
        st.caption("A pull request can be opened once AuraFix has a fix that passes the tests.")
        return
    if not repo_full:
        st.info("This run used a sample project, so there is no GitHub repository to open a pull "
                "request against. Run AuraFix on a GitHub repository to enable this.")
        return
    if not st.session_state.get("gh_token", "").strip():
        st.warning("Add a GitHub token on the Setup page to open a pull request.")
        st.page_link(PAGES["setup"], label="Go to Setup", icon=":material/key:")
        return

    st.write(f"Open the fix as a pull request on {repo_full}. AuraFix forks the repository if you "
             "cannot push to it.")
    st.checkbox("Open as draft (recommended)", key="pr_draft")
    if st.button("Open pull request", type="primary", icon=":material/merge:", key="pr_btn"):
        with st.spinner("Committing the fix and opening the pull request..."):
            res = _open_pr_from_result(final, bool(st.session_state.get("pr_draft", True)))
        st.session_state["pr_created"] = res
        if res.get("success"):
            st.rerun()
        else:
            _status("err", "Could not open the pull request", str(res.get("error", "")))


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
        "You are AuraFix's assistant. The user just ran an autonomous bug-fixing pipeline and is "
        "asking follow-up questions about the diagnosis and fix. Answer using ONLY the context "
        "provided, concisely and technically. If they ask you to change the fix, explain what you "
        "would change; you cannot re-run the pipeline from this chat, so tell them to adjust the "
        "bug description and run again."
    )
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    prompt = f"Context about the run:\n{context}\n\nConversation so far:\n{convo}\n\nUser: {question}\n\nAnswer:"
    try:
        return call_llm(prompt, system_instruction=sys_inst)
    except Exception:  # noqa: BLE001
        return "Sorry, the model could not be reached just now. Please try again."


def _render_chat(final: dict) -> None:
    st.caption("Ask why this is the root cause, whether the fix is safe, or what else to check.")
    for m in st.session_state.get("chat", []):
        with st.chat_message(m["role"]):
            # st.markdown escapes raw HTML by default, so this is injection-safe.
            st.markdown(m["content"])

    q = st.chat_input("e.g. Why is this the root cause? Could this fix break anything?")
    if q:
        try:
            question = sanitize.clamp(q, sanitize.MAX_BUG_REPORT, "Question")
        except ValidationError as e:
            st.error(str(e))
            return
        decision = ratelimit.check_chat(_identity())
        if not decision.allowed:
            st.warning(decision.reason)
            return
        st.session_state.setdefault("chat", []).append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            ans = _answer_followup(final, question, st.session_state["chat"])
        st.session_state["chat"].append({"role": "assistant", "content": ans})
        st.rerun()


# --------------------------------------------------------------------------
# App shell
# --------------------------------------------------------------------------
def _render_account_menu() -> None:
    user = _current_user() or {}
    who = user.get("name") or user.get("email") or "Signed in"
    with st.popover(who, icon=":material/account_circle:", use_container_width=True):
        if user.get("email"):
            st.caption(user["email"])
        if auth.auth_enabled():
            if st.button("Sign out", icon=":material/logout:", use_container_width=True):
                _sign_out()
                st.rerun()
        else:
            st.caption("Sign-in is turned off on this server.")


def _render_footer() -> None:
    _html(f'<div class="af-foot"><span>AuraFix. Built by Vidit Gupta.</span><span class="links">'
          f'<a href="{INSTAGRAM_URL}" target="_blank" rel="noopener noreferrer" aria-label="Instagram">{_IG_SVG}</a>'
          f'<a href="{LINKEDIN_URL}" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn">{_LI_SVG}</a>'
          f'</span></div>')


def main() -> None:
    _html(CSS)

    if _IMPORT_ERROR is not None:
        st.error("AuraFix could not start due to a configuration problem.")
        if os.environ.get("APP_ENV", "development").lower() != "production":
            st.code(str(_IMPORT_ERROR))
        st.stop()

    _require_auth()
    _keep_widget_state()
    _init_defaults()
    st.session_state["_have_key"] = _apply_session_credentials()

    PAGES["fix"] = st.Page(page_fix, title="Fix a bug", icon=":material/build:", default=True)
    PAGES["setup"] = st.Page(page_setup, title="Setup", icon=":material/key:", url_path="setup")
    PAGES["guide"] = st.Page(page_guide, title="How it works", icon=":material/menu_book:",
                             url_path="how-it-works")
    nav = st.navigation([PAGES["fix"], PAGES["setup"], PAGES["guide"]], position="top")

    nav.run()
    _render_footer()


if __name__ == "__main__":
    main()
