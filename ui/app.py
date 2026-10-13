"""
app.py - the AuraFix dashboard.

A hosted, multi-user Streamlit UI for the autonomous bug-fixing pipeline:

  * Sign in with Firebase (Google / GitHub), verified on the server.
  * Bring your own LLM key (Gemini / OpenAI / Anthropic / any
    OpenAI-compatible endpoint). No shared server key, so no shared quota or
    cost, and each user's credentials stay in their own session only.
  * Describe a bug, run the pipeline, review the diagnosis and fix, and open
    a pull request under your own GitHub token.

Security notes live next to the code that needs them (auth gate, rate
limiting, input validation, safe rendering). Run in production via
``python server.py`` (adds security headers); local dev can use
``streamlit run ui/app.py``.
"""

from __future__ import annotations

import os
import sys

# Make the project root importable regardless of how the app is launched.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

st.set_page_config(page_title="AuraFix", layout="wide")

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
    from agents.runtime import LLMSettings, set_settings, clear_settings, DEFAULT_CHAT_MODELS
    from security import auth, sanitize, ratelimit
    from security.sanitize import ValidationError
    from ui.firebase_auth import firebase_login
except Exception as e:  # noqa: BLE001
    _IMPORT_ERROR = e

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
INSTAGRAM_URL = "https://www.instagram.com/_viditgupta/"
LINKEDIN_URL = "https://www.linkedin.com/in/gupta-vidit/"

BUNDLED_SAMPLES = {
    "Login crash on empty password (auth)": {
        "path": "sample-repo", "id": "local-sample-repo",
        "bug": "Login page crashes when the password field is empty",
        "blurb": "a login flow that crashes on an empty password (missing None check)",
    },
    "Wrong discount math, Python (shopping cart)": {
        "path": "sample-shop", "id": "local-sample-shop",
        "bug": "Discounts are applied incorrectly. A 10% discount subtracts 10 from the price "
               "instead of taking 10% off.",
        "blurb": "a Python checkout that computes discounts with the wrong formula",
    },
    "Wrong discount math, Java (Maven)": {
        "path": "sample-java", "id": "local-sample-java",
        "bug": "Discounts are applied incorrectly. A 10% discount subtracts 10 from the price "
               "instead of taking 10% off.",
        "blurb": "a Java/Maven checkout with the wrong discount formula (needs Maven to run its tests)",
    },
}

# Provider metadata for the sidebar. Models are editable text, not locked
# dropdowns, since provider catalogues change and users know their own models.
PROVIDERS = {
    "Google Gemini": {
        "id": "gemini", "needs_base": False,
        "key_help": "Free key: https://aistudio.google.com/apikey",
        "model_help": "e.g. gemini-flash-latest, gemini-2.5-flash-lite",
    },
    "OpenAI": {
        "id": "openai", "needs_base": False,
        "key_help": "Key: https://platform.openai.com/api-keys",
        "model_help": "e.g. gpt-4o-mini, gpt-4o",
    },
    "Anthropic (Claude)": {
        "id": "anthropic", "needs_base": False, "default_model": "claude-3-5-haiku-latest",
        "key_help": "Key: https://console.anthropic.com/settings/keys",
        "model_help": "e.g. claude-3-5-haiku-latest, claude-3-5-sonnet-latest",
    },
    "OpenAI-compatible (Groq, OpenRouter, Together, ...)": {
        "id": "openai_compatible", "needs_base": True,
        "default_base": "https://api.groq.com/openai/v1",
        "key_help": "Your provider's API key",
        "model_help": "Model id from your provider, e.g. llama-3.3-70b-versatile",
    },
}

EMBED_CHOICES = {
    "Automatic (match provider, else local)": "auto",
    "Local, no key needed (recommended)": "local",
    "Google Gemini": "gemini",
    "OpenAI": "openai",
}

_SUCCESS_STATUSES = (
    {STATUS_PR_CREATED, STATUS_PR_COMMITTED_LOCAL, STATUS_PR_READY} if not _IMPORT_ERROR else set()
)

# --------------------------------------------------------------------------
# Styling: flat, muted dark, one accent. No gradients, shadows, glass, hover
# animation, orbs, or external fonts.
# --------------------------------------------------------------------------
CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1000px;}

/* Top-right credit with social links */
.af-credit {position: fixed; top: 10px; right: 16px; z-index: 1000; font-size: .8rem;
    color: #9aa0ac; display: flex; align-items: center; gap: 10px;}
.af-credit .by {color: #c7cbd4;}
.af-credit a {color: #9aa0ac; display: inline-flex; align-items: center;}
.af-credit a:hover {color: #e6e8ee;}
.af-credit svg {width: 16px; height: 16px;}

.af-kicker {font-size: .72rem; font-weight: 600; letter-spacing: .12em;
    text-transform: uppercase; color: #8c93a0; margin-bottom: 6px;}
.af-title {font-size: 2.4rem; font-weight: 700; line-height: 1.1; margin: 0 0 8px 0; color: #eef0f4;}
.af-sub {color: #9aa0ac; font-size: 1.02rem; max-width: 680px; margin: 0;}
.af-langs {margin-top: 12px; color: #8c93a0; font-size: .85rem;}

.af-card-h {font-size: 1rem; font-weight: 600; margin: 2px 0 10px 0; color: #e6e8ee;}

/* Numbered step list (deliberately not three cards in a row) */
.af-steps {list-style: none; padding: 0; margin: 14px 0 0 0;}
.af-steps li {display: flex; gap: 12px; padding: 8px 0; border-top: 1px solid #242833;}
.af-steps li:first-child {border-top: none;}
.af-steps .n {color: #5b8def; font-weight: 700; font-size: .85rem; min-width: 1.4em;}
.af-steps .t {font-weight: 600; color: #e6e8ee;}
.af-steps .d {color: #9aa0ac; font-size: .9rem;}

/* Flat status badge (no emoji, no gradient, no shadow) */
.af-badge {display: inline-block; border: 1px solid; border-radius: .25rem;
    padding: 8px 12px; font-weight: 600; font-size: .92rem;}
.af-badge .af-detail {display: block; font-weight: 400; color: #c7cbd4; font-size: .85rem; margin-top: 4px;}

.af-note {color: #8c93a0; font-size: .82rem;}
a {color: #5b8def;}
hr {border-color: #242833;}
</style>
"""

# Monochrome brand marks (not decorative icons; no emoji). Rendered inline so
# they inherit the muted text colour.
_IG_SVG = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
           'stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="5"/>'
           '<circle cx="12" cy="12" r="4"/><circle cx="17.3" cy="6.7" r="1" fill="currentColor" stroke="none"/></svg>')
_LI_SVG = ('<svg viewBox="0 0 24 24" fill="currentColor"><path d="M4.98 3.5A2.5 2.5 0 1 0 5 8.5a2.5 2.5 0 0 0 0-5zM3 9h4v12H3zM9 9h3.8v1.7h.05c.53-.95 1.83-1.95 3.76-1.95 4.02 0 4.76 2.5 4.76 5.76V21h-4v-5.3c0-1.26-.02-2.9-1.76-2.9-1.77 0-2.04 1.38-2.04 2.8V21H9z"/></svg>')


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
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
    colors = {"ok": "#7fcf9f", "warn": "#e0c179", "err": "#e59a9a", "muted": "#9aa0ac"}
    c = colors.get(kind, "#9aa0ac")
    detail_html = f'<span class="af-detail">{sanitize.escape(detail)}</span>' if detail else ""
    st.markdown(
        f'<div class="af-badge" style="border-color:{c};color:{c}">{sanitize.escape(label)}{detail_html}</div>',
        unsafe_allow_html=True,
    )


def _dev_detail(e: Exception) -> None:
    """Show error internals only in dev mode; never leak them in production."""
    if not _IMPORT_ERROR and auth.dev_mode():
        st.code(str(e))


# --------------------------------------------------------------------------
# Authentication gate
# --------------------------------------------------------------------------
def _current_user() -> dict | None:
    return st.session_state.get("auth_user")


def _render_login() -> None:
    st.markdown('<div class="af-kicker">Autonomous AI Agent</div>'
                '<div class="af-title">AuraFix</div>'
                '<p class="af-sub">Sign in to run the autonomous bug-fixing pipeline. '
                'You bring your own model key; nothing is charged to a shared account.</p>',
                unsafe_allow_html=True)
    st.write("")

    with st.container(border=True):
        st.markdown('<div class="af-card-h">Sign in</div>', unsafe_allow_html=True)
        result = firebase_login(auth.client_config(), auth.enabled_providers(), key="login")
        if result and isinstance(result, dict):
            nonce = result.get("nonce")
            if nonce and nonce != st.session_state.get("_consumed_nonce"):
                st.session_state["_consumed_nonce"] = nonce
                try:
                    claims = auth.verify_token(result.get("idToken", ""))
                except auth.AuthError as e:
                    st.error(str(e))
                    return
                st.session_state["auth_user"] = claims
                st.rerun()
        st.markdown('<p class="af-note">Sign-in is handled by Firebase. We receive only your '
                    'basic profile (name, email) to identify your session.</p>', unsafe_allow_html=True)


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


# --------------------------------------------------------------------------
# Sidebar: account + bring-your-own-key
# --------------------------------------------------------------------------
def _render_sidebar() -> dict:
    """Renders account info and the BYOK config, returns the run options."""
    with st.sidebar:
        user = _current_user() or {}
        st.markdown("### Account")
        who = user.get("name") or user.get("email") or "Signed in"
        st.caption(f"Signed in as {who}")
        if auth.auth_enabled() and st.button("Sign out", use_container_width=True):
            for k in ("auth_user", "_consumed_nonce", "result", "pr_created", "chat"):
                st.session_state.pop(k, None)
            st.rerun()

        st.divider()
        st.markdown("### Model (your key)")
        provider_label = st.selectbox("Provider", list(PROVIDERS.keys()), key="provider_label")
        meta = PROVIDERS[provider_label]
        pid = meta["id"]
        default_model = meta.get("default_model") or DEFAULT_CHAT_MODELS.get(pid, "")

        st.text_input("Model", value=st.session_state.get(f"model_{pid}", default_model),
                      key=f"model_{pid}", help=meta["model_help"])
        if meta["needs_base"]:
            st.text_input("Base URL", value=st.session_state.get(f"base_{pid}", meta.get("default_base", "")),
                          key=f"base_{pid}", help="Your provider's OpenAI-compatible endpoint.")
        st.text_input("API key", type="password", key=f"key_{pid}", help=meta["key_help"],
                      placeholder="Paste your key (kept only for this session)")

        with st.expander("Embeddings (for code search)"):
            st.caption("Used to index the repo. Local needs no key and no quota.")
            embed_label = st.selectbox("Embedding source", list(EMBED_CHOICES.keys()),
                                       index=1, key="embed_label")
            embed_provider = EMBED_CHOICES[embed_label]
            if embed_provider in ("gemini", "openai"):
                st.text_input(f"{embed_label} embedding key (optional if same as above)",
                              type="password", key=f"embedkey_{embed_provider}")

        st.divider()
        st.markdown("### GitHub (optional)")
        st.caption("Needed only to open a pull request, under your own account.")
        st.text_input("GitHub token", type="password", key="gh_token",
                      help="Fine-grained or classic token with repo/public_repo scope.",
                      placeholder="ghp_... (kept only for this session)")

        st.divider()
        st.markdown("### Run settings")
        confidence_threshold = st.slider("Auto-PR confidence threshold", 0.0, 1.0, 0.7, 0.05,
                                          help="Fixes below this confidence are flagged for review.")
        max_retries = st.number_input("Max fix attempts", 1, 5, 3)
        max_rag_results = st.number_input("Code chunks to retrieve", 1, 15, 5)

        st.caption("Your keys live only in this browser session. They are not stored on the "
                   "server or written to logs.")

    return {
        "provider_label": provider_label,
        "confidence_threshold": confidence_threshold,
        "max_retries": int(max_retries),
        "max_rag_results": int(max_rag_results),
    }


def _build_settings(provider_label: str) -> LLMSettings:
    """Construct validated LLMSettings from the sidebar. Raises ValidationError
    (shown to the user) if the key/base URL are missing or malformed."""
    meta = PROVIDERS[provider_label]
    pid = meta["id"]
    default_model = meta.get("default_model") or DEFAULT_CHAT_MODELS.get(pid, "")

    api_key = sanitize.clean_api_key(st.session_state.get(f"key_{pid}", ""))
    model = (st.session_state.get(f"model_{pid}") or default_model).strip()
    if not model:
        raise ValidationError("Please enter a model name.")

    base = None
    if meta["needs_base"]:
        base = sanitize.validate_base_url(st.session_state.get(f"base_{pid}", ""))

    embed_provider = EMBED_CHOICES[st.session_state.get("embed_label", "Local, no key needed (recommended)")]
    embed_key = ""
    if embed_provider in ("gemini", "openai"):
        raw = st.session_state.get(f"embedkey_{embed_provider}", "")
        if raw.strip():
            embed_key = sanitize.clean_api_key(raw)

    return LLMSettings(
        chat_provider=pid, chat_model=model, chat_api_key=api_key, chat_base_url=base,
        embed_provider=embed_provider, embed_api_key=embed_key,
    )


def _apply_session_credentials(provider_label: str) -> bool:
    """Install this session's LLM + GitHub credentials into the request context
    so any code path (run or chat) uses them. Returns True if an LLM key is
    configured. Silent on missing key (the run handler reports it)."""
    set_github_token(st.session_state.get("gh_token", ""))
    try:
        settings = _build_settings(provider_label)
    except ValidationError:
        # Always clear, so a previous session's key can never linger on a
        # reused worker thread and be used by this one.
        clear_settings()
        return False
    set_settings(settings)
    return True


# --------------------------------------------------------------------------
# Result renderers (all escape or use native components: no raw user HTML)
# --------------------------------------------------------------------------
def _verdict(final: dict) -> None:
    status = final.get("status", "")
    reason = final.get("outcome_reason", "")
    label = {
        STATUS_PR_CREATED: "Pull request opened",
        STATUS_PR_COMMITTED_LOCAL: "Fix committed locally",
        STATUS_PR_READY: "Fix found and validated",
        STATUS_NEEDS_HUMAN_REVIEW: "Investigated, needs human review",
        STATUS_ERROR: "Run did not complete",
    }.get(status, "Run finished")
    kind = "ok" if status in _SUCCESS_STATUSES else ("warn" if status == STATUS_NEEDS_HUMAN_REVIEW else "err")
    # On error, show a generic reason to users (debug off); detail stays in logs.
    if status == STATUS_ERROR and not auth.dev_mode():
        reason = "Something went wrong while running the pipeline. Please try again."
    _status(kind, label, reason)


def _render_explainability(final: dict) -> None:
    issue = final.get("issue") or {}
    rc = final.get("root_cause") or {}
    trace = final.get("call_graph_trace") or {}
    targeted = (final.get("validation") or {}).get("targeted_result") or {}

    with st.container(border=True):
        st.markdown('<div class="af-card-h">Explainability report</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Severity", str(issue.get("severity", "-")).title())
        conf = rc.get("confidence")
        c2.metric("Confidence", f"{conf:.0%}" if isinstance(conf, (int, float)) else "-")
        c3.metric("Tests", f"{targeted.get('passed_count', 0)}/{targeted.get('total', 0)}" if targeted else "-")
        c4.metric("Fix attempts", final.get("attempt", 0))

        if isinstance(conf, (int, float)):
            st.progress(min(max(float(conf), 0.0), 1.0), text="Diagnosis confidence")

        investigated = trace.get("files_visited") or []
        if investigated:
            st.markdown("**Investigated files** (real execution path via call graph):")
            st.code("\n".join(str(f) for f in investigated), language="text")

        if rc.get("root_cause"):
            st.markdown("**Root cause**")
            st.info(rc["root_cause"])
            if rc.get("affected_function"):
                files = ", ".join(rc.get("affected_files", []))
                st.caption(f"Location: {files} :: {rc['affected_function']}()")

        if rc.get("reasoning"):
            st.markdown("**Reasoning**")
            for i, step in enumerate(rc["reasoning"], 1):
                st.markdown(f"{i}. {step}")

        with st.expander("Raw structured output (JSON)"):
            st.json({"issue": issue, "root_cause": rc}, expanded=False)


def _render_fix(final: dict) -> None:
    patch = final.get("patch") or {}
    diff = patch.get("diff")
    if not diff:
        return
    with st.container(border=True):
        st.markdown('<div class="af-card-h">Proposed fix</div>', unsafe_allow_html=True)
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
        return
    with st.container(border=True):
        st.markdown('<div class="af-card-h">Validation</div>', unsafe_allow_html=True)
        if validation.get("apply_error"):
            _status("err", "The diff did not apply cleanly")
            st.code(str(validation["apply_error"]), language="text")
            return
        if validation.get("overall_passed"):
            _status("ok", "All tests passed",
                    f"{targeted.get('passed_count', 0)}/{targeted.get('total', 0)}")
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
        return
    with st.container(border=True):
        st.markdown('<div class="af-card-h">Pipeline timeline</div>', unsafe_allow_html=True)
        for step in history:
            # st.markdown escapes HTML by default, so these node/summary strings
            # render safely without manual escaping (which would double-encode).
            st.markdown(f"**{step.get('node', '')}**: {step.get('summary', '')}")


# --------------------------------------------------------------------------
# Pull request
# --------------------------------------------------------------------------
def _open_pr_from_result(final: dict, draft: bool) -> dict:
    import re
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
            safe_url = sanitize.safe_external_url(created.get("url"))
            _status("ok", "Pull request opened")
            if safe_url:
                st.link_button("Open the pull request", safe_url)
            else:
                st.caption("The pull request was created.")
            if created.get("mode") == "fork":
                st.caption(f"Opened from your fork {sanitize.escape(str(created.get('fork_full_name')))} "
                           "(no push access to the upstream repo).")
            return

        if status not in _SUCCESS_STATUSES:
            st.caption("A pull request can be opened once AuraFix has a validated fix.")
            return
        if not repo_full:
            st.info("This run used a local/sample repo, so there is no GitHub repo to open a PR "
                    "against. Re-run with a GitHub URL to enable pull requests.")
            return
        if not st.session_state.get("gh_token", "").strip():
            st.warning("Add a GitHub token in the sidebar to open a pull request, then try again.")
            return

        st.write(f"Open the reviewed fix as a pull request on {sanitize.escape(str(repo_full))}. "
                 "AuraFix forks if you cannot push, otherwise pushes directly.")
        draft = st.checkbox("Open as draft (recommended)", value=True, key="pr_draft")
        if st.button("Open pull request", type="primary", key="pr_btn"):
            with st.spinner("Committing the fix and opening the pull request..."):
                res = _open_pr_from_result(final, draft)
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
    with st.container(border=True):
        st.markdown('<div class="af-card-h">Ask about this fix</div>', unsafe_allow_html=True)
        st.caption("Ask why this is the root cause, whether the fix is safe, or what else to check.")

        for m in st.session_state.get("chat", []):
            with st.chat_message(m["role"]):
                # st.markdown escapes raw HTML by default (unsafe_allow_html=False),
                # so this is injection-safe while still rendering code blocks.
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
# About
# --------------------------------------------------------------------------
ABOUT_MD = """\
AuraFix turns a plain-English bug report into a validated fix and a pull request,
autonomously, with a human in the loop at review time.

**The pipeline (orchestrated with LangGraph):**

1. Issue understanding: parses your report into structured fields.
2. Repository indexing (RAG): embeds the repo and retrieves the most relevant code.
3. Code navigation: builds a call graph to trace the real execution path.
4. Root-cause analysis: weighs the evidence into one diagnosis with a confidence score.
5. Patch generation: writes a minimal diff that fixes the root cause.
6. Validation: applies the patch and runs the tests, retrying with the error context, then escalates if it cannot fix it.
7. Pull request: commits on a branch and opens a PR (forking if you lack push access).

**Supported languages** (structural navigation and test validation):

| Language | Navigation | Test validation |
|----------|------------|-----------------|
| Python | ast (full fidelity) | pytest |
| Java | tree-sitter | Maven / Gradle |
| C / C++ | tree-sitter | CMake + CTest / make test |
| C# | tree-sitter | dotnet test |

Every fix is AI-generated. Review it before merging and keep PRs as drafts until you have checked them.
"""


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
def _render_credit() -> None:
    st.markdown(
        f'<div class="af-credit"><span class="by">Built by Vidit Gupta</span>'
        f'<a href="{INSTAGRAM_URL}" target="_blank" rel="noopener noreferrer" '
        f'aria-label="Instagram">{_IG_SVG}</a>'
        f'<a href="{LINKEDIN_URL}" target="_blank" rel="noopener noreferrer" '
        f'aria-label="LinkedIn">{_LI_SVG}</a></div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    _render_credit()

    if _IMPORT_ERROR is not None:
        st.error("AuraFix could not start due to a configuration problem.")
        if os.environ.get("APP_ENV", "development").lower() != "production":
            st.code(str(_IMPORT_ERROR))
        st.stop()

    _require_auth()

    options = _render_sidebar()
    have_key = _apply_session_credentials(options["provider_label"])

    # Hero
    st.markdown(
        '<div class="af-kicker">Autonomous AI Agent</div>'
        '<div class="af-title">AuraFix</div>'
        '<p class="af-sub">Describe a bug in plain English. A pipeline of AI agents investigates the '
        'repository, pinpoints the root cause, writes a minimal fix, proves it with tests, and opens a '
        'pull request.</p>'
        '<div class="af-langs">Supported languages: Python, Java, C, C++, C#.</div>',
        unsafe_allow_html=True,
    )

    # How it works (numbered list, not three cards in a row)
    st.markdown(
        '<ol class="af-steps">'
        '<li><span class="n">1</span><span><span class="t">Point it at a repo.</span> '
        '<span class="d">The bundled sample or any public GitHub URL.</span></span></li>'
        '<li><span class="n">2</span><span><span class="t">Describe the bug.</span> '
        '<span class="d">Plain English. Add a stack trace for extra accuracy.</span></span></li>'
        '<li><span class="n">3</span><span><span class="t">Review and open a PR.</span> '
        '<span class="d">See the diagnosis and fix, then open a pull request.</span></span></li>'
        '</ol>',
        unsafe_allow_html=True,
    )
    st.write("")

    if not have_key:
        st.info("Add your model provider and API key in the sidebar to get started.")

    # Inputs
    sources = ["Bundled sample repo", "GitHub URL"]
    if sanitize.local_path_enabled():
        sources.insert(1, "Local path")

    with st.container(border=True):
        st.markdown('<div class="af-card-h">1. Choose a repository</div>', unsafe_allow_html=True)
        source = st.radio("Repository source", sources, horizontal=True, label_visibility="collapsed")
        repo_path_input, repo_url_input, sample = "", "", None
        if source == "Bundled sample repo":
            sample_name = st.selectbox("Sample", list(BUNDLED_SAMPLES.keys()), label_visibility="collapsed")
            sample = BUNDLED_SAMPLES[sample_name]
            st.caption(f"Using bundled {sample['path']}: {sample['blurb']} (no GitHub needed).")
        elif source == "Local path":
            repo_path_input = st.text_input("Local path", placeholder="/path/to/repo",
                                            label_visibility="collapsed")
        else:
            repo_url_input = st.text_input("GitHub URL", placeholder="https://github.com/owner/repo",
                                           label_visibility="collapsed")
            st.caption("Pulls the public code and lets you open a pull request from the result.")

    with st.container(border=True):
        st.markdown('<div class="af-card-h">2. Describe the bug</div>', unsafe_allow_html=True)
        default_bug = sample["bug"] if (source == "Bundled sample repo" and sample) else ""
        bug_report = st.text_area("Bug description", value=default_bug, height=90,
                                  label_visibility="collapsed",
                                  placeholder="e.g. Login page crashes when the password field is empty")
        with st.expander("Advanced (optional)"):
            stack_trace = st.text_area("Stack trace", height=110,
                                       help="File paths here seed the code navigation with exact entry points.")
            failing_test = st.text_input("Failing test name")
            branch_override = st.text_input("Base branch override", placeholder="main / master / develop")

    run = st.button("Run AuraFix", type="primary", use_container_width=True, disabled=not have_key)

    if run:
        _handle_run(source, sample, repo_path_input, repo_url_input, bug_report,
                    stack_trace, failing_test, branch_override, options)

    # Results
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

    st.markdown("---")
    with st.expander("About AuraFix"):
        st.markdown(ABOUT_MD)
    st.caption("AuraFix. Every fix is AI-generated; review before merging.")


def _handle_run(source, sample, repo_path_input, repo_url_input, bug_report,
                stack_trace, failing_test, branch_override, options) -> None:
    import re

    # Validate the bug report and seeds first.
    try:
        bug_report = sanitize.clamp(bug_report, sanitize.MAX_BUG_REPORT, "Bug description")
        if not bug_report:
            st.error("Please describe the bug first.")
            return
        stack_trace = sanitize.clamp(stack_trace, sanitize.MAX_STACK_TRACE, "Stack trace")
        failing_test = sanitize.clamp(failing_test, sanitize.MAX_SHORT_FIELD, "Failing test")
        branch_override = sanitize.clamp(branch_override, sanitize.MAX_SHORT_FIELD, "Branch")
    except ValidationError as e:
        st.error(str(e))
        return

    # Rate-limit the (expensive) run per identity before doing any work.
    decision = ratelimit.check_run(_identity())
    if not decision.allowed:
        wait = f" Try again in about {decision.retry_after}s." if decision.retry_after else ""
        st.warning(decision.reason + wait)
        return

    repo_path, repo_id, repo_full_name = None, None, None
    try:
        if source == "Bundled sample repo":
            repo_path, repo_id = sample["path"], sample["id"]
        elif source == "Local path":
            repo_path = sanitize.validate_local_path(repo_path_input)
            repo_id = repo_path
        else:
            target = sanitize.validate_github_url(repo_url_input)
            with st.spinner(f"Cloning {target.full_name}..."):
                cloned = clone_repo(target.url)
            if not cloned["success"]:
                _status("err", "Clone failed", str(cloned.get("error", "")))
                return
            repo_path, repo_id, repo_full_name = cloned["repo_path"], target.url, cloned["repo_full_name"]
    except ValidationError as e:
        st.error(str(e))
        return

    seeds = _files_from_text(stack_trace) + _files_from_text(failing_test)
    request = BugFixRequest(
        bug_report=bug_report, repo_path=repo_path, repo_id=repo_id,
        repo_full_name=repo_full_name, base_branch=branch_override or None,
        create_pr=False,
        seed_files=seeds or None, confidence_threshold=options["confidence_threshold"],
        max_retries=options["max_retries"], max_rag_results=options["max_rag_results"],
    )
    st.session_state.pop("pr_created", None)
    st.session_state["chat"] = []
    with st.spinner("Running the pipeline: issue, indexing, navigation, root cause, patch, validation..."):
        st.session_state["result"] = run_pipeline(request)


def _files_from_text(text: str) -> list:
    import re
    if not text:
        return []
    return list(dict.fromkeys(re.findall(r"[\w./\\-]+\.py", text)))


if __name__ == "__main__":
    main()
