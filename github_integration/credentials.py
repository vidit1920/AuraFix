"""
credentials.py — the per-request GitHub token.

Like the LLM key (see agents.runtime), the GitHub token must not be a single
shared server-wide secret in a multi-user app: a pull request should be opened
under *the signed-in user's own* token, and one user's token must never be
visible to another. So the active token lives in a ``ContextVar`` set at the
top of each Streamlit run, and the clone / PR code reads it through here,
falling back to the ``GITHUB_TOKEN`` environment variable for local CLI use.
"""

from __future__ import annotations

import contextvars
import os

_PLACEHOLDER = "your-github-token-here"
_current: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "aurafix_github_token", default=None
)


def set_github_token(token: str | None) -> None:
    """Bind this request's token. An empty value is stored as "" (not None):
    it means "this web session has no token", which must NOT fall back to the
    server's GITHUB_TOKEN, or every visitor would clone, read and push with
    the operator's own credentials (e.g. their private repositories)."""
    _current.set((token or "").strip())


def get_github_token() -> str | None:
    """The active token. In a web session: only the user's own token. With no
    session bound at all (CLI / local scripts): the GITHUB_TOKEN env var."""
    token = _current.get()
    if token is not None:
        return token or None
    env = (os.environ.get("GITHUB_TOKEN") or "").strip()
    if env and env != _PLACEHOLDER:
        return env
    return None


# Environment for every git subprocess: never prompt. A headless server can't
# answer a prompt (it fails with "could not read Password"), and on a desktop
# Git Credential Manager would pop up a login window and stall the run.
NONINTERACTIVE_GIT_ENV = {"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never", "GIT_ASKPASS": ""}


def git_env() -> dict:
    return {**os.environ, **NONINTERACTIVE_GIT_ENV}


def authenticated_github_url(repo_full_name: str, token: str) -> str:
    """https URL with the token as the *password* of the x-access-token user.

    The old form, https://TOKEN@github.com/..., puts the token in the username
    slot, so git still asks for a password and fails for anything that needs
    auth (private clones, pushes). This form works for classic and
    fine-grained personal access tokens and for GitHub App tokens."""
    return f"https://x-access-token:{token}@github.com/{repo_full_name}.git"
