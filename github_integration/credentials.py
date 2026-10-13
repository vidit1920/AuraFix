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
    _current.set((token or "").strip() or None)


def get_github_token() -> str | None:
    """The active token: the per-session one if set, else the environment
    variable. Returns None if neither is a usable (non-placeholder) value."""
    token = _current.get()
    if token:
        return token
    env = (os.environ.get("GITHUB_TOKEN") or "").strip()
    if env and env != _PLACEHOLDER:
        return env
    return None
