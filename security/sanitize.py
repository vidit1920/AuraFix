"""
sanitize.py — validate and normalise every value that enters the app from a
user or an external service, before it reaches a subprocess, an HTTP header,
the filesystem, or the page.

Nothing here trusts its input. The functions return a ``(ok, value, error)``
style result or raise ``ValidationError`` so callers can show a friendly
message instead of leaking a stack trace or, worse, acting on malicious input
(a ``javascript:`` link rendered as a URL, a newline smuggled into an API-key
HTTP header, a path traversal into the server filesystem, a multi-megabyte
"bug report" used to run up someone's token bill).
"""

from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass
from urllib.parse import urlparse

# Length caps — generous for real use, small enough to blunt abuse.
MAX_BUG_REPORT = 8000
MAX_STACK_TRACE = 20000
MAX_SHORT_FIELD = 300
MAX_API_KEY = 400
MAX_URL = 2000

# GitHub's own identifier rules (approximately): owner is alphanumeric with
# single hyphens; repo allows word chars, dot and hyphen.
_OWNER = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})"
_REPO = r"[A-Za-z0-9_.-]{1,100}"
_GITHUB_URL_RE = re.compile(rf"^https://github\.com/({_OWNER})/({_REPO}?)$")
_FULL_NAME_RE = re.compile(rf"^({_OWNER})/({_REPO})$")

# Control characters that must never appear in a header value, URL, or key.
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")


class ValidationError(ValueError):
    """A user-facing validation failure (message is safe to display)."""


@dataclass(frozen=True)
class RepoTarget:
    url: str              # normalised https URL, no credentials, no .git
    full_name: str        # "owner/repo"


def escape(text: str) -> str:
    """HTML-escape a string for safe interpolation into markup."""
    return html.escape(text or "", quote=True)


def clamp(text: str, max_len: int, field: str = "input") -> str:
    """Trim and length-check a free-text field."""
    text = (text or "").strip()
    if len(text) > max_len:
        raise ValidationError(f"{field} is too long (max {max_len} characters).")
    return text


# --------------------------------------------------------------------------
# GitHub
# --------------------------------------------------------------------------
def validate_github_url(raw: str) -> RepoTarget:
    """Accept only ``https://github.com/owner/repo`` (an optional ``.git`` or
    trailing slash is tolerated and stripped). Anything else — another host,
    an ``ssh://`` or ``git@`` form, a URL with embedded credentials or a query
    string, a non-GitHub link — is rejected. This is the single gate for the
    "GitHub URL" input, and it runs before any clone."""
    raw = clamp(raw, MAX_URL, "GitHub URL")
    if not raw:
        raise ValidationError("Please paste a GitHub URL.")
    if _CTRL_RE.search(raw):
        raise ValidationError("That URL contains invalid characters.")

    cleaned = raw
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]
    cleaned = cleaned.rstrip("/")

    m = _GITHUB_URL_RE.match(cleaned)
    if not m:
        raise ValidationError(
            "Enter a public GitHub repository URL like "
            "https://github.com/owner/repo (other hosts and SSH URLs aren't accepted)."
        )
    owner, repo = m.group(1), m.group(2)
    if not repo:
        raise ValidationError("That GitHub URL is missing the repository name.")
    return RepoTarget(url=f"https://github.com/{owner}/{repo}", full_name=f"{owner}/{repo}")


def validate_full_name(raw: str) -> str:
    """Validate an ``owner/repo`` string (used where a full name is passed
    around rather than a URL)."""
    raw = clamp(raw, MAX_SHORT_FIELD, "repository name")
    m = _FULL_NAME_RE.match(raw)
    if not m:
        raise ValidationError("Expected a repository in 'owner/repo' form.")
    return raw


def safe_external_url(url: str | None) -> str | None:
    """Return ``url`` only if it is an https link to github.com (or a
    subdomain) — used before rendering a PR link the pipeline produced. Any
    other scheme/host returns None, so a surprising value is shown as text,
    never as a clickable or script-bearing link."""
    if not url:
        return None
    if _CTRL_RE.search(url) or len(url) > MAX_URL:
        return None
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if parsed.scheme != "https":
        return None
    host = (parsed.hostname or "").lower()
    if host == "github.com" or host.endswith(".github.com"):
        return url
    return None


# --------------------------------------------------------------------------
# Credentials (BYOK)
# --------------------------------------------------------------------------
def clean_api_key(raw: str) -> str:
    """Strip surrounding whitespace and reject control characters / internal
    whitespace. Keys are sent in HTTP headers, so a stray newline would be a
    header-injection vector — reject it rather than trusting the HTTP client
    to. Does not validate the key against the provider (that happens on first
    use)."""
    key = (raw or "").strip()
    if not key:
        raise ValidationError("Please paste an API key.")
    if len(key) > MAX_API_KEY:
        raise ValidationError("That API key looks too long.")
    if _CTRL_RE.search(key) or any(c.isspace() for c in key):
        raise ValidationError("That API key contains invalid characters (no spaces or line breaks).")
    return key


def validate_base_url(raw: str) -> str:
    """Validate a custom OpenAI-compatible base URL."""
    url = clamp(raw, MAX_URL, "Base URL")
    if _CTRL_RE.search(url):
        raise ValidationError("That Base URL contains invalid characters.")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValidationError("Base URL must start with http:// or https://")
    if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1"):
        raise ValidationError("Use an https:// Base URL (http is only allowed for localhost).")
    return url.rstrip("/")


# --------------------------------------------------------------------------
# Local filesystem path (gated feature)
# --------------------------------------------------------------------------
def local_path_enabled() -> bool:
    """The 'Local path' repo source lets the app read an arbitrary directory
    on the server, so it is OFF unless explicitly enabled (and intended only
    for running the app on your own machine)."""
    return os.environ.get("AURAFIX_ALLOW_LOCAL_PATH", "").lower() in ("1", "true", "yes")


def validate_local_path(raw: str) -> str:
    """Resolve and confine a local path. If ``AURAFIX_LOCAL_PATH_BASE`` is set,
    the path must live inside it (prevents traversal to e.g. /etc)."""
    if not local_path_enabled():
        raise ValidationError("The local-path source is disabled on this deployment.")
    path = clamp(raw, MAX_URL, "Local path")
    resolved = os.path.realpath(path)
    if not os.path.isdir(resolved):
        raise ValidationError("That local path doesn't exist or isn't a directory.")
    base = os.environ.get("AURAFIX_LOCAL_PATH_BASE")
    if base:
        base_real = os.path.realpath(base)
        if os.path.commonpath([resolved, base_real]) != base_real:
            raise ValidationError("That path is outside the allowed directory.")
    return resolved
