"""
auth.py — Firebase Authentication, verified on the server.

Flow:
  1. The browser signs in with Firebase (Google / GitHub / email) inside the
     sign-in component (ui/firebase_auth). Firebase hands the browser a signed
     ID token (a JWT).
  2. The component passes that token back to Python.
  3. :func:`verify_token` verifies the token's signature, issuer, audience and
     expiry with the Firebase Admin SDK — using a **service account**, which
     is the server-side secret. A token the client merely *claims* is valid is
     never trusted; only a token that passes this check authenticates a user.

What is secret and what is not:
  * The Firebase *web config* (apiKey, authDomain, projectId, appId, …) is
    public by design — it ships to every browser — so it lives in ordinary
    env vars and is safe to send to the component.
  * The *service account* (used here for verification) is a real secret and
    must be provided as an env var / mounted secret file, never committed.

Fail-closed: if auth is enabled but the server isn't configured, the app
refuses everyone rather than silently letting people in.
"""

from __future__ import annotations

import json
import os
import threading

_init_lock = threading.Lock()
_admin_ready = False
_admin_error: str | None = None


class AuthError(RuntimeError):
    """A user-facing authentication failure (safe to display)."""


def auth_enabled() -> bool:
    """Auth is on unless explicitly disabled for local development. Disabling
    it only takes effect when the app is NOT behind a public URL — see
    :func:`_dev_bypass_allowed`."""
    return os.environ.get("AURAFIX_AUTH_DISABLED", "").lower() not in ("1", "true", "yes")


def _dev_bypass_allowed() -> bool:
    """A development convenience: skip login when AURAFIX_AUTH_DISABLED is set
    AND the app is running in an explicitly non-production environment. This
    never trips on Render, where APP_ENV should be 'production'."""
    disabled = os.environ.get("AURAFIX_AUTH_DISABLED", "").lower() in ("1", "true", "yes")
    env = os.environ.get("APP_ENV", "development").lower()
    return disabled and env != "production"


def dev_mode() -> bool:
    return _dev_bypass_allowed()


def client_config() -> dict:
    """The public Firebase web config for the sign-in component, from env."""
    cfg = {
        "apiKey": os.environ.get("FIREBASE_API_KEY", ""),
        "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN", ""),
        "projectId": os.environ.get("FIREBASE_PROJECT_ID", ""),
        "appId": os.environ.get("FIREBASE_APP_ID", ""),
        "messagingSenderId": os.environ.get("FIREBASE_MESSAGING_SENDER_ID", ""),
        "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET", ""),
    }
    return cfg


def client_config_present() -> bool:
    cfg = client_config()
    return bool(cfg["apiKey"] and cfg["authDomain"] and cfg["projectId"] and cfg["appId"])


def enabled_providers() -> list[str]:
    """Which social providers to show buttons for (default google + github)."""
    raw = os.environ.get("FIREBASE_PROVIDERS", "google,github")
    allowed = {"google", "github", "email"}
    return [p.strip() for p in raw.split(",") if p.strip() in allowed] or ["google"]


def _ensure_admin() -> None:
    """Lazily initialise the Firebase Admin SDK from a service account. Safe to
    call repeatedly; the heavy init runs once."""
    global _admin_ready, _admin_error
    if _admin_ready or _admin_error:
        return
    with _init_lock:
        if _admin_ready or _admin_error:
            return
        try:
            import firebase_admin
            from firebase_admin import credentials

            raw = os.environ.get("FIREBASE_SERVICE_ACCOUNT", "").strip()
            key_file = os.environ.get("FIREBASE_SERVICE_ACCOUNT_FILE", "").strip()

            if raw:
                cred = credentials.Certificate(json.loads(raw))
            elif key_file and os.path.isfile(key_file):
                cred = credentials.Certificate(key_file)
            elif os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                cred = credentials.ApplicationDefault()
            else:
                raise AuthError(
                    "Firebase is not configured on the server. Set FIREBASE_SERVICE_ACCOUNT "
                    "(the service-account JSON) to enable sign-in."
                )

            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
            _admin_ready = True
        except AuthError as e:
            _admin_error = str(e)
        except Exception as e:  # noqa: BLE001
            _admin_error = f"Firebase Admin initialisation failed: {type(e).__name__}."


def config_error() -> str | None:
    """Returns a setup error if auth is required but the server can't verify
    tokens, else None. Used to show operators a clear message and to fail
    closed for users."""
    if not auth_enabled() or _dev_bypass_allowed():
        return None
    if not client_config_present():
        return ("Sign-in is unavailable: the Firebase web config is missing. Set the "
                "FIREBASE_API_KEY / FIREBASE_AUTH_DOMAIN / FIREBASE_PROJECT_ID / FIREBASE_APP_ID "
                "environment variables.")
    _ensure_admin()
    return _admin_error


def verify_token(id_token: str) -> dict:
    """Verify a Firebase ID token and return its claims. Raises AuthError on
    any problem. The returned dict is trusted; the input token is not."""
    if not id_token or not isinstance(id_token, str) or len(id_token) > 8000:
        raise AuthError("Invalid sign-in token.")
    _ensure_admin()
    if _admin_error:
        raise AuthError(_admin_error)

    from firebase_admin import auth as fb_auth

    try:
        # check_revoked=True rejects tokens from a disabled/deleted account.
        claims = fb_auth.verify_id_token(id_token, check_revoked=True)
    except Exception:  # noqa: BLE001 — never leak vendor internals to the user
        raise AuthError("Your sign-in could not be verified. Please sign in again.")

    # Cap how long a persisted ("keep me signed in") session may last. auth_time
    # is when the user actually authenticated; it does not change when Firebase
    # silently refreshes the token, so this enforces a hard re-login after the
    # configured number of days.
    import time as _time
    auth_time = claims.get("auth_time")
    if auth_time:
        max_days = session_max_days()
        if _time.time() - float(auth_time) > max_days * 86400:
            raise AuthError(f"Your {max_days}-day session has expired. Please sign in again.")

    return {
        "uid": claims.get("uid") or claims.get("user_id") or "",
        "email": claims.get("email") or "",
        "email_verified": bool(claims.get("email_verified")),
        "name": claims.get("name") or "",
        "picture": claims.get("picture") or "",
        "provider": (claims.get("firebase", {}) or {}).get("sign_in_provider", ""),
        "auth_time": claims.get("auth_time"),
        "exp": claims.get("exp"),
    }


def session_max_days() -> int:
    """How long a 'keep me signed in' session may last before the user must
    re-authenticate. Default 30 days; override with AURAFIX_SESSION_MAX_DAYS."""
    try:
        return max(1, int(os.environ.get("AURAFIX_SESSION_MAX_DAYS", "30")))
    except ValueError:
        return 30


def admin_emails() -> set[str]:
    raw = os.environ.get("AURAFIX_ADMIN_EMAILS", "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def is_admin(email: str | None) -> bool:
    return bool(email) and email.lower() in admin_emails()
