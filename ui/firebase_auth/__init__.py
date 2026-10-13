"""
A small bidirectional Streamlit component that runs Firebase sign-in in an
iframe and returns the resulting ID token to Python.

It is a static (no build step) component: ``declare_component(path=...)``
serves the ``frontend/`` directory, and the frontend talks to Streamlit over
the documented ``postMessage`` protocol. The frontend loads the Firebase Web
SDK from gstatic (allowed by the app's CSP) and, after a successful popup
sign-in, posts ``{idToken, profile, nonce}`` back here.

The token returned by this component is *unverified* — it must be checked with
``security.auth.verify_token`` before the user is considered signed in.
"""

from __future__ import annotations

import os

import streamlit.components.v1 as components

_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
_component = components.declare_component("aurafix_firebase_auth", path=_FRONTEND_DIR)


def firebase_login(config: dict, providers: list[str], persist: bool = False,
                   signout: bool = False, key: str = "firebase_login") -> dict | None:
    """Render the sign-in buttons and manage persistence. Returns a dict the
    first time something happens, else None.

    Args:
        persist: if True, a successful sign-in uses Firebase local persistence
            so the user stays signed in across browser restarts (the server
            caps this at 30 days). If False, the session ends with the tab.
        signout: if True, the component signs the user out of Firebase
            (clearing persistence) instead of restoring/showing buttons.

    Returned dict shapes:
        {"idToken": str, "profile": {...}, "nonce": str, "restored": bool}
            — a new or restored sign-in to verify.
        {"signedOut": True, "nonce": str} — a completed sign-out.
    The ``nonce`` lets the caller ignore a value it has already consumed."""
    return _component(config=config, providers=providers, persist=bool(persist),
                      signout=bool(signout), key=key, default=None)
