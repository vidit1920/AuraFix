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


def firebase_login(config: dict, providers: list[str], key: str = "firebase_login") -> dict | None:
    """Render the sign-in buttons. Returns a dict with the (unverified) token
    the first time a sign-in completes, else None.

    Returned dict shape: {"idToken": str, "profile": {...}, "nonce": str}.
    The ``nonce`` lets the caller ignore a value it has already consumed."""
    return _component(config=config, providers=providers, key=key, default=None)
