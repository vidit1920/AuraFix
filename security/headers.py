"""
headers.py — a Starlette/ASGI middleware that adds security response headers
to every page Streamlit serves.

Streamlit has no built-in hook for response headers, so the app is launched
through an ``st.App`` ASGI wrapper (see server.py) with this middleware
installed. The headers here are the browser-enforced half of the app's
defence: a Content-Security-Policy that pins script/style/connect origins, a
clickjacking guard, MIME-sniffing and referrer controls, HSTS, and a
restrictive Permissions-Policy.

The CSP is tuned for two things that must keep working:
  1. Streamlit's own frontend — ES-module scripts from 'self', heavy inline
     styles, a same-origin WebSocket, blob/data images, and web workers.
  2. The Firebase sign-in iframe (ui/firebase_auth) — the Firebase JS SDK is
     loaded from www.gstatic.com and talks to Google's identity endpoints,
     and Firebase hosts a helper iframe on <project>.firebaseapp.com.

Because one CSP applies to every same-origin response, the Firebase origins
are allowed globally. That is a deliberate, scoped widening (Google identity
origins only), not 'unsafe' wildcards. Everything can be overridden from the
environment (CONTENT_SECURITY_POLICY) if a deployment needs to adjust it.
"""

from __future__ import annotations

import os

from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Streamlit's index.html contains exactly one tiny inline bootstrap script
# (`window.prerenderReady = false`). Rather than open script-src with
# 'unsafe-inline', we allow just that script by its content hash. It is stable
# for a pinned Streamlit version; if a Streamlit upgrade changes it, the
# browser console prints the new hash to drop in here.
_STREAMLIT_INLINE_HASH = "'sha256-THxz9J9+wf7rhwR2YkXJoSZ13UCmYyC+3lNDv2+8ugA='"

# Google / Firebase identity origins the sign-in iframe needs.
_FIREBASE_SCRIPT = "https://www.gstatic.com https://apis.google.com"
_FIREBASE_CONNECT = (
    "https://*.googleapis.com https://securetoken.googleapis.com "
    "https://identitytoolkit.googleapis.com https://*.firebaseio.com"
)
_FIREBASE_FRAME = "https://*.firebaseapp.com https://*.web.app https://accounts.google.com https://apis.google.com"


def _default_csp() -> str:
    # 'unsafe-inline' for styles is unavoidable with Streamlit (it injects
    # inline <style> everywhere, and our theme CSS is injected the same way).
    # Scripts are restricted to 'self' plus the Firebase SDK origin — note
    # there is no 'unsafe-eval' and no 'unsafe-inline' for scripts.
    directives = {
        "default-src": "'self'",
        "base-uri": "'self'",
        "object-src": "'none'",
        "script-src": f"'self' {_STREAMLIT_INLINE_HASH} {_FIREBASE_SCRIPT}",
        "style-src": "'self' 'unsafe-inline'",
        "img-src": "'self' data: blob: https://*.googleusercontent.com https://avatars.githubusercontent.com",
        "font-src": "'self' data:",
        # 'self' authorises same-origin ws:/wss: (Streamlit's live connection)
        # in modern browsers, plus the Firebase identity endpoints.
        "connect-src": f"'self' {_FIREBASE_CONNECT}",
        "worker-src": "'self' blob:",
        "frame-src": f"'self' {_FIREBASE_FRAME}",
        # The app may embed its own component iframes, so allow same-origin
        # framing — but nobody else may frame the app (anti-clickjacking).
        "frame-ancestors": "'self'",
        "form-action": "'self'",
    }
    return "; ".join(f"{k} {v}" for k, v in directives.items())


def _permissions_policy() -> str:
    # Deny powerful features the app never uses.
    denied = ["camera", "microphone", "geolocation", "payment", "usb",
              "magnetometer", "gyroscope", "accelerometer", "interest-cohort"]
    return ", ".join(f"{feature}=()" for feature in denied)


class SecurityHeadersMiddleware:
    """Pure-ASGI middleware (no Starlette BaseHTTPMiddleware) so it adds no
    buffering overhead to Streamlit's streaming responses."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._csp = os.environ.get("CONTENT_SECURITY_POLICY") or _default_csp()
        self._permissions = _permissions_policy()
        # HSTS is safe to send always: browsers ignore it over plain http, and
        # Render serves the app over https.
        self._hsts = "max-age=31536000; includeSubDomains"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                existing = {k.lower() for k, _ in headers}
                additions = {
                    b"content-security-policy": self._csp.encode("latin-1"),
                    b"x-content-type-options": b"nosniff",
                    b"x-frame-options": b"SAMEORIGIN",
                    b"referrer-policy": b"strict-origin-when-cross-origin",
                    b"permissions-policy": self._permissions.encode("latin-1"),
                    b"strict-transport-security": self._hsts.encode("latin-1"),
                    b"cross-origin-opener-policy": b"same-origin-allow-popups",
                    b"x-permitted-cross-domain-policies": b"none",
                }
                for name, value in additions.items():
                    if name.decode("latin-1") not in existing:
                        headers.append((name, value))
            await send(message)

        await self.app(scope, receive, send_with_headers)
