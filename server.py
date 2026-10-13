"""
server.py — production entry point.

Streamlit's own ``streamlit run`` gives no hook for adding security response
headers. Streamlit 1.65 ships an ASGI wrapper (``st.App``) that does accept
custom middleware, so the app is launched through it with
``SecurityHeadersMiddleware`` installed. This is what the Dockerfile runs; it
binds the port Render provides via ``$PORT``.

Local development can still use plain ``streamlit run ui/app.py`` — this file
is only needed to get the security headers in front of the app.
"""

from __future__ import annotations

import os

import streamlit as st
from starlette.middleware import Middleware

from security.headers import SecurityHeadersMiddleware

# Resolve the UI script relative to this file so the working directory doesn't
# matter.
_APP_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "app.py")

app = st.App(_APP_SCRIPT, middleware=[Middleware(SecurityHeadersMiddleware)])


def main() -> None:
    port = int(os.environ.get("PORT", os.environ.get("STREAMLIT_SERVER_PORT", "8501")))
    app.run(
        config={
            "server.port": port,
            "server.address": "0.0.0.0",
            "server.headless": True,
            # CORS + XSRF stay on; see .streamlit/config.toml for the rest.
            "browser.gatherUsageStats": False,
        }
    )


if __name__ == "__main__":
    main()
