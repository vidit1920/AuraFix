# AuraFix security review

This document records the hardening pass done on AuraFix and the residual risks
you should know about. It maps one-to-one to the checklist that was requested.

**Scope:** the hosted Streamlit app, its launcher, the LLM/RAG layer, and the
GitHub/pipeline code. **Deployment model:** Docker on Render, multi-user,
bring-your-own-key.

---

## Summary

| # | Area | Status | Where |
|---|------|--------|-------|
| 1 | Hide API keys | Done | `agents/runtime.py`, `agents/llm_client.py` |
| 2 | Environment variables | Done | `.env.example`, `.gitignore`, `.dockerignore` |
| 3 | Admin routes | N/A + mitigated | `security/auth.py` (`is_admin`), local-path gate |
| 4 | Authentication | Done | `security/auth.py`, `ui/firebase_auth/` |
| 5 | Authorization / least access | Done | per-session creds, gated local-path |
| 6 | Form sanitising / XSS | Done | `security/sanitize.py`, `ui/app.py` |
| 7 | Rate limiting | Done | `security/ratelimit.py` |
| 8 | Secure API endpoints | Done | CORS/XSRF, headers, no custom unauth routes |
| 9 | CORS | Done | `.streamlit/config.toml` |
| 10 | Security headers | Done | `security/headers.py`, `server.py` |
| 11 | Debug mode off | Done | `.streamlit/config.toml`, generic errors |
| 12 | Dependencies updated | Done | `requirements.txt` |
| 13 | Remove unneeded | Done | flake8 + empty `docker/` removed |
| 14 | Exposed files | Done | static serving off, ignore files |
| 15 | Secure database | Done (by design) | embedded ChromaDB, not networked |
| 16 | Password hashing | N/A | no app-managed passwords (Firebase) |
| 17 | Git history secrets | Clean | verified, see below |
| 18 | Full audit | This document | |

---

## 1. API keys are never exposed

- There is **no shared server LLM key**. Each user's key is read from the
  sidebar into a per-request `ContextVar` (`agents/runtime.py`) that is isolated
  per session/thread, used to make calls, and never written to disk, logs, or
  the page.
- Keys are **scrubbed from error messages** before display
  (`llm_client._scrub`, `providers._as_llm_error`), and provider error bodies
  are summarised rather than dumped.
- Keys are sent only in outbound HTTPS request headers to the provider the user
  chose. Key inputs reject whitespace/newlines to prevent header injection
  (`sanitize.clean_api_key`).
- GitHub tokens follow the same per-session model (`github_integration/credentials.py`).

## 2. Environment variables

- `.env` has never been committed (see #17) and is excluded by `.gitignore` and
  `.dockerignore`, along with service-account JSON, `*.pem`, `*.key`, and
  `secrets.toml`.
- `.env.example` documents every variable with guidance and **no real values**.
- Secrets are supplied to the container at runtime only; none are baked into the
  image.

## 3. Admin routes

A Streamlit app exposes no custom HTTP admin routes, so there is nothing of that
kind to leave unprotected. The two sensitive capabilities are handled directly:

- The **local-path repo source** (which would read the server's filesystem) is
  **disabled by default** (`AURAFIX_ALLOW_LOCAL_PATH`) and, when enabled, is
  confined to an allowed base directory (`sanitize.validate_local_path`).
- An **admin allowlist** (`AURAFIX_ADMIN_EMAILS`, `auth.is_admin`) is available
  to gate any admin-only feature you add later.

## 4. Authentication

- Sign-in uses **Firebase** (Google / GitHub). The browser obtains a signed ID
  token; the server **verifies it with the Firebase Admin SDK** using a service
  account (`auth.verify_token`, `check_revoked=True`). A token the client merely
  claims is never trusted.
- The app **fails closed**: if auth is enabled but the Firebase web config or
  service account is missing, every user is refused with a clear operator
  message (`auth.config_error`).
- A dev bypass exists only when `AURAFIX_AUTH_DISABLED=true` **and**
  `APP_ENV != production`, so it cannot accidentally open the live site.

## 5. Least-privilege access

- With BYOK, a user can only ever spend **their own** LLM quota and act under
  **their own** GitHub token; there is no shared credential to abuse.
- Per-session `ContextVar`s mean one user's credentials are never visible to
  another session.
- Server filesystem access via "local path" is off by default (#3).

## 6. Input sanitising and XSS

- All free-text inputs are length-clamped and validated
  (`security/sanitize.py`): the GitHub URL is accepted only as
  `https://github.com/owner/repo` (no other host/scheme, no SSH, no embedded
  credentials); custom base URLs must be http(s) and non-local must be https.
- The result renderers were rewritten to use **native Streamlit components**;
  the few remaining `unsafe_allow_html` blocks interpolate only static strings
  plus **HTML-escaped** values (`sanitize.escape`).
- Any URL rendered as a link is first checked to be an `https://github.com` URL
  (`sanitize.safe_external_url`); anything else is shown as plain text, so a
  surprising value can't become a `javascript:` link.
- LLM/chat output is escaped before rendering.

## 7. Rate limiting

- `security/ratelimit.py` enforces per-identity sliding windows on pipeline runs
  (default 5 / 10 min and 40 / day) and chat (40 / 10 min), plus a **global
  ceiling** (60 runs / 10 min) so a crowd can't exhaust the box. Identity is the
  Firebase uid, else the client IP.

## 8. Secure endpoints

- The only HTTP surface is Streamlit's own, now behind CORS + XSRF + CSP. No
  unauthenticated custom routes are added. The Firebase sign-in component is a
  static, same-origin asset.

## 9. CORS

- `server.enableCORS = true` and `server.enableXsrfProtection = true`. Set
  `STREAMLIT_SERVER_CORS_ALLOWED_ORIGINS=https://<your-app>.onrender.com` to pin
  the browser origin.

## 10. Security headers

`security/headers.py` (installed via the `server.py` ASGI launcher) adds, on
every response: a strict **Content-Security-Policy** (scripts limited to `self`,
one hashed Streamlit bootstrap, and the Firebase SDK origin - no `unsafe-eval`,
no `unsafe-inline` scripts), **HSTS**, **X-Frame-Options: SAMEORIGIN**,
**X-Content-Type-Options: nosniff**, **Referrer-Policy**, a restrictive
**Permissions-Policy**, and **COOP** tuned to allow the sign-in popup. The CSP
was verified against the running app with no console violations.

## 11. Debug mode off

- `client.showErrorDetails = "none"` and `fileWatcherType = "none"`; pipeline
  errors show a generic message in production (internals only in dev).
  `gatherUsageStats = false`.

## 12. Dependencies updated

- All direct dependencies were upgraded and pinned (`requirements.txt`).
  Notably **GitPython >= 3.1.60** (fixes the 2026 argument/config-injection and
  ReDoS CVEs), **urllib3 >= 2.8.0**, **PyJWT >= 2.15.1**, and an aligned
  OpenTelemetry set. `pip check` passes. The offline test suite passes (17/17),
  including the multi-language extractor that was updated for the new
  tree-sitter API.

## 13. Removed what isn't needed

- Dropped `flake8`/`pyflakes`/`pycodestyle` (never invoked) and `pypdf`
  (unused), and removed an empty `docker/` directory. Chat calls use `httpx`
  rather than pulling in OpenAI/Anthropic SDKs, keeping the surface small.

## 14. Exposed files

- `server.enableStaticServing = false`, so the app does not serve arbitrary
  files from its directory. `.gitignore`/`.dockerignore` keep secrets, the vector
  cache, and sample `.git` folders out of the repo and image.

## 15. Database

- There is no SQL/user database. RAG uses **ChromaDB strictly as an embedded,
  in-process library** against a local directory outside the web root - never
  the client/server mode that the 2026 ChromaDB CVEs (auth bypass / cross-tenant
  / `trust_remote_code` RCE) apply to. Collections are namespaced by repo +
  commit + embedding backend.

## 16. Password hashing

- The app stores **no passwords**. Identity is delegated to Firebase, so there
  is no password hashing for AuraFix to get wrong. (The passwords in
  `sample-repo/` are demo bug material, not real auth.) The only secret at rest
  is the Firebase service account, supplied as a runtime secret.

## 17. Git history

- The full history was scanned for the live key values and for key-shaped
  patterns (`AIza…`, `ghp_…`, `github_pat_…`, `sk-…`, `AQ.…`, PEM blocks).
  **No secret was ever committed**; only `.env.example` (placeholders) is
  tracked. No history rewrite is needed.
- Hygiene recommendation: since the key currently in your local `.env` has been
  used during development, consider rotating it once before go-live. Not
  required by any exposure found here.

---

## Residual risks and recommendations

- **Rate limiting is per-instance.** It is correct on a single Render instance.
  If you scale to multiple instances, move the limiter behind a shared store
  (e.g. Redis) using the same interface.
- **Test execution is hardened, not fully isolated.** A cloned repo's test suite
  runs as a non-root user under OS resource limits (CPU, memory, file size,
  process count) and wall-clock timeouts, but shares the container's network and
  kernel. For untrusted public repos at scale, run each job in a disposable
  sandbox (gVisor, Firecracker, or a per-run container) and disable outbound
  network for the test step.
- **Firebase must be configured** (web config + service account, Google/GitHub
  providers enabled, authorized domains) or the app will correctly refuse all
  sign-ins.
- **Pin your CORS origin** (`STREAMLIT_SERVER_CORS_ALLOWED_ORIGINS`) to your
  Render URL once the domain is known.
- **CSP and the Streamlit inline-script hash** are tied to the pinned Streamlit
  version; if you upgrade Streamlit and see a console CSP error, update the hash
  in `security/headers.py` with the value the browser prints.
