# AuraFix - Autonomous Bug Fixer

AuraFix takes a plain-English bug report and a repository, then runs a pipeline of
AI agents that **investigate the code, find the root cause, write a minimal fix,
prove it with tests, and open a pull request** - orchestrated with
[LangGraph](https://github.com/langchain-ai/langgraph).

It is built to run as a **hosted, multi-user app**:

- **Bring your own key (BYOK).** Each user pastes their own LLM key in the app.
  Gemini, OpenAI, Anthropic, or any OpenAI-compatible endpoint (Groq, OpenRouter,
  Together, a local server) are supported. There is no shared server key, so no
  shared quota or bill, and one user's key is never visible to another.
- **Sign in with Firebase** (Google / GitHub), verified on the server.
- **Opens pull requests under the user's own GitHub token.**

---

## How it works

| Stage | Module | What it does |
|-------|--------|--------------|
| Issue understanding | `agents/issue_agent.py` | Parses the bug report into structured JSON |
| Repo indexing (RAG) | `rag/` | Chunks + embeds the repo into ChromaDB, retrieves relevant code |
| Code navigation | `analyzers/` | Call-graph trace of the real execution path (Python `ast` + tree-sitter) |
| Root cause | `agents/root_cause_agent.py` | Combines the evidence into a diagnosis + confidence |
| Patch / fix | `agents/patch_agent.py` | Generates a minimal unified diff (retries with failure context) |
| Test validation | `validators/test_validator.py` | Applies the diff and runs the tests, sandboxed |
| PR creation | `github_integration/pr_agent.py` | Forks/branches/commits/pushes and opens the PR |
| Orchestrator | `graph/` | LangGraph state machine (retry + confidence gating) |
| Dashboard | `ui/app.py` | Streamlit UI + explainability report |
| LLM layer | `agents/runtime.py`, `agents/providers.py` | Per-session provider + key routing (BYOK) |
| Security | `security/` | Auth, rate limiting, input sanitisation, response headers |

---

## Run it locally

Requires **Python 3.12+** and **git**.

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
copy .env.example .env           # then edit .env
```

For local development you can skip Firebase by leaving `AURAFIX_AUTH_DISABLED=true`
and `APP_ENV=development` in `.env`, and set one LLM key (e.g. `GEMINI_API_KEY`)
so the CLI/tests work. Then:

```bash
# Dev (no security headers, hot reload):
streamlit run ui/app.py

# Production-style (adds security headers via the ASGI launcher):
python server.py
```

Open <http://localhost:8501>.

---

## Enable real sign-in (Firebase)

1. Create a project at <https://console.firebase.google.com>.
2. In **Authentication > Sign-in method**, enable **Google** and **GitHub**
   (GitHub needs an OAuth app; paste its client id/secret into Firebase).
3. Add your app's domain under **Authentication > Settings > Authorized domains**.
4. From **Project settings > General**, copy the web config into these env vars:
   `FIREBASE_API_KEY`, `FIREBASE_AUTH_DOMAIN`, `FIREBASE_PROJECT_ID`,
   `FIREBASE_APP_ID` (and optionally the sender id / storage bucket).
5. From **Project settings > Service accounts**, generate a private key and set
   `FIREBASE_SERVICE_ACCOUNT` to the JSON (single line) - this is the server
   secret used to verify tokens. **Never commit it.**
6. Set `AURAFIX_AUTH_DISABLED=false` (or remove it) and `APP_ENV=production`.

See `.env.example` for the full list.

---

## Deploy on Render (Docker)

The repo ships a `Dockerfile` that runs as a non-root user and starts
`server.py` (which binds Render's `$PORT` and adds the security headers).

1. New **Web Service** from this repo, environment **Docker**.
2. Add environment variables (from `.env.example`): `APP_ENV=production`, the
   `FIREBASE_*` config, `FIREBASE_SERVICE_ACCOUNT` (as a secret), and optionally
   `STREAMLIT_SERVER_CORS_ALLOWED_ORIGINS=https://<your-app>.onrender.com`.
3. Do **not** set a shared LLM key - users bring their own in the app.

Secrets are provided at runtime only; nothing sensitive is baked into the image
(`.dockerignore` excludes `.env`, service-account JSON, and key material).

---

## Security

See [`SECURITY.md`](SECURITY.md) for the full hardening write-up. In short:

- API keys and GitHub tokens live only in the request's in-memory context, never
  stored or logged, and are scrubbed from any error text.
- Firebase ID tokens are verified server-side; the app fails closed if auth is
  enabled but unconfigured.
- Strict Content-Security-Policy and the usual hardening headers (HSTS,
  X-Frame-Options, nosniff, Referrer-Policy, Permissions-Policy).
- CORS and XSRF protection on; static file serving off; debug/error details off.
- Per-user and global rate limiting on runs and chat.
- All inputs validated (GitHub URL, base URL, keys, local path); all rendered
  output escaped; the filesystem "local path" source is disabled by default.
- Untrusted cloned-repo test code runs as a non-root user under OS resource
  limits and wall-clock timeouts.

---

## Tests

Deterministic, offline (no API calls - real `git apply` + real `pytest`):

```bash
python -m pytest tests/ -v
```

---

## Supported languages

| Language | Code navigation | Test validation |
|----------|-----------------|-----------------|
| **Python** | `ast` (full fidelity) | pytest |
| **Java** | tree-sitter | Maven / Gradle |
| **C / C++** | tree-sitter | CMake + CTest / `make test` |
| **C#** | tree-sitter | `dotnet test` |

Running a non-Python language's tests requires that language's toolchain on the
host. The LLM can read and write any language; these five have structural
navigation + automated validation wired in.

---

*Built as a portfolio project by Vidit Gupta. Every fix you see in a PR is
generated and validated automatically - review before merging.*
