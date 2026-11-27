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
| Repo indexing | `rag/vector_store.py` | Chunks + embeds the repo into ChromaDB (SFR-Embedding-Code-400M when installed) |
| Multi-agent RAG | `rag/agentic.py` | Query Planner (LLM) -> semantic, keyword (BM25) and call-graph retrievers -> rank fusion -> Relevance Critic (LLM), looping for follow-up searches |
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

### Optional: SFR code embeddings

[Salesforce/SFR-Embedding-Code-400M_R](https://huggingface.co/Salesforce/SFR-Embedding-Code-400M_R)
is a code-retrieval model that finds relevant code noticeably better than the
default MiniLM. It runs locally with no key, but needs about 2 GB of RAM and
downloads ~870 MB on first use, so it is not part of `requirements.txt` (it
does not fit Render's free 512 MB instance):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-sfr.txt
```

When installed it becomes the default on the Setup page; otherwise the app uses
MiniLM. The weights are licensed **CC-BY-NC-4.0** (research / non-commercial
use only). Its custom model code needs transformers 4.x (pinned in
`requirements-sfr.txt`).

### Retrieval modes

- **Multi-agent RAG** (default): a planner agent writes several searches, three
  retriever agents run them (semantic, keyword, call graph), and a critic agent
  keeps what matters and can request one more round. Costs 2 to 3 extra LLM calls.
- **Fast**: one round with a rule-based plan and no extra LLM calls. Useful on
  free-tier keys with tight daily quotas.

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

A [`render.yaml`](render.yaml) blueprint is included that declares the service,
the Docker build, the health check, and every required env var (secrets marked
`sync: false`, so no real values are committed).

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

### Benchmark (every supported language)

`benchmarks/cases/` holds small repositories with one real bug each, tests that
fail because of it, and a reference fix: 4 Python, 2 Java, 2 C#, 2 C and 2 C++,
plus the three bundled samples.

```bash
# No LLM: every buggy repo must fail its tests and every reference fix must pass,
# through AuraFix's own test runner (needs Maven, .NET 8 SDK, CMake + Ninja/make).
python -m benchmarks.run_benchmark --offline

# Live: run the full pipeline on every repo and score it (status, right file and
# function, and an independent re-run of the tests on the patched code).
python -m benchmarks.run_benchmark --mode multi_agent --embed sfr
python -m benchmarks.run_benchmark --only java-,cs- --resume   # subsets / resume
```

Scorecards land in `benchmarks/results/` (`*-latest.md`). If a model hits its
quota the runner moves to the next one in `--models`; `--resume` skips cases
already solved, so a free-tier key can finish across sessions. Toolchains
installed outside PATH can be listed in `AURAFIX_TOOL_PATHS`.

### Continuous checks (GitHub Actions)

`.github/workflows/ci.yml` runs on every push and pull request: the unit tests,
and the offline benchmark with Java, .NET 8 and C/C++ toolchains installed, so a
broken test runner for any language fails the build. A weekly (and on-demand)
job runs the live benchmark when the `GEMINI_API_KEY` repository secret is set
and uploads the scorecard.

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
