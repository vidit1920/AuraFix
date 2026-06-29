# 🛠️ AuraFix — Autonomous Bug Fixer

AuraFix takes a plain-English bug report and a repository, then runs a pipeline of
AI agents that **investigate the code, find the root cause, write a minimal fix,
prove it with tests, and open a pull request** — all autonomously, orchestrated
with [LangGraph](https://github.com/langchain-ai/langgraph).

It's built to be usable for real open-source contribution: if you don't have push
access to a repo (the usual case), AuraFix **forks it, pushes to your fork, and
opens a cross-fork PR** for you.

---

## How it works

```
Bug report ─▶ Issue Agent ─▶ Repo Indexing (RAG) ─▶ Code Navigation (call graph)
                                                              │
                                                              ▼
   PR  ◀─ PR Agent ◀─ Test Validator ◀─ Patch/Fix Agent ◀─ Root Cause Agent
                          │  ▲                                     
                          │  └────────── retry (≤3) with failure context
                          ▼
                    human review  (low confidence / couldn't fix)
```

| Stage | Module | What it does |
|-------|--------|--------------|
| Issue understanding | `agents/issue_agent.py` | Parses the bug report into structured JSON |
| Repo indexing (RAG) | `rag/` | Chunks + embeds the repo into ChromaDB, retrieves relevant code |
| Code navigation | `analyzers/call_graph.py` | AST call-graph trace — the *real* execution path (ground truth) |
| Root cause | `agents/root_cause_agent.py` | Combines all evidence into a diagnosis + confidence |
| Patch / fix | `agents/patch_agent.py` | Generates a minimal unified diff (retries with failure context) |
| Test validation | `validators/test_validator.py` | Applies the diff and runs pytest |
| PR creation | `github_integration/pr_agent.py` | Forks/branches/commits/pushes and opens the PR |
| Orchestrator | `graph/` | LangGraph state machine tying it all together (retry + gating) |
| Dashboard | `ui/app.py` | Streamlit UI + explainability report |

---

## Setup (VS Code)

Requires **Python 3.11+** (developed on 3.13) and **git**.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your keys
copy .env.example .env         # Windows  (cp on macOS/Linux)
# then edit .env and set GEMINI_API_KEY (free: https://aistudio.google.com/apikey)
```

---

## Run it

**Dashboard (recommended):**
```bash
streamlit run ui/app.py
```
Opens at <http://localhost:8501>. The form is pre-filled with a bundled buggy
sample repo — just click **Run AuraFix**.

**Command-line demo:**
```bash
python -m graph.workflow
```

**Tests** (fast, deterministic, no API calls — real `git apply` + real pytest):
```bash
python -m pytest tests/ -v
```

---

## Opening real pull requests (open-source contribution)

1. Create a GitHub token with **`repo`** scope at <https://github.com/settings/tokens>
   and add it to `.env` as `GITHUB_TOKEN`.
2. In the dashboard, choose **GitHub URL**, paste the project you want to contribute
   to, describe the bug, tick **"Open a real GitHub PR"**, and run.
3. AuraFix forks the repo → applies + validates the fix → opens the PR from your fork.

> ⚠️ **Use responsibly.** Review every fix before it reaches a maintainer, keep PRs
> as drafts until you've checked them, and follow each project's `CONTRIBUTING`
> guidelines. Don't mass-open AI-generated PRs.

---

## Supported languages

| Language | Code navigation | Test validation |
|----------|-----------------|-----------------|
| **Python** | `ast` (full fidelity) | pytest |
| **Java** | tree-sitter | Maven / Gradle |
| **C / C++** | tree-sitter | CMake + CTest / `make test` |
| **C#** | tree-sitter | `dotnet test` |

The LLM can read and write any language; these are the ones with structural
**call-graph navigation** + **automated test validation** wired in. Running a
non-Python language's tests requires that language's toolchain (JDK + Maven/Gradle,
.NET SDK, CMake + a compiler) installed on the machine running AuraFix. Python is
the most battle-tested path; C/C++ test execution is best-effort since build
setups vary.

## Notes

- **Free-tier quota:** the Gemini free tier allows ~20 requests/day *per model*. A
  full run uses a handful. If you hit a `429`, switch models (the app defaults to
  `gemini-2.5-flash`; set `GEMINI_MODEL` in your shell to override) or wait for the
  daily reset. The offline tests never touch the API.
- **The sample bug** lives in `sample-repo/auth/auth_service.py` (`.strip()` on a
  `None` password). The bundled `sample-repo` is its own git repo so the
  apply/commit steps work out of the box.

---

*Built as a portfolio project. The fix you see in a PR is generated and validated
automatically — but a human should always review before merging.*
