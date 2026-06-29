"""
pr_agent.py — Agent 7 in the pipeline.

Creates a branch, commits the applied patch, pushes it, and opens a
GitHub pull request with a templated description.

DESIGN NOTE — why this agent makes NO LLM calls:
By the time this agent runs, every fact it needs already exists in the
pipeline's state (Root Cause Agent's diagnosis, the diff, validation
results). This agent's only job is to format those facts into Markdown
and call GitHub's API — there's no reasoning left to do. Calling the
LLM here would just burn quota on a step that's pure templating, and
would introduce a chance of the description drifting from the actual
facts (e.g. hallucinating a claim not supported by the diagnosis).

GitPython handles the local git operations (branch, commit, push).
PyGithub handles the actual PR creation via GitHub's REST API — "PR"
is a GitHub-specific concept layered on top of plain git, so it always
needs an API call, not just git commands.

CONTRIBUTION FLOW (the part that makes this usable for real open-source
work, not just repos you own):
You almost never have write access to an upstream repo you want to
contribute to (e.g. some popular OSS project). Pushing a branch to it
directly will just fail with 403. The standard contributor workflow,
which open_pull_request() automates, is:

    1. fork the upstream repo into your own account
    2. push your fix branch to YOUR fork
    3. open a PR from `yourname:branch` -> `upstream:default_branch`

open_pull_request() auto-detects which mode applies: if you DO have push
access (your own repo, or you're a collaborator) it pushes directly and
opens a same-repo PR; otherwise it forks. Either way the result is a real
PR opened by AuraFix.
"""

import os
import time
from git import Repo, GitCommandError
from github import Github, GithubException


def _token() -> str | None:
    """Returns a usable GITHUB_TOKEN, or None if it's missing/placeholder."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token or token == "your-github-token-here":
        return None
    return token


def _authenticated_remote_url(repo_full_name: str, token: str) -> str:
    """Builds an https remote URL with the token embedded so `git push`
    authenticates non-interactively (a headless run can't answer a
    username/password prompt)."""
    return f"https://{token}@github.com/{repo_full_name}.git"


def create_branch_and_commit(
    repo_path: str,
    branch_name: str,
    commit_message: str,
    base_branch: str = "main",
) -> dict:
    """
    Creates a new branch off base_branch, stages all current changes
    (the already-applied patch), and commits them.

    Assumes the patch has ALREADY been applied to the working tree by
    the Patch Agent / Test Validator before this runs — this function
    does not apply any diff itself, it only handles branch + commit.

    Returns:
        {"success": bool, "error": str | None}
    """
    try:
        repo = Repo(repo_path)
    except Exception as e:
        return {"success": False, "error": f"Could not open repo at {repo_path}: {e}"}

    try:
        # Create and check out the new branch from the current HEAD.
        # Assumes the caller already has base_branch checked out —
        # this agent doesn't switch to base_branch first, since the
        # orchestrator should control which branch the patch was
        # applied against in the first place.
        new_branch = repo.create_head(branch_name)
        new_branch.checkout()
    except Exception as e:
        return {"success": False, "error": f"Could not create/checkout branch {branch_name}: {e}"}

    try:
        repo.git.add(A=True)
        if not repo.is_dirty(untracked_files=True) and not repo.index.diff("HEAD"):
            return {"success": False, "error": "No changes to commit — was the patch actually applied?"}
        repo.index.commit(commit_message)
    except GitCommandError as e:
        return {"success": False, "error": f"Commit failed: {e}"}

    return {"success": True, "error": None}


def push_branch(repo_path: str, branch_name: str, remote_name: str = "origin") -> dict:
    """
    Pushes branch_name to the remote. Requires the repo's remote to
    already be configured with credentials (e.g. via GITHUB_TOKEN
    embedded in the remote URL, or an SSH key) — this function does
    not handle authentication setup itself.

    Returns:
        {"success": bool, "error": str | None}
    """
    try:
        repo = Repo(repo_path)
        remote = repo.remote(remote_name)
        remote.push(refspec=f"{branch_name}:{branch_name}")
        return {"success": True, "error": None}
    except Exception as e:
        return {"success": False, "error": f"Push failed: {e}"}


def build_pr_description(
    root_cause: dict,
    diff: str,
    test_result: dict,
    lint_result: dict = None,
) -> str:
    """
    Templates the PR description from already-computed facts. Every
    section here is a direct read of upstream agent output — nothing
    is invented or re-reasoned about at this stage.

    Args:
        root_cause: output of agents.root_cause_agent.analyze_root_cause().
        diff: the unified diff string from agents.patch_agent.generate_patch().
        test_result: the "targeted_result" dict from
            validators.test_validator.validate_patch().
        lint_result: optional, output of a lint validator if one ran.

    Returns:
        Markdown-formatted PR description.
    """
    affected_files = ", ".join(f"`{f}`" for f in root_cause.get("affected_files", []))
    confidence = root_cause.get("confidence", 0)
    reasoning_lines = "\n".join(f"- {step}" for step in root_cause.get("reasoning", []))

    tests_passed = test_result.get("passed", False)
    test_summary = (
        f"✅ {test_result.get('passed_count', 0)}/{test_result.get('total', 0)} passed"
        if tests_passed
        else f"❌ {test_result.get('failed_count', 0)}/{test_result.get('total', 0)} failed"
    )

    lint_line = ""
    if lint_result is not None:
        lint_passed = lint_result.get("passed", False)
        lint_line = f"- Static analysis: {'✅ passed' if lint_passed else '❌ failed'}\n"

    description = f"""## Summary
This PR was generated automatically by AuraFix AI in response to a reported bug.

**Root cause:** {root_cause.get('root_cause', 'unknown')}
**Affected file(s):** {affected_files or 'unknown'}
**Confidence:** {confidence}

## Reasoning
{reasoning_lines or '(no reasoning provided)'}

## Validation
{lint_line}- Tests: {test_summary}

## Diff
```diff
{diff}
```

---
*Generated automatically by AuraFix AI*
"""
    return description


def create_pull_request(
    repo_full_name: str,
    branch_name: str,
    title: str,
    description: str,
    base: str = "main",
) -> dict:
    """
    Opens a pull request via GitHub's API.

    Args:
        repo_full_name: "owner/repo", e.g. "octocat/Hello-World".
        branch_name: the head branch (already pushed to the remote).
        title: PR title.
        description: PR body, typically from build_pr_description().
        base: the branch to merge into.

    Returns:
        {"success": bool, "url": str | None, "error": str | None}

    Requires GITHUB_TOKEN to be set in the environment.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token or token == "your-github-token-here":
        return {
            "success": False,
            "url": None,
            "error": "GITHUB_TOKEN is not set. Generate one at "
                     "https://github.com/settings/tokens and add it to .env",
        }

    try:
        gh = Github(token)
        repo = gh.get_repo(repo_full_name)
        pr = repo.create_pull(title=title, body=description, head=branch_name, base=base)
        return {"success": True, "url": pr.html_url, "error": None}
    except GithubException as e:
        return {"success": False, "url": None, "error": f"GitHub API error: {e.data.get('message', str(e))}"}
    except Exception as e:
        return {"success": False, "url": None, "error": str(e)}


def _ensure_fork(gh: "Github", upstream_full_name: str, poll_timeout: int = 60) -> dict:
    """
    Makes sure the authenticated user has a fork of upstream, creating one
    if needed. Forking is asynchronous on GitHub's side, so when we create
    a new fork we poll until it's queryable before returning.

    Returns:
        {"success": bool, "fork_full_name": str | None, "created": bool, "error": str | None}
    """
    try:
        me = gh.get_user()
        upstream = gh.get_repo(upstream_full_name)
        fork_full_name = f"{me.login}/{upstream.name}"

        # Already forked? Then there's nothing to wait for.
        try:
            gh.get_repo(fork_full_name)
            return {"success": True, "fork_full_name": fork_full_name, "created": False, "error": None}
        except GithubException:
            pass

        fork = upstream.create_fork()
        target_name = getattr(fork, "full_name", fork_full_name)

        deadline = time.time() + poll_timeout
        while time.time() < deadline:
            try:
                gh.get_repo(target_name)
                return {"success": True, "fork_full_name": target_name, "created": True, "error": None}
            except GithubException:
                time.sleep(2)

        # Fork was requested but isn't queryable yet — surface it rather
        # than blocking forever; the caller can decide to retry.
        return {"success": False, "fork_full_name": target_name, "created": True,
                "error": f"fork of {upstream_full_name} was not ready within {poll_timeout}s"}
    except GithubException as e:
        return {"success": False, "fork_full_name": None, "created": False,
                "error": f"GitHub API error while forking: {e.data.get('message', str(e))}"}


def _set_push_remote(repo_path: str, remote_name: str, repo_full_name: str, token: str) -> None:
    """(Re)points a git remote at repo_full_name with the token embedded,
    so the subsequent push authenticates. Replaces the remote if it already
    exists, so repeated runs don't accumulate stale remotes."""
    repo = Repo(repo_path)
    url = _authenticated_remote_url(repo_full_name, token)
    try:
        repo.delete_remote(remote_name)
    except (GitCommandError, Exception):
        pass
    repo.create_remote(remote_name, url)


def _has_push_access(repo) -> bool:
    """Whether the authenticated user can push to this repo. PyGithub
    populates `.permissions` when the repo is fetched as an authenticated
    user; guarded with getattr since it isn't always present."""
    perms = getattr(repo, "permissions", None)
    return bool(perms and getattr(perms, "push", False))


def open_pull_request(
    repo_path: str,
    upstream_full_name: str,
    branch_name: str,
    title: str,
    description: str,
    base_branch: str | None = None,
    draft: bool = False,
) -> dict:
    """
    Opens a REAL pull request the way an open-source contributor would.

    Assumes the fix is already committed on `branch_name` in the local repo
    at `repo_path` (create_branch_and_commit handles that). This function
    handles the push + PR, auto-selecting the right strategy:

      - direct mode  (you can push to upstream): push the branch to upstream
        and open a same-repo PR.
      - fork mode    (you cannot): fork upstream, push the branch to your
        fork, and open a cross-fork PR (`yourname:branch` -> upstream:base).

    Args:
        repo_path: local clone of the UPSTREAM repo with the fix committed.
        upstream_full_name: "owner/repo" of the project being contributed to.
        branch_name: the already-committed fix branch.
        title / description: PR title and body (from build_pr_* ).
        base_branch: target branch; defaults to upstream's default branch.
        draft: open the PR as a draft (recommended for AI-assisted fixes
            you want a human to review before marking ready).

    Returns:
        {
            "success": bool,
            "url": str | None,
            "mode": "direct" | "fork" | None,
            "fork_full_name": str | None,
            "branch": str,
            "error": str | None,
        }
    """
    base_result = {"success": False, "url": None, "mode": None,
                   "fork_full_name": None, "branch": branch_name, "error": None}

    token = _token()
    if not token:
        base_result["error"] = ("GITHUB_TOKEN is not set. Create a token with 'repo' scope at "
                                 "https://github.com/settings/tokens and add it to .env.")
        return base_result

    try:
        gh = Github(token)
        me = gh.get_user().login
        upstream = gh.get_repo(upstream_full_name)
        base = base_branch or upstream.default_branch
    except GithubException as e:
        base_result["error"] = f"GitHub API error: {e.data.get('message', str(e))}"
        return base_result
    except Exception as e:  # noqa: BLE001
        base_result["error"] = str(e)
        return base_result

    # Decide direct vs fork.
    if _has_push_access(upstream):
        mode = "direct"
        push_target = upstream_full_name
        head = branch_name
        fork_full_name = None
    else:
        mode = "fork"
        fork = _ensure_fork(gh, upstream_full_name)
        if not fork["success"]:
            base_result.update(mode=mode, error=fork["error"])
            return base_result
        fork_full_name = fork["fork_full_name"]
        push_target = fork_full_name
        # Cross-fork PRs identify the source branch as "forkowner:branch".
        head = f"{fork_full_name.split('/')[0]}:{branch_name}"

    # Push the committed branch to the chosen target (upstream or fork).
    try:
        _set_push_remote(repo_path, "aurafix-push", push_target, token)
        push = push_branch(repo_path, branch_name, remote_name="aurafix-push")
        if not push["success"]:
            # Scrub the token out of any error text before surfacing it.
            err = (push["error"] or "").replace(token, "***")
            base_result.update(mode=mode, fork_full_name=fork_full_name,
                               error=f"Push to {push_target} failed: {err}")
            return base_result
    except Exception as e:  # noqa: BLE001
        base_result.update(mode=mode, fork_full_name=fork_full_name,
                           error=f"Push setup failed: {str(e).replace(token, '***')}")
        return base_result

    # Open the PR on the UPSTREAM repo (cross-fork head in fork mode).
    try:
        pr = upstream.create_pull(
            title=title, body=description, head=head, base=base,
            draft=draft, maintainer_can_modify=True,
        )
        return {"success": True, "url": pr.html_url, "mode": mode,
                "fork_full_name": fork_full_name, "branch": branch_name, "error": None}
    except GithubException as e:
        msg = e.data.get("message", str(e)) if hasattr(e, "data") else str(e)
        # A very common, non-fatal case: a PR for this head already exists.
        base_result.update(mode=mode, fork_full_name=fork_full_name,
                           error=f"GitHub API error opening PR: {msg}")
        return base_result
    except Exception as e:  # noqa: BLE001
        base_result.update(mode=mode, fork_full_name=fork_full_name, error=str(e))
        return base_result


def build_pr_title(root_cause: dict) -> str:
    """
    Generates a concise PR title from the root cause diagnosis.
    Kept simple and template-based rather than LLM-generated, for the
    same reason as the rest of this agent — no new information needed.
    """
    function = root_cause.get("affected_function", "")
    files = root_cause.get("affected_files", [])
    file_hint = os.path.basename(files[0]) if files else "code"
    if function:
        return f"Fix bug in {function}() ({file_hint})"
    return f"Fix bug in {file_hint}"


if __name__ == "__main__":
    # Quick manual test — run with: python -m github.pr_agent
    # This builds a description and title from a mock diagnosis, but
    # does NOT actually create a branch/push/PR against any real repo
    # unless GITHUB_TOKEN and a real repo_full_name are supplied —
    # template-building is safe to test standalone, PR creation is not
    # (it would actually open a real PR against a real repo).
    mock_root_cause = {
        "root_cause": "password.strip() called without a None check",
        "confidence": 1.0,
        "affected_files": ["auth/auth_service.py"],
        "affected_function": "authenticate",
        "reasoning": [
            "login.py passes password directly to authenticate()",
            "auth_service.py calls .strip() without checking for None",
        ],
    }
    mock_diff = """--- a/auth/auth_service.py
+++ b/auth/auth_service.py
@@ -15,7 +15,9 @@
     \"\"\"
     Validates a username/password pair against stored user records.
     \"\"\"
-    password = password.strip()
+    if password is None:
+        return {"success": False, "message": "Password required"}
+    password = password.strip()
"""
    mock_test_result = {"passed": True, "total": 4, "passed_count": 4, "failed_count": 0}

    print("--- PR Title ---")
    print(build_pr_title(mock_root_cause))

    print("\n--- PR Description ---")
    print(build_pr_description(mock_root_cause, mock_diff, mock_test_result))
