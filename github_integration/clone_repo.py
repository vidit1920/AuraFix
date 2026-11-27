"""
clone_repo.py — shallow-clones a GitHub repo so the pipeline can run
against a URL, not just a local path.

Kept deliberately small: the only real decision is clone depth. We use
`--depth 1` because the whole pipeline reads *current* file contents and
runs the *current* tests — it never inspects git history — so pulling the
full history would just be slower for no benefit.

The dashboard uses this for its "GitHub URL" input. For a private repo,
the clone needs a GITHUB_TOKEN with read access; we embed it in the clone
URL so git can authenticate non-interactively (the alternative, prompting
for credentials, would hang a headless run).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile


def parse_repo_full_name(url: str) -> str | None:
    """
    Extracts "owner/repo" from a GitHub URL, which is what PyGithub's
    create_pull() needs. Handles https and ssh forms, with or without a
    trailing ".git". Returns None if it doesn't look like a GitHub URL.

        https://github.com/octocat/Hello-World.git -> "octocat/Hello-World"
        git@github.com:octocat/Hello-World.git      -> "octocat/Hello-World"
    """
    match = re.search(r"github\.com[/:]([^/]+/[^/]+?)(?:\.git)?/?$", url.strip())
    return match.group(1) if match else None


def _friendly_clone_error(err: str, had_token: bool) -> str:
    """Turn git's stderr into something a user can act on."""
    low = err.lower()
    if any(m in low for m in ("repository not found", "could not read username", "could not read password",
                              "authentication failed", "terminal prompts disabled", "403", "404")):
        if had_token:
            return ("Couldn't access this repository. Check the URL, and that your GitHub token "
                    "can read it (for a private repository it needs the repo scope).")
        return ("Couldn't access this repository. Check the URL. If it's private, add a GitHub "
                "token on the Setup page that can read it, or make the repository public.")
    if "could not resolve host" in low or "unable to access" in low:
        return "Couldn't reach GitHub. Check the server's internet connection and try again."
    return err or "git clone failed"


def clone_repo(url: str, depth: int = 1, dest: str | None = None) -> dict:
    """
    Shallow-clones `url` into a temp directory (or `dest` if given).

    Returns:
        {
            "success": bool,
            "repo_path": str | None,      # local path to the clone
            "repo_full_name": str | None, # "owner/repo", for PR creation
            "error": str | None,
        }
    """
    repo_full_name = parse_repo_full_name(url)
    target_dir = dest or tempfile.mkdtemp(prefix="aurafix_repo_")

    # If a token is present, embed it so a private clone authenticates
    # without prompting. Public clones work fine without it. The token is the
    # signed-in user's own (per-session), falling back to the environment.
    from github_integration.credentials import authenticated_github_url, get_github_token, git_env
    clone_url = url
    token = get_github_token()
    if token and repo_full_name and url.startswith("https://github.com/"):
        clone_url = authenticated_github_url(repo_full_name, token)

    try:
        result = subprocess.run(
            # credential.helper= : no stored credentials or login pop-ups; the
            # only auth is the token above (public repos need none).
            ["git", "-c", "credential.helper=", "clone", "--depth", str(depth), clone_url, target_dir],
            capture_output=True,
            text=True,
            timeout=180,
            env=git_env(),
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "repo_path": None, "repo_full_name": repo_full_name,
                "error": "git clone timed out after 180s"}

    if result.returncode != 0 and token and clone_url != url:
        # A wrong or expired token makes GitHub reject even a public repo that
        # needs no auth at all, so retry once anonymously before giving up.
        shutil.rmtree(target_dir, ignore_errors=True)
        os.makedirs(target_dir, exist_ok=True)
        anon = subprocess.run(
            ["git", "-c", "credential.helper=", "clone", "--depth", str(depth), url, target_dir],
            capture_output=True, text=True, timeout=180, env=git_env(),
        )
        if anon.returncode == 0:
            return {"success": True, "repo_path": target_dir, "repo_full_name": repo_full_name,
                    "error": None}

    if result.returncode != 0:
        # Scrub the token out of any error text before surfacing it.
        err = result.stderr.strip()
        if token:
            err = err.replace(token, "***")
        return {"success": False, "repo_path": None, "repo_full_name": repo_full_name,
                "error": _friendly_clone_error(err, bool(token))}

    return {"success": True, "repo_path": target_dir, "repo_full_name": repo_full_name, "error": None}


if __name__ == "__main__":
    # Quick manual test — run with: python -m github_integration.clone_repo
    for u in [
        "https://github.com/octocat/Hello-World.git",
        "git@github.com:octocat/Hello-World.git",
        "https://github.com/octocat/Hello-World",
        "not-a-github-url",
    ]:
        print(f"{u!r:60} -> {parse_repo_full_name(u)}")
