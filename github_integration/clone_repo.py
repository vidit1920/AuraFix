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
    # without prompting. Public clones work fine without it.
    clone_url = url
    token = os.environ.get("GITHUB_TOKEN")
    if token and token != "your-github-token-here" and url.startswith("https://github.com/"):
        clone_url = url.replace("https://", f"https://{token}@", 1)

    try:
        result = subprocess.run(
            ["git", "clone", "--depth", str(depth), clone_url, target_dir],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "repo_path": None, "repo_full_name": repo_full_name,
                "error": "git clone timed out after 180s"}

    if result.returncode != 0:
        # Scrub the token out of any error text before surfacing it.
        err = result.stderr.strip()
        if token:
            err = err.replace(token, "***")
        return {"success": False, "repo_path": None, "repo_full_name": repo_full_name,
                "error": err or "git clone failed"}

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
