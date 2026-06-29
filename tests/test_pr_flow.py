"""
test_pr_flow.py — deterministic tests for the real PR / open-source
contribution flow, with NO network and NO real PR ever opened.

We mock PyGithub (the GitHub API) and the git push so we can assert the
*logic* of open_pull_request():
  - direct mode (you can push to upstream): push to upstream, same-repo PR.
  - fork mode (you can't): fork upstream, push to the fork, cross-fork PR
    with head "you:branch".
  - missing token: a clear, non-crashing error.

Run with:  python -m pytest tests/test_pr_flow.py -v
"""

import pytest
from github import GithubException

import github_integration.pr_agent as pr_agent


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------
class _FakeUser:
    login = "me"


class _FakePerms:
    def __init__(self, push):
        self.push = push


class _FakePR:
    def __init__(self, repo_full_name):
        self.html_url = f"https://github.com/{repo_full_name}/pull/1"


class _FakeRepo:
    def __init__(self, full_name, can_push, registry, default_branch="main"):
        self.full_name = full_name
        self.name = full_name.split("/")[-1]
        self.permissions = _FakePerms(can_push)
        self.default_branch = default_branch
        self._registry = registry
        self.created_pulls = []

    def create_fork(self):
        fork = _FakeRepo(f"me/{self.name}", can_push=True, registry=self._registry)
        self._registry[fork.full_name] = fork  # now queryable, like a real fork becoming ready
        return fork

    def create_pull(self, title, body, head, base, draft=False, maintainer_can_modify=True):
        self.created_pulls.append({"title": title, "head": head, "base": base, "draft": draft})
        return _FakePR(self.full_name)


def _install(monkeypatch, registry):
    """Point pr_agent at a fake GitHub backed by `registry`, and stub the
    git push + remote setup so nothing touches a real repo or network."""
    class _FakeGithub:
        def __init__(self, token):
            self.token = token

        def get_user(self):
            return _FakeUser()

        def get_repo(self, full_name):
            if full_name not in registry:
                raise GithubException(404, {"message": "Not Found"}, None)
            return registry[full_name]

    monkeypatch.setenv("GITHUB_TOKEN", "ghp_faketoken123")
    monkeypatch.setattr(pr_agent, "Github", _FakeGithub)
    monkeypatch.setattr(pr_agent, "_set_push_remote", lambda *a, **k: None)
    monkeypatch.setattr(pr_agent, "push_branch",
                        lambda repo_path, branch_name, remote_name="origin": {"success": True, "error": None})


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_fork_mode_opens_cross_fork_pr(monkeypatch):
    """No push access to upstream -> fork it, push to the fork, open a
    cross-fork PR whose head is 'me:branch' on the UPSTREAM repo."""
    registry = {}
    upstream = _FakeRepo("owner/project", can_push=False, registry=registry)
    registry["owner/project"] = upstream
    _install(monkeypatch, registry)

    result = pr_agent.open_pull_request(
        repo_path="sample-repo",
        upstream_full_name="owner/project",
        branch_name="bugfix/aurafix-authenticate",
        title="Fix bug",
        description="body",
        draft=True,
    )

    assert result["success"] is True
    assert result["mode"] == "fork"
    assert result["fork_full_name"] == "me/project"
    assert result["url"] == "https://github.com/owner/project/pull/1"  # PR lives on upstream
    # The PR was opened on upstream with a cross-fork head and as a draft.
    assert len(upstream.created_pulls) == 1
    pr = upstream.created_pulls[0]
    assert pr["head"] == "me:bugfix/aurafix-authenticate"
    assert pr["base"] == "main"
    assert pr["draft"] is True


def test_direct_mode_opens_same_repo_pr(monkeypatch):
    """Push access to upstream -> push directly, open a same-repo PR with a
    plain branch head and no fork."""
    registry = {}
    upstream = _FakeRepo("me/myproject", can_push=True, registry=registry, default_branch="master")
    registry["me/myproject"] = upstream
    _install(monkeypatch, registry)

    result = pr_agent.open_pull_request(
        repo_path="sample-repo",
        upstream_full_name="me/myproject",
        branch_name="bugfix/x",
        title="Fix bug",
        description="body",
    )

    assert result["success"] is True
    assert result["mode"] == "direct"
    assert result["fork_full_name"] is None
    pr = upstream.created_pulls[0]
    assert pr["head"] == "bugfix/x"          # not cross-fork
    assert pr["base"] == "master"            # upstream's default branch
    assert pr["draft"] is False


def test_missing_token_fails_cleanly(monkeypatch):
    """No GITHUB_TOKEN -> a clear error, no crash, no PR."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    result = pr_agent.open_pull_request(
        repo_path="sample-repo",
        upstream_full_name="owner/project",
        branch_name="bugfix/x",
        title="t",
        description="d",
    )
    assert result["success"] is False
    assert "GITHUB_TOKEN" in result["error"]


def test_push_failure_reported(monkeypatch):
    """If the push fails, report it without claiming a PR was opened — and
    scrub the token out of the surfaced error."""
    registry = {}
    upstream = _FakeRepo("me/myproject", can_push=True, registry=registry)
    registry["me/myproject"] = upstream
    _install(monkeypatch, registry)
    monkeypatch.setattr(pr_agent, "push_branch",
                        lambda repo_path, branch_name, remote_name="origin":
                        {"success": False, "error": "fatal: https://ghp_faketoken123@github.com/... denied"})

    result = pr_agent.open_pull_request(
        repo_path="sample-repo", upstream_full_name="me/myproject",
        branch_name="bugfix/x", title="t", description="d",
    )
    assert result["success"] is False
    assert "Push to" in result["error"]
    assert "ghp_faketoken123" not in result["error"]   # token scrubbed
    assert upstream.created_pulls == []                 # no PR attempted
