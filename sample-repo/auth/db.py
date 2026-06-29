"""
db.py — fake in-memory "database" so the sample repo runs standalone
without needing an actual database. Just enough to complete the call
chain: login.py -> auth_service.py -> db.py
"""

_FAKE_USERS = {
    "alice": {"username": "alice", "password": "correct-horse-battery-staple"},
    "bob": {"username": "bob", "password": "hunter2"},
}


def get_user(username: str) -> dict | None:
    """Looks up a user record by username. Returns None if not found."""
    return _FAKE_USERS.get(username)
