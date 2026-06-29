"""
auth_service.py — core authentication logic.

THE BUG: authenticate() calls password.strip() without checking if
password is None first. When the login form submits an empty password
field, it arrives here as None (not ""), and .strip() on None raises
AttributeError, crashing the request instead of returning a clean
"password required" response.
"""

from auth.db import get_user


def authenticate(username: str, password: str) -> dict:
    """
    Validates a username/password pair against stored user records.
    """
    password = password.strip()  # BUG: crashes if password is None

    user = get_user(username)
    if user is None:
        return {"success": False, "message": "User not found"}

    if user["password"] != password:
        return {"success": False, "message": "Incorrect password"}

    return {"success": True, "message": "Login successful"}
