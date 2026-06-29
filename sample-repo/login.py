"""
login.py — handles the login flow for the sample app.

This module deliberately contains one bug for AuraFix to find and fix:
when the password field is submitted empty, the frontend sends None
instead of an empty string, and authenticate() passes it straight
through to auth_service without checking.
"""

from auth.auth_service import authenticate


def handle_login_request(username: str, password: str) -> dict:
    """
    Entry point for a login attempt. Called by the (not implemented
    here) web framework route handler.
    """
    if not username:
        return {"success": False, "message": "Username is required"}

    result = authenticate(username, password)
    return result
