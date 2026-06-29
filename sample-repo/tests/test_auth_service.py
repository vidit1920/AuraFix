"""
test_auth_service.py

Includes the test that currently FAILS because of the bug:
test_empty_password_does_not_crash. This is the test AuraFix's
Test Validator should run after applying a patch — if the patch
is correct, this test (and all others) should pass.
"""

import pytest
from auth.auth_service import authenticate


def test_correct_login_succeeds():
    result = authenticate("alice", "correct-horse-battery-staple")
    assert result["success"] is True


def test_wrong_password_fails_cleanly():
    result = authenticate("alice", "wrong-password")
    assert result["success"] is False
    assert result["message"] == "Incorrect password"


def test_unknown_user_fails_cleanly():
    result = authenticate("nobody", "anything")
    assert result["success"] is False
    assert result["message"] == "User not found"


def test_empty_password_does_not_crash():
    """
    THIS TEST CURRENTLY FAILS. The bug: authenticate() calls
    password.strip() without a None check, so an empty password
    field (which arrives as None) raises AttributeError instead
    of returning a clean error response.
    """
    result = authenticate("alice", None)
    assert result["success"] is False
    assert "password" in result["message"].lower()
