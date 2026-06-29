"""
test_discount.py

Includes the tests that currently FAIL because of the discount bug.
After AuraFix applies the correct percentage formula, all of these
should pass.
"""

from pricing.discount import apply_discount


def test_no_discount_returns_full_price():
    # 0% off -> full price. (Passes even with the bug: 100 - 0 == 100.)
    assert apply_discount(100, 0) == 100


def test_ten_percent_off_200():
    """THIS TEST CURRENTLY FAILS. 10% off 200 should be 180, but the bug
    returns 190 (200 - 10)."""
    assert apply_discount(200, 10) == 180


def test_half_off():
    """THIS TEST CURRENTLY FAILS. 50% off 50 should be 25, but the bug
    returns 0 (50 - 50)."""
    assert apply_discount(50, 50) == 25
