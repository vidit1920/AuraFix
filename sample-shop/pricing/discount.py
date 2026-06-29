"""
discount.py — discount calculation for the shop.

THE BUG: apply_discount() treats the percentage as a flat amount to
subtract, instead of applying it as a percentage of the price. So a
"10% off" on a $200 item wrongly returns $190 (200 - 10) instead of
$180 (200 - 10%). This is a wrong-formula bug, not a crash — the code
runs fine and returns a plausible-looking number, which is exactly why
it slips past casual testing and needs a real fix, not a guard.
"""


def apply_discount(price: float, percent: float) -> float:
    """
    Returns the price after applying a `percent`% discount.

    Example: apply_discount(200, 10) should return 180.0 (10% off 200).
    """
    return price - percent  # BUG: subtracts percent as a flat amount, not a percentage
