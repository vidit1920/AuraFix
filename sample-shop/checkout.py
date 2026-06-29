"""
checkout.py — entry point for the shop's checkout flow.

This is what the (not-implemented-here) web layer would call. It passes
the price and the chosen discount straight through to the pricing layer,
so the call chain is: checkout.py -> pricing/discount.py
"""

from pricing.discount import apply_discount


def calculate_checkout_total(price: float, discount_percent: float) -> float:
    """
    Computes the final price a customer pays after their discount.
    """
    if price < 0:
        raise ValueError("price cannot be negative")
    return apply_discount(price, discount_percent)
