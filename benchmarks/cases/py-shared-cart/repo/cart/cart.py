def add_item(name: str, qty: int = 1, items: list = []) -> list:
    """Return the cart (a list of (name, qty)) with the item added."""
    items.append((name, qty))
    return items


def cart_total(items: list, prices: dict) -> float:
    return sum(prices[name] * qty for name, qty in items)
