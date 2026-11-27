from cart.cart import add_item, cart_total


def test_new_carts_start_empty():
    first = add_item("apple")
    second = add_item("pear")
    assert first == [("apple", 1)]
    assert second == [("pear", 1)]


def test_add_to_existing_cart():
    cart = add_item("apple", 2, [])
    cart = add_item("pear", 1, cart)
    assert cart == [("apple", 2), ("pear", 1)]


def test_total():
    assert cart_total([("apple", 2), ("pear", 1)], {"apple": 0.5, "pear": 2.0}) == 3.0
