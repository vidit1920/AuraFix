from inventory.pagination import page_count, paginate
from inventory.store import list_products


def test_first_page_starts_at_first_item():
    assert paginate(list(range(10)), 1, 3) == [0, 1, 2]


def test_last_partial_page():
    assert paginate(list(range(10)), 4, 3) == [9]


def test_page_count():
    assert page_count(23, 10) == 3


def test_store_first_page():
    assert list_products(1)["items"][0] == "product-1"
