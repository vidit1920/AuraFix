"""Pagination helpers for the inventory listing page."""


def page_count(total_items: int, page_size: int) -> int:
    """Number of pages needed to show total_items, page_size per page."""
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    return (total_items + page_size - 1) // page_size


def paginate(items: list, page: int, page_size: int) -> list:
    """Items shown on a 1-based page."""
    if page < 1:
        raise ValueError("page numbers start at 1")
    start = (page - 1) * page_size
    return items[start:start + page_size]
