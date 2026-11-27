from inventory.pagination import page_count, paginate

PRODUCTS = [f"product-{i}" for i in range(1, 24)]


def list_products(page: int, page_size: int = 10) -> dict:
    return {
        "page": page,
        "pages": page_count(len(PRODUCTS), page_size),
        "items": paginate(PRODUCTS, page, page_size),
    }
