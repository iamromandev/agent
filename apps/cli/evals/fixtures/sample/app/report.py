from app.items import load_items


def summary(path):
    items = load_items(path)
    return f"{len(items)} items"
