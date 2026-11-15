def load_items(path):
    """Read one item per non-empty line."""
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]
