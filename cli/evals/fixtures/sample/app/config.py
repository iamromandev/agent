def parse_config(text):
    """Parse `key=value` lines into a dict."""
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    return result
