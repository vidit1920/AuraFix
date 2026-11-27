def parse_config(text: str) -> dict:
    """Parse KEY=VALUE lines. Blank lines and # comments are ignored."""
    config = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        config[key.strip()] = value.strip()
    return config
