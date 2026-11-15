from app.config import parse_config


def run(text):
    return parse_config(text)


def reload(text):
    settings = parse_config(text)
    settings["reloaded"] = "yes"
    return settings
