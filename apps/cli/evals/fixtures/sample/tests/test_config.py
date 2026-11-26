from app.config import parse_config


def test_parse_config():
    assert parse_config("a = 1\n# c\nb=2") == {"a": "1", "b": "2"}
