from appconfig.parser import parse_config


def test_simple_pairs():
    assert parse_config("A=1\nB = two") == {"A": "1", "B": "two"}


def test_comments_and_blank_lines():
    assert parse_config("# c\n\nA=1") == {"A": "1"}


def test_value_containing_equals():
    cfg = parse_config("DATABASE_URL=postgres://db/app?sslmode=require")
    assert cfg["DATABASE_URL"] == "postgres://db/app?sslmode=require"
