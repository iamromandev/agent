from app.items import load_items


def test_load_items(tmp_path):
    path = tmp_path / "items.txt"
    path.write_text("a\n\nb\n")
    assert load_items(str(path)) == ["a", "b"]
