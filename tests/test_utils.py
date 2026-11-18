from datetime import date

import pytest

from app.utils import flatten, get_by_path, parse_date, parse_number, path_ancestors, set_by_path


@pytest.mark.parametrize("raw, expected", [
    ("$1,234.56", 1234.56),
    ("1.234,56 €", 1234.56),
    ("(45.00)", -45.0),
    ("EGP 2,500", 2500.0),
    ("12,5", 12.5),
    (42, 42.0),
    ("", None),
    (None, None),
    ("n/a", None),
])
def test_parse_number(raw, expected):
    assert parse_number(raw) == expected


def test_parse_number_rejects_garbage():
    with pytest.raises(ValueError):
        parse_number("abc")


@pytest.mark.parametrize("raw, expected", [
    ("2026-03-03", date(2026, 3, 3)),
    ("March 3, 2026", date(2026, 3, 3)),
    ("03/04/2026", date(2026, 3, 4)),
    ("3 Mar 2026", date(2026, 3, 3)),
    (None, None),
])
def test_parse_date(raw, expected):
    assert parse_date(raw) == expected


def test_paths_roundtrip():
    data: dict = {}
    set_by_path(data, "vendor.name", "Acme")
    set_by_path(data, "line_items[1].amount", 10.0)
    assert data == {"vendor": {"name": "Acme"}, "line_items": [None, {"amount": 10.0}]}
    assert get_by_path(data, "line_items[1].amount") == 10.0
    assert get_by_path(data, "missing.path", "x") == "x"
    assert flatten({"a": {"b": 1}, "c": [{"d": 2}]}) == {"a.b": 1, "c[0].d": 2}


def test_path_ancestors():
    assert path_ancestors("line_items[2].amount") == [
        "line_items[2].amount", "line_items[].amount", "line_items.amount", "line_items"]
