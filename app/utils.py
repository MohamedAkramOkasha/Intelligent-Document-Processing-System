"""Shared helpers: value coercion and nested-path manipulation."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from typing import Any

from dateutil import parser as date_parser

_CURRENCY_CHARS = re.compile(r"[^\d,.\-()]")
_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def utcnow() -> datetime:
    """Naive UTC timestamp (SQLite does not keep tz info, so we store naive UTC consistently)."""
    return datetime.now(UTC).replace(tzinfo=None)


# --------------------------------------------------------------------------- numbers
def parse_number(value: Any) -> float | None:
    """Parse numbers such as ``"$1,234.56"``, ``"1.234,56 €"`` or ``"(45.00)"``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s or s.lower() in {"null", "none", "n/a", "na", "-"}:
        return None
    negative = s.startswith("(") and s.endswith(")") or s.startswith("-") or s.endswith("-")
    s = _CURRENCY_CHARS.sub("", s).strip("()-")
    if not s:
        raise ValueError(f"not a number: {value!r}")
    if "," in s and "." in s:
        # whichever separator comes last is the decimal separator
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        head, _, tail = s.rpartition(",")
        s = f"{head.replace(',', '')}.{tail}" if len(tail) in (1, 2) else s.replace(",", "")
    try:
        number = float(s)
    except ValueError as exc:
        raise ValueError(f"not a number: {value!r}") from exc
    return -number if negative else number


# --------------------------------------------------------------------------- dates
def parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()
    if s.lower() in {"null", "none", "n/a"}:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        pass
    try:
        return date_parser.parse(s, dayfirst=False, fuzzy=False).date()
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"unparseable date: {value!r}") from exc


def clean_str(value: Any) -> str | None:
    if value is None:
        return None
    s = re.sub(r"\s+", " ", str(value)).strip()
    return s or None


# --------------------------------------------------------------------------- paths
def parse_path(path: str) -> list[str | int]:
    return [int(idx) if idx else key for key, idx in _PATH_TOKEN.findall(path)]


def get_by_path(data: Any, path: str, default: Any = None) -> Any:
    cur = data
    for tok in parse_path(path):
        try:
            cur = cur[tok]
        except (KeyError, IndexError, TypeError):
            return default
    return cur


def set_by_path(data: dict, path: str, value: Any) -> None:
    tokens = parse_path(path)
    if not tokens:
        raise ValueError("empty path")
    cur: Any = data
    for tok, nxt in zip(tokens, tokens[1:]):
        container: Any = [] if isinstance(nxt, int) else {}
        if isinstance(tok, int):
            while len(cur) <= tok:
                cur.append(None)
            if cur[tok] is None:
                cur[tok] = container
        elif cur.get(tok) is None:
            cur[tok] = container
        cur = cur[tok]
    last = tokens[-1]
    if isinstance(last, int):
        while len(cur) <= last:
            cur.append(None)
    cur[last] = value


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts/lists into ``{"vendor.name": ..., "line_items[0].amount": ...}``."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        if not obj and prefix:
            out[prefix] = None
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        if not obj and prefix:
            out[prefix] = []
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def generic_path(path: str) -> str:
    """``line_items[3].amount`` -> ``line_items[].amount``."""
    return re.sub(r"\[\d+\]", "[]", path)


def path_ancestors(path: str) -> list[str]:
    """Candidate keys for looking up a path, from most to least specific."""
    candidates = [path, generic_path(path), re.sub(r"\[\d+\]", "", path)]
    tokens = re.sub(r"\[\d+\]", "", path).split(".")
    for i in range(len(tokens) - 1, 0, -1):
        candidates.append(".".join(tokens[:i]))
    seen: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.append(c)
    return seen


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value
