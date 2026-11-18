"""Rule-based classifier and extractor.

Used when no Gemini API key is configured (offline / demo / tests) and as an automatic
fallback when the LLM call fails. Results carry deliberately modest confidence so that
they are routed to human review unless they are strongly corroborated by the text.
"""

from __future__ import annotations

import re
from typing import Any

from app.schemas.documents import DocumentType
from app.schemas.extraction import RawExtraction
from app.utils import parse_date, parse_number

HEURISTIC_CONFIDENCE = 0.65

_KEYWORDS: dict[DocumentType, dict[str, float]] = {
    DocumentType.INVOICE: {
        "invoice": 3, "bill to": 2, "due date": 2, "amount due": 2, "invoice number": 3,
        "invoice no": 3, "remit": 1, "purchase order": 1, "net 30": 1, "balance due": 2,
    },
    DocumentType.RECEIPT: {
        "receipt": 3, "cashier": 2, "change due": 2, "thank you for shopping": 2, "visa": 1,
        "mastercard": 1, "cash": 1, "store": 1, "tip": 1, "transaction": 1, "register": 1,
    },
    DocumentType.CONTRACT: {
        "agreement": 3, "contract": 2, "hereby": 2, "parties": 2, "whereas": 2,
        "governing law": 2, "terminate": 1, "in witness whereof": 2, "effective date": 1,
        "obligations": 1, "shall": 1,
    },
}

MONEY = r"[-]?[$€£¥₹]?\s?\d{1,3}(?:[,\s]\d{3})*(?:[.,]\d{2})|[-]?[$€£¥₹]?\s?\d+(?:[.,]\d{2})"
DATE_PATTERNS = [
    r"\d{4}-\d{2}-\d{2}",
    r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}",
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}",
    r"\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?,?\s+\d{4}",
]
DATE = "(" + "|".join(DATE_PATTERNS) + ")"
CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR"}
CURRENCY_CODES = ("USD", "EUR", "GBP", "EGP", "SAR", "AED", "CAD", "AUD", "INR", "JPY", "CHF")


def classify(text: str) -> tuple[DocumentType, float]:
    lower = text.lower()
    scores = {
        t: sum(w * min(lower.count(k), 3) for k, w in kws.items()) for t, kws in _KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    total = sum(scores.values())
    if scores[best] < 3:
        return DocumentType.GENERIC, 0.5
    # invoice and receipt share vocabulary; their margin determines confidence
    margin = scores[best] / total if total else 0
    return best, round(min(0.9, 0.45 + 0.5 * margin), 2)


# --------------------------------------------------------------------------- helpers
def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _labelled(text: str, labels: str, value: str) -> str | None:
    m = re.search(rf"(?:{labels})\s*[:#.]?\s*(?:no\.?|number|#)?\s*[:#]?\s*{value}", text, re.I)
    return m.group(1).strip() if m else None


def _money(s: str) -> float | None:
    try:
        return parse_number(s)
    except ValueError:
        return None


def _amount_on_line(lines: list[str], include: tuple[str, ...], exclude: tuple[str, ...] = ()) -> tuple[float | None, str | None]:
    """Return the last monetary amount on the last line matching one of ``include``."""
    found: tuple[float | None, str | None] = (None, None)
    for line in lines:
        low = line.lower()
        if any(k in low for k in include) and not any(k in low for k in exclude):
            amounts = re.findall(MONEY, line)
            if amounts:
                found = (_money(amounts[-1]), line)
    return found


def _currency(text: str) -> str | None:
    for code in CURRENCY_CODES:
        if re.search(rf"\b{code}\b", text):
            return code
    for sym, code in CURRENCY_SYMBOLS.items():
        if sym in text:
            return code
    return None


def _date_after(text: str, labels: str) -> tuple[str | None, str | None]:
    m = re.search(rf"(?:{labels})\s*[:]?\s*{DATE}", text, re.I)
    if not m:
        return None, None
    try:
        return parse_date(m.group(1)).isoformat(), m.group(0)
    except ValueError:
        return None, None


def _first_date(text: str) -> tuple[str | None, str | None]:
    m = re.search(DATE, text)
    if not m:
        return None, None
    try:
        return parse_date(m.group(1)).isoformat(), m.group(0)
    except ValueError:
        return None, None


class _Builder:
    def __init__(self) -> None:
        self.raw = RawExtraction()

    def set(self, path: str, value: Any, evidence: str | None = None, conf: float = HEURISTIC_CONFIDENCE) -> None:
        if value in (None, "", []):
            return
        from app.utils import set_by_path

        set_by_path(self.raw.data, path, value)
        self.raw.field_confidence[path] = conf
        if evidence:
            self.raw.evidence[path] = evidence[:200]


_LINE_ITEM = re.compile(
    rf"^(?P<desc>[A-Za-z].*?)\s+(?P<qty>\d+(?:\.\d+)?)\s+(?:x\s+)?(?P<unit>{MONEY})\s+(?P<amount>{MONEY})$"
)
_SIMPLE_ITEM = re.compile(rf"^(?P<desc>[A-Za-z].*?)\s+(?P<amount>{MONEY})$")
_TOTAL_WORDS = ("total", "tax", "vat", "gst", "subtotal", "balance", "amount due", "change", "tip",
                "cash", "visa", "card", "discount", "shipping", "paid")


def _line_items(lines: list[str], allow_simple: bool) -> list[dict]:
    items = []
    for line in lines:
        low = line.lower()
        if any(w in low for w in _TOTAL_WORDS):
            continue
        clean = re.sub(r"\s*\|\s*", " ", line)
        m = _LINE_ITEM.match(clean)
        if m:
            items.append({
                "description": m["desc"].strip(" -:"), "quantity": float(m["qty"]),
                "unit_price": _money(m["unit"]), "amount": _money(m["amount"]),
            })
        elif allow_simple and (m := _SIMPLE_ITEM.match(clean)) and not re.search(DATE, line):
            items.append({"description": m["desc"].strip(" -:"), "quantity": 1,
                          "unit_price": _money(m["amount"]), "amount": _money(m["amount"])})
    return items


# --------------------------------------------------------------------------- extractors
def extract(text: str, doc_type: DocumentType) -> RawExtraction:
    return {
        DocumentType.INVOICE: _extract_invoice,
        DocumentType.RECEIPT: _extract_receipt,
        DocumentType.CONTRACT: _extract_contract,
        DocumentType.GENERIC: _extract_generic,
    }[doc_type](text)


def _extract_invoice(text: str) -> RawExtraction:
    b, lines = _Builder(), _lines(text)
    m = re.search(r"invoice\s*(?:no\.?|number|num|#|id)\s*[:#.]?\s*([A-Z0-9][A-Z0-9\-/_.]*\d[A-Z0-9\-/_]*)", text, re.I)
    if m:
        b.set("invoice_number", m.group(1).rstrip("."), m.group(0))
    date, ev = _date_after(text, r"invoice\s+date|date\s+of\s+issue|issue\s+date|dated?")
    b.set("invoice_date", date, ev)
    date, ev = _date_after(text, r"due\s+date|payment\s+due|due")
    b.set("due_date", date, ev)
    po = _labelled(text, r"P\.?O\.?|purchase\s+order", r"([A-Z0-9\-]{3,})")
    b.set("purchase_order", po)
    vendor = next((ln for ln in lines if "invoice" not in ln.lower() and re.search(r"[A-Za-z]{3}", ln)), None)
    b.set("vendor.name", vendor, vendor, conf=0.55)
    m = re.search(r"bill(?:ed)?\s+to\s*[:]?\s*\n?\s*([^\n]+)", text, re.I)
    if m:
        b.set("customer.name", m.group(1).strip(), m.group(0))
    emails = re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    if emails:
        b.set("vendor.email", emails[0], emails[0], conf=0.5)
    b.set("currency", _currency(text))
    b.set("line_items", _line_items(lines, allow_simple=False))
    sub, ev = _amount_on_line(lines, ("subtotal", "sub total", "sub-total"))
    b.set("subtotal", sub, ev)
    tax, ev = _amount_on_line(lines, ("tax", "vat", "gst"), ("tax id", "vat no", "tax no", "vat number"))
    b.set("tax_amount", tax, ev)
    m = re.search(r"(?:tax|vat|gst)[^\n%]*?(\d{1,2}(?:\.\d+)?)\s*%", text, re.I)
    if m:
        b.set("tax_rate", float(m.group(1)), m.group(0))
    disc, ev = _amount_on_line(lines, ("discount",))
    b.set("discount", disc, ev)
    ship, ev = _amount_on_line(lines, ("shipping", "freight", "delivery"))
    b.set("shipping", ship, ev)
    total, ev = _amount_on_line(lines, ("total", "amount due", "balance due"), ("subtotal", "sub total", "sub-total"))
    b.set("total_amount", total, ev)
    m = re.search(r"(net\s*\d+|due\s+on\s+receipt|payment\s+terms?\s*[:]?\s*[^\n]+)", text, re.I)
    if m:
        b.set("payment_terms", m.group(1).strip(), m.group(0))
    return b.raw


def _extract_receipt(text: str) -> RawExtraction:
    b, lines = _Builder(), _lines(text)
    if lines:
        b.set("merchant.name", lines[0], lines[0], conf=0.6)
    phone = re.search(r"\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}", text)
    if phone:
        b.set("merchant.phone", phone.group(0), phone.group(0))
    m = re.search(r"(?:receipt|trans(?:action)?|order|ticket)\s*(?:no\.?|number|#|id)\s*[:#]?\s*([A-Z0-9-]{3,})", text, re.I)
    if m:
        b.set("receipt_number", m.group(1), m.group(0))
    date, ev = _first_date(text)
    b.set("transaction_date", date, ev)
    m = re.search(r"\b(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)", text, re.I)
    if m:
        b.set("transaction_time", m.group(1), m.group(0))
    m = re.search(r"\b(visa|mastercard|amex|american express|discover|debit|credit|cash)\b", text, re.I)
    if m:
        b.set("payment_method", m.group(1).lower(), m.group(0))
    m = re.search(r"(?:\*{2,}|x{2,}|#{2,})\s?(\d{4})", text, re.I)
    if m:
        b.set("card_last4", m.group(1), m.group(0))
    b.set("currency", _currency(text))
    b.set("line_items", _line_items(lines, allow_simple=True))
    sub, ev = _amount_on_line(lines, ("subtotal", "sub total", "sub-total"))
    b.set("subtotal", sub, ev)
    tax, ev = _amount_on_line(lines, ("tax", "vat", "gst"))
    b.set("tax_amount", tax, ev)
    tip, ev = _amount_on_line(lines, ("tip", "gratuity"))
    b.set("tip", tip, ev)
    total, ev = _amount_on_line(lines, ("total",), ("subtotal", "sub total", "sub-total", "total items"))
    b.set("total_amount", total, ev)
    return b.raw


def _extract_contract(text: str) -> RawExtraction:
    b, lines = _Builder(), _lines(text)
    title = next((ln for ln in lines[:10] if re.search(r"agreement|contract|lease|addendum", ln, re.I)), None)
    b.set("title", title, title)
    if title:
        b.set("contract_type", re.sub(r"(?i)\s*(agreement|contract)\s*$", "", title).strip().lower() or None, title, conf=0.5)
    m = re.search(r"(?:contract|agreement)\s*(?:no\.?|number|#|id)\s*[:#]?\s*([A-Z0-9-]{3,})", text, re.I)
    if m:
        b.set("contract_number", m.group(1), m.group(0))
    m = re.search(r"between\s+(.+?)\s*(?:\(([^)]*)\))?[,;]?\s+and\s+(.+?)\s*(?:\(([^)]*)\))?(?:[.,;\n]|$)", text, re.I | re.S)
    if m:
        parties = []
        for name, role in ((m.group(1), m.group(2)), (m.group(3), m.group(4))):
            name = re.sub(r"\s+", " ", name).strip(" ,")
            role = re.sub(r"(?i)hereinafter|referred to as|the|[\"“”']", "", role or "").strip() or None
            parties.append({"name": name[:120], "role": role})
        b.set("parties", parties, m.group(0)[:200])
    date, ev = _date_after(text, r"effective\s+(?:as\s+of|date|on)?|commenc\w+\s+on|dated|made\s+on")
    b.set("effective_date", date, ev)
    date, ev = _date_after(text, r"expir\w*\s+(?:on|date)?|terminat\w+\s+on|until|end\s+date")
    b.set("expiration_date", date, ev)
    m = re.search(r"(?:term\s+of|period\s+of|for\s+a\s+(?:term|period)\s+of)\s+((?:\w+\s+)?\(?\d*\)?\s*(?:months?|years?|days?))", text, re.I)
    if m:
        b.set("term", m.group(1).strip(), m.group(0))
    m = re.search(r"laws?\s+of\s+(?:the\s+)?([A-Z][\w ]+?)(?:[.,;\n]|$)", text)
    if m:
        b.set("governing_law", m.group(1).strip(), m.group(0))
    m = re.search(rf"(?:total|contract)\s+(?:value|fee|price|amount|consideration)[^\d$€£]*({MONEY})", text, re.I)
    if m:
        b.set("contract_value", _money(m.group(1)), m.group(0))
    b.set("currency", _currency(text))
    b.set("confidentiality", True if re.search(r"confidential", text, re.I) else None)
    obligations = [s.strip() for s in re.findall(r"[^.\n]*\bshall\b[^.\n]*\.", text)][:8]
    b.set("key_obligations", obligations, conf=0.5)
    return b.raw


def _extract_generic(text: str) -> RawExtraction:
    b, lines = _Builder(), _lines(text)
    if lines:
        b.set("title", lines[0][:150], lines[0])
    date, ev = _first_date(text)
    b.set("document_date", date, ev)
    m = re.search(r"(?:ref(?:erence)?|no\.?|number|id)\s*[:#]\s*([A-Z0-9-]{3,})", text, re.I)
    if m:
        b.set("reference_number", m.group(1), m.group(0))
    kvs = []
    for line in lines:
        m = re.match(r"^([A-Za-z][\w /&().-]{1,40}):\s*(.+)$", line)
        if m:
            kvs.append({"key": m.group(1).strip(), "value": m.group(2).strip()})
    b.set("key_values", kvs[:20], conf=0.6)
    summary = " ".join(lines[1:4])[:300] if len(lines) > 1 else None
    b.set("summary", summary, conf=0.4)
    return b.raw
