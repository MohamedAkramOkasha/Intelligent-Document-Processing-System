"""Test configuration: isolated data dir, heuristic extractor, no network."""

from __future__ import annotations

import os
import tempfile

# Must run before any `app` import: settings and the DB engine are created at import time.
_TMP = tempfile.mkdtemp(prefix="idp_test_")
os.environ.update({
    "DATA_DIR": _TMP,
    "GEMINI_API_KEY": "",
    "GOOGLE_API_KEY": "",
    "API_KEY": "",
    "DATABASE_URL": "",
})

import pytest  # noqa: E402

from app.processing.llm import Attachment  # noqa: E402

INVOICE_TEXT = """Sunrise Office Supplies
INVOICE
Invoice No: SOS-10593
Invoice Date: March 3, 2026
Due Date: April 2, 2026
Bill To: Lakeside Dental Clinic
A4 Copy Paper (box)     10       $32.00   $320.00
Ink Cartridge Black      4       $45.00   $180.00
Subtotal: $500.00
Sales Tax (10%): $50.00
Total Due: $550.00
"""


class FakeGemini:
    """Stands in for GeminiClient; returns queued responses (or raises queued exceptions)."""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[tuple[str, list[Attachment]]] = []

    def generate_json(self, prompt: str, attachments=None):
        self.calls.append((prompt, attachments or []))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def invoice_text() -> str:
    return INVOICE_TEXT


@pytest.fixture
def gemini_invoice_response() -> dict:
    return {
        "data": {
            "invoice_number": "SOS-10593",
            "invoice_date": "2026-03-03",
            "due_date": "2026-04-02",
            "vendor": {"name": "Sunrise Office Supplies"},
            "customer": {"name": "Lakeside Dental Clinic"},
            "currency": "usd",
            "line_items": [
                {"description": "A4 Copy Paper (box)", "quantity": 10, "unit_price": "$32.00", "amount": "$320.00"},
                {"description": "Ink Cartridge Black", "quantity": 4, "unit_price": 45, "amount": 180},
            ],
            "subtotal": "500.00",
            "tax_rate": 10,
            "tax_amount": 50,
            "total_amount": "$550.00",
        },
        "field_confidence": {
            "invoice_number": 0.99, "invoice_date": 0.97, "due_date": 0.97, "vendor.name": 0.98,
            "customer.name": 0.95, "currency": 0.9, "line_items": 0.95, "subtotal": 0.98,
            "tax_rate": 0.95, "tax_amount": 0.97, "total_amount": 0.99,
        },
        "evidence": {"invoice_number": "Invoice No: SOS-10593", "total_amount": "Total Due: $550.00"},
    }
