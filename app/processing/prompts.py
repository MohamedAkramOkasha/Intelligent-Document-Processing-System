"""Prompt templates for the Gemini classifier and extractor."""

from __future__ import annotations

import json

from app.schemas.documents import DocumentType, model_for

CLASSIFY_PROMPT = """You are a document classification engine for a back-office automation system.
Classify the document into exactly one of these types:
- invoice: a bill requesting payment for goods/services (has invoice number, amounts due)
- receipt: proof of a completed payment / point-of-sale purchase
- contract: an agreement between parties (terms, obligations, signatures)
- generic: any other business document (letters, forms, reports, statements, ...)

Return ONLY JSON: {{"document_type": "<type>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}

Document text (may be empty or contain OCR noise; an image of the document may be attached):
<<<
{text}
>>>"""

_TYPE_HINTS = {
    DocumentType.INVOICE: (
        "- vendor = the party issuing the invoice (seller); customer = bill-to / buyer.\n"
        "- subtotal = sum of line items before tax, discount and shipping; total_amount = final total.\n"
        "- tax_rate is a percentage number (15 for 15%), not a fraction.\n"
        "- Include every line item row; amount = line total."
    ),
    DocumentType.RECEIPT: (
        "- merchant = the store/business that issued the receipt.\n"
        "- total_amount is the amount actually paid (including tax and tip).\n"
        "- card_last4 = last four digits of the card if printed."
    ),
    DocumentType.CONTRACT: (
        "- parties: every contracting party with its role (e.g. 'Service Provider', 'Client').\n"
        "- key_obligations: up to 8 short sentences summarising the main obligations.\n"
        "- termination_terms / renewal_terms / payment_terms: concise summaries of the clauses.\n"
        "- signatories: names (and titles) of people who signed."
    ),
    DocumentType.GENERIC: (
        "- summary: 1-3 sentences describing the document.\n"
        "- key_values: the most important labelled facts (max 20)."
    ),
}

EXTRACT_PROMPT = """You are an expert data-extraction engine for {doc_type} documents.
Extract the information into a JSON object whose "data" member conforms to this JSON Schema:
{schema}

Return ONLY a JSON object with exactly this top-level shape:
{{
  "data": {{ ...object matching the schema... }},
  "field_confidence": {{ "<field path>": <0.0-1.0>, ... }},
  "evidence": {{ "<field path>": "<short verbatim snippet from the document>", ... }}
}}

Rules:
- Use only information present in the document. NEVER invent values; use null when absent.
- Dates in ISO format YYYY-MM-DD. Numbers as plain JSON numbers (no currency symbols or
  thousands separators). Currency as an ISO 4217 code (USD, EUR, GBP, EGP, ...).
- Field paths use dot notation with list indices, e.g. "vendor.name", "line_items[0].amount".
- Give a confidence for every non-null field:
  0.95-1.0 clearly printed and unambiguous; 0.75-0.94 minor ambiguity (OCR noise, inferred
  formatting); below 0.75 partially legible, guessed or inferred from context.
- For line items you may give a single confidence per row path such as "line_items[0]".
{hints}

Document text (may contain OCR errors; when an image/PDF is attached prefer it if they disagree):
<<<
{text}
>>>"""

REPAIR_PROMPT = """Your previous JSON output failed validation with these errors:
{errors}

Previous output:
{previous}

Return the corrected JSON object only, keeping the same top-level shape."""


def build_extract_prompt(doc_type: DocumentType, text: str) -> str:
    schema = json.dumps(model_for(doc_type).model_json_schema(), separators=(",", ":"))
    return EXTRACT_PROMPT.format(
        doc_type=doc_type.value, schema=schema, hints=_TYPE_HINTS[doc_type], text=text or "(no text layer)"
    )
