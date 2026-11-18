"""Pydantic models describing the structured data extracted from each document type.

Values coming from the LLM (or OCR heuristics) are messy: ``"$1,200.00"``, ``"March 3rd, 2024"``,
``"usd"``. The annotated types below normalise them so downstream code always sees clean
``float`` / ``date`` / upper-case ISO currency values.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from app.utils import clean_str, parse_date, parse_number


def _currency(v: Any) -> str | None:
    s = clean_str(v)
    if not s:
        return None
    symbols = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR", "₦": "NGN", "E£": "EGP"}
    return symbols.get(s, s.upper()[:3] if s.isalpha() else s.upper())


Number = Annotated[float | None, BeforeValidator(parse_number)]
FlexDate = Annotated[date | None, BeforeValidator(parse_date)]
Text = Annotated[str | None, BeforeValidator(clean_str)]
Currency = Annotated[str | None, BeforeValidator(_currency)]


class DocumentType(str, Enum):
    INVOICE = "invoice"
    RECEIPT = "receipt"
    CONTRACT = "contract"
    GENERIC = "generic"


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class Party(_Base):
    @model_validator(mode="before")
    @classmethod
    def _from_string(cls, value: Any) -> Any:
        return {"name": value} if isinstance(value, str) else value

    name: Text = Field(None, description="Legal or trading name")
    role: Text = Field(None, description="Role in the document, e.g. vendor, client, landlord")
    address: Text = Field(None, description="Full postal address on one line")
    tax_id: Text = Field(None, description="VAT / GST / EIN / tax registration number")
    email: Text = None
    phone: Text = None


class LineItem(_Base):
    description: Text = None
    quantity: Number = None
    unit_price: Number = None
    amount: Number = Field(None, description="Line total (quantity x unit price)")


class KeyValue(_Base):
    key: Text = None
    value: Text = None


class InvoiceData(_Base):
    invoice_number: Text = None
    invoice_date: FlexDate = None
    due_date: FlexDate = None
    purchase_order: Text = Field(None, description="PO number if present")
    vendor: Party = Field(default_factory=Party, description="Seller / issuer of the invoice")
    customer: Party = Field(default_factory=Party, description="Buyer / bill-to party")
    currency: Currency = Field(None, description="ISO 4217 code, e.g. USD")
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: Number = Field(None, description="Sum of line items before tax/discount")
    discount: Number = None
    shipping: Number = None
    tax_rate: Number = Field(None, description="Tax rate in percent, e.g. 15 for 15%")
    tax_amount: Number = None
    total_amount: Number = Field(None, description="Final invoice total")
    amount_due: Number = None
    payment_terms: Text = None
    bank_details: Text = None
    notes: Text = None


class ReceiptData(_Base):
    merchant: Party = Field(default_factory=Party)
    receipt_number: Text = None
    transaction_date: FlexDate = None
    transaction_time: Text = None
    payment_method: Text = Field(None, description="cash, card, visa, mastercard, ...")
    card_last4: Text = None
    currency: Currency = None
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: Number = None
    tax_amount: Number = None
    tip: Number = None
    total_amount: Number = None


class ContractData(_Base):
    title: Text = None
    contract_number: Text = None
    contract_type: Text = Field(None, description="e.g. service agreement, NDA, lease, employment")
    parties: list[Party] = Field(default_factory=list)
    effective_date: FlexDate = None
    expiration_date: FlexDate = None
    term: Text = Field(None, description="Contract duration as written, e.g. '12 months'")
    contract_value: Number = None
    currency: Currency = None
    payment_terms: Text = None
    renewal_terms: Text = None
    termination_terms: Text = None
    governing_law: Text = None
    confidentiality: bool | None = None
    key_obligations: list[str] = Field(default_factory=list)
    signatories: list[str] = Field(default_factory=list)
    signed_date: FlexDate = None


class GenericData(_Base):
    title: Text = None
    document_date: FlexDate = None
    issuer: Text = None
    recipient: Text = None
    reference_number: Text = None
    summary: Text = None
    key_values: list[KeyValue] = Field(default_factory=list)


DOCUMENT_MODELS: dict[DocumentType, type[_Base]] = {
    DocumentType.INVOICE: InvoiceData,
    DocumentType.RECEIPT: ReceiptData,
    DocumentType.CONTRACT: ContractData,
    DocumentType.GENERIC: GenericData,
}

# Fields that must be present for a document to be auto-approved.
REQUIRED_FIELDS: dict[DocumentType, list[str]] = {
    DocumentType.INVOICE: ["invoice_number", "invoice_date", "vendor.name", "total_amount"],
    DocumentType.RECEIPT: ["merchant.name", "transaction_date", "total_amount"],
    DocumentType.CONTRACT: ["title", "parties", "effective_date"],
    DocumentType.GENERIC: ["title"],
}

# Headline fields shown in tables / exports.
SUMMARY_FIELDS: dict[DocumentType, dict[str, str]] = {
    DocumentType.INVOICE: {
        "reference": "invoice_number", "date": "invoice_date", "counterparty": "vendor.name",
        "total": "total_amount", "currency": "currency",
    },
    DocumentType.RECEIPT: {
        "reference": "receipt_number", "date": "transaction_date", "counterparty": "merchant.name",
        "total": "total_amount", "currency": "currency",
    },
    DocumentType.CONTRACT: {
        "reference": "contract_number", "date": "effective_date", "counterparty": "parties[0].name",
        "total": "contract_value", "currency": "currency",
    },
    DocumentType.GENERIC: {
        "reference": "reference_number", "date": "document_date", "counterparty": "issuer",
    },
}


def model_for(doc_type: DocumentType | str) -> type[_Base]:
    return DOCUMENT_MODELS[DocumentType(doc_type)]
