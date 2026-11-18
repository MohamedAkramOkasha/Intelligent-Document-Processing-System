from app.processing.validation import coerce, validate
from app.schemas.documents import DocumentType


def _rules(issues):
    return {i.rule for i in issues}


def test_coerce_normalises_values():
    data, issues = coerce(DocumentType.INVOICE, {
        "invoice_date": "March 3, 2026", "total_amount": "$1,200.50", "currency": "usd", "vendor": "Acme Ltd",
    })
    assert not issues
    assert data["invoice_date"] == "2026-03-03"
    assert data["total_amount"] == 1200.50
    assert data["currency"] == "USD"
    assert data["vendor"]["name"] == "Acme Ltd"


def test_coerce_nulls_invalid_values_instead_of_failing():
    data, issues = coerce(DocumentType.INVOICE, {"invoice_number": "A1", "invoice_date": "not a date",
                                                 "line_items": [{"quantity": "many", "amount": 5}]})
    assert data["invoice_number"] == "A1"
    assert data["invoice_date"] is None
    assert data["line_items"][0]["quantity"] is None
    assert {i.field for i in issues} == {"invoice_date", "line_items[0].quantity"}


def test_valid_invoice_has_no_errors():
    data, _ = coerce(DocumentType.INVOICE, {
        "invoice_number": "1", "invoice_date": "2026-01-01", "due_date": "2026-01-31",
        "vendor": {"name": "V"}, "subtotal": 100, "tax_amount": 10, "tax_rate": 10, "total_amount": 110,
        "line_items": [{"quantity": 2, "unit_price": 50, "amount": 100}],
    })
    assert [i for i in validate(DocumentType.INVOICE, data) if i.severity == "error"] == []


def test_invoice_math_and_dates():
    data, _ = coerce(DocumentType.INVOICE, {
        "invoice_number": "1", "invoice_date": "2026-02-01", "due_date": "2026-01-01", "vendor": {"name": "V"},
        "subtotal": 100, "tax_amount": 10, "total_amount": 150,
        "line_items": [{"quantity": 2, "unit_price": 40, "amount": 90}],
    })
    rules = _rules(validate(DocumentType.INVOICE, data))
    assert {"total_reconciliation", "line_items_sum", "line_item_math", "due_before_issue"} <= rules


def test_missing_required_fields():
    data, _ = coerce(DocumentType.RECEIPT, {})
    issues = validate(DocumentType.RECEIPT, data)
    assert {i.field for i in issues if i.rule == "required"} == {"merchant.name", "transaction_date", "total_amount"}


def test_contract_rules():
    data, _ = coerce(DocumentType.CONTRACT, {
        "title": "NDA", "parties": [{"name": "A"}], "effective_date": "2026-05-01", "expiration_date": "2026-04-01",
    })
    rules = _rules(validate(DocumentType.CONTRACT, data))
    assert {"expiry_before_effective", "party_count"} <= rules
