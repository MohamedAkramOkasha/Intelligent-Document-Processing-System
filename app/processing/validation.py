"""Schema coercion and business-rule validation of extracted data."""

from __future__ import annotations

import copy
import re
from datetime import date, timedelta
from typing import Any

from pydantic import ValidationError

from app.schemas.documents import REQUIRED_FIELDS, DocumentType, model_for
from app.schemas.extraction import ValidationIssue
from app.utils import get_by_path, set_by_path, to_jsonable

EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")


def coerce(doc_type: DocumentType, data: dict[str, Any]) -> tuple[dict[str, Any], list[ValidationIssue]]:
    """Validate ``data`` against the Pydantic model for ``doc_type``.

    Values that cannot be coerced (e.g. an unparseable date) are nulled out one by one and
    reported as issues instead of failing the whole document.
    """
    model = model_for(doc_type)
    working = copy.deepcopy(data) if isinstance(data, dict) else {}
    issues: list[ValidationIssue] = []
    for _ in range(25):
        try:
            obj = model.model_validate(working)
            return to_jsonable(obj.model_dump(mode="json")), issues
        except ValidationError as exc:
            for err in exc.errors():
                path = "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in err["loc"]).lstrip(".")
                if not path:
                    working = {}
                    continue
                bad = get_by_path(working, path)
                top = path.split(".")[0].split("[")[0]
                if bad is None:  # already nulled and still invalid -> fall back to the default
                    working.pop(top, None)
                    continue
                issues.append(ValidationIssue(
                    field=path, rule="schema", severity="error",
                    message=f"Invalid value {bad!r}: {err['msg']}",
                ))
                try:
                    set_by_path(working, path, None)
                except (TypeError, ValueError, IndexError):
                    working.pop(top, None)
    obj = model.model_validate({})
    return to_jsonable(obj.model_dump(mode="json")), issues


# --------------------------------------------------------------------------- rules
def _close(a: float, b: float, rel: float = 0.005, abs_tol: float = 0.02) -> bool:
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


def _d(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def validate(doc_type: DocumentType, data: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for path in REQUIRED_FIELDS[doc_type]:
        if _is_missing(get_by_path(data, path)):
            issues.append(ValidationIssue(field=path, rule="required", message=f"Required field '{path}' is missing"))

    if doc_type in (DocumentType.INVOICE, DocumentType.RECEIPT):
        issues += _validate_amounts(doc_type, data)
    if doc_type == DocumentType.INVOICE:
        issues += _validate_invoice(data)
    elif doc_type == DocumentType.RECEIPT:
        tx = _d(data.get("transaction_date"))
        if tx and tx > date.today() + timedelta(days=1):
            issues.append(ValidationIssue(field="transaction_date", rule="future_date", severity="warning",
                                          message="Transaction date is in the future"))
    elif doc_type == DocumentType.CONTRACT:
        issues += _validate_contract(data)

    currency = data.get("currency")
    if currency and not re.fullmatch(r"[A-Z]{3}", str(currency)):
        issues.append(ValidationIssue(field="currency", rule="currency_code", severity="warning",
                                      message=f"Currency '{currency}' is not an ISO 4217 code"))
    for party_path in ("vendor", "customer", "merchant"):
        email = get_by_path(data, f"{party_path}.email")
        if email and not EMAIL_RE.match(str(email)):
            issues.append(ValidationIssue(field=f"{party_path}.email", rule="email_format", severity="warning",
                                          message=f"'{email}' is not a valid email address"))
    return issues


def _validate_amounts(doc_type: DocumentType, data: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    items = data.get("line_items") or []
    for i, item in enumerate(items):
        q, u, a = item.get("quantity"), item.get("unit_price"), item.get("amount")
        if q is not None and u is not None and a is not None and not _close(q * u, a):
            issues.append(ValidationIssue(
                field=f"line_items[{i}].amount", rule="line_item_math", severity="warning",
                message=f"Line {i + 1}: {q} x {u} = {q * u:.2f}, but amount is {a:.2f}"))

    subtotal, total = data.get("subtotal"), data.get("total_amount")
    tax = data.get("tax_amount") or 0.0
    extras = (data.get("shipping") or 0.0) + (data.get("tip") or 0.0) - abs(data.get("discount") or 0.0)
    item_sum = sum(it.get("amount") or 0.0 for it in items) if items and all(
        it.get("amount") is not None for it in items) else None

    if item_sum is not None and subtotal is not None and not _close(item_sum, subtotal):
        issues.append(ValidationIssue(
            field="subtotal", rule="line_items_sum", severity="error",
            message=f"Line items sum to {item_sum:.2f} but subtotal is {subtotal:.2f}"))
    base = subtotal if subtotal is not None else item_sum
    if base is not None and total is not None:
        expected = base + tax + extras
        if not _close(expected, total):
            # Some documents show tax-inclusive prices - accept total == base too.
            if not _close(base + extras, total):
                issues.append(ValidationIssue(
                    field="total_amount", rule="total_reconciliation", severity="error",
                    message=f"Subtotal {base:.2f} + tax {tax:.2f} + adjustments {extras:.2f} = "
                            f"{expected:.2f}, but total is {total:.2f}"))
    if total is not None and total < 0:
        issues.append(ValidationIssue(field="total_amount", rule="negative_total", severity="warning",
                                      message="Total amount is negative (credit note?)"))
    rate = data.get("tax_rate")
    if rate is not None:
        if not 0 <= rate <= 100:
            issues.append(ValidationIssue(field="tax_rate", rule="tax_rate_range", severity="warning",
                                          message=f"Tax rate {rate} is outside 0-100%"))
        elif subtotal and data.get("tax_amount") is not None:
            taxable = subtotal - abs(data.get("discount") or 0.0)
            if not _close(taxable * rate / 100, data["tax_amount"], rel=0.02, abs_tol=0.05) and not _close(
                    subtotal * rate / 100, data["tax_amount"], rel=0.02, abs_tol=0.05):
                issues.append(ValidationIssue(
                    field="tax_amount", rule="tax_rate_math", severity="warning",
                    message=f"Tax {data['tax_amount']:.2f} does not match {rate}% of {taxable:.2f}"))
    return issues


def _validate_invoice(data: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    inv, due = _d(data.get("invoice_date")), _d(data.get("due_date"))
    if inv and due and due < inv:
        issues.append(ValidationIssue(field="due_date", rule="due_before_issue",
                                      message=f"Due date {due} is before invoice date {inv}"))
    if inv and inv > date.today() + timedelta(days=1):
        issues.append(ValidationIssue(field="invoice_date", rule="future_date", severity="warning",
                                      message="Invoice date is in the future"))
    if inv and inv.year < 1990:
        issues.append(ValidationIssue(field="invoice_date", rule="implausible_date", severity="warning",
                                      message=f"Invoice date {inv} looks implausible"))
    amount_due, total = data.get("amount_due"), data.get("total_amount")
    if amount_due is not None and total is not None and amount_due > total + 0.01:
        issues.append(ValidationIssue(field="amount_due", rule="amount_due_exceeds_total", severity="warning",
                                      message="Amount due exceeds invoice total"))
    return issues


def _validate_contract(data: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    eff, exp = _d(data.get("effective_date")), _d(data.get("expiration_date"))
    if eff and exp and exp <= eff:
        issues.append(ValidationIssue(field="expiration_date", rule="expiry_before_effective",
                                      message=f"Expiration date {exp} is not after effective date {eff}"))
    parties = data.get("parties") or []
    if 0 < len(parties) < 2:
        issues.append(ValidationIssue(field="parties", rule="party_count", severity="warning",
                                      message="A contract normally has at least two parties"))
    for i, p in enumerate(parties):
        if not (p or {}).get("name"):
            issues.append(ValidationIssue(field=f"parties[{i}].name", rule="required",
                                          message=f"Party {i + 1} has no name"))
    value = data.get("contract_value")
    if value is not None and value < 0:
        issues.append(ValidationIssue(field="contract_value", rule="negative_value", severity="warning",
                                      message="Contract value is negative"))
    return issues
