"""Export processed documents to CSV, Excel (multi-sheet, styled) and JSON."""

from __future__ import annotations

import io
import json
from collections.abc import Iterable
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.schemas.documents import SUMMARY_FIELDS, DocumentType
from app.storage.models import Document
from app.utils import flatten, get_by_path

SHEET_NAMES = {
    DocumentType.INVOICE.value: "Invoices",
    DocumentType.RECEIPT.value: "Receipts",
    DocumentType.CONTRACT.value: "Contracts",
    DocumentType.GENERIC.value: "Other",
}


def query_documents(
    db: Session,
    statuses: Iterable[str] | None = None,
    document_type: str | None = None,
    batch_id: str | None = None,
    ids: Iterable[str] | None = None,
) -> list[Document]:
    stmt = select(Document).order_by(Document.created_at)
    if statuses:
        stmt = stmt.where(Document.status.in_(list(statuses)))
    if document_type:
        stmt = stmt.where(Document.document_type == document_type)
    if batch_id:
        stmt = stmt.where(Document.batch_id == batch_id)
    if ids:
        stmt = stmt.where(Document.id.in_(list(ids)))
    return list(db.scalars(stmt))


def _flat_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Flatten for tabular output; line items go to their own sheet, scalar lists are joined."""
    out: dict[str, Any] = {}
    for key, value in (data or {}).items():
        if key == "line_items":
            out["line_item_count"] = len(value or [])
        elif isinstance(value, list) and all(not isinstance(v, (dict, list)) for v in value):
            out[key] = "; ".join(str(v) for v in value) if value else None
        else:
            out.update(flatten(value, key) if isinstance(value, (dict, list)) else {key: value})
    return out


def _summary_row(doc: Document) -> dict[str, Any]:
    data = doc.final_data or {}
    mapping = SUMMARY_FIELDS.get(DocumentType(doc.document_type), {}) if doc.document_type else {}
    return {
        "document_id": doc.id,
        "filename": doc.filename,
        "batch_id": doc.batch_id,
        "document_type": doc.document_type,
        "status": doc.status,
        "confidence": doc.overall_confidence,
        "reference": get_by_path(data, mapping["reference"]) if "reference" in mapping else None,
        "date": get_by_path(data, mapping["date"]) if "date" in mapping else None,
        "counterparty": get_by_path(data, mapping["counterparty"]) if "counterparty" in mapping else None,
        "total": get_by_path(data, mapping["total"]) if "total" in mapping else None,
        "currency": get_by_path(data, mapping["currency"]) if "currency" in mapping else None,
        "human_reviewed": doc.reviewed_at is not None,
        "reviewed_by": doc.reviewed_by,
        "extractor": doc.extractor,
        "pages": doc.page_count,
        "processed_at": doc.processed_at.isoformat() if doc.processed_at else None,
        "error": doc.error,
    }


def build_frames(docs: list[Document]) -> dict[str, pd.DataFrame]:
    threshold = get_settings().field_confidence_threshold
    summary, items, issues, low_conf = [], [], [], []
    per_type: dict[str, list[dict[str, Any]]] = {}

    for doc in docs:
        summary.append(_summary_row(doc))
        data = doc.final_data or {}
        if doc.document_type:
            per_type.setdefault(doc.document_type, []).append(
                {"document_id": doc.id, "filename": doc.filename, "status": doc.status,
                 "confidence": doc.overall_confidence, **_flat_fields(data)}
            )
        for i, item in enumerate(data.get("line_items") or [], start=1):
            items.append({"document_id": doc.id, "filename": doc.filename, "document_type": doc.document_type,
                          "line": i, **{k: item.get(k) for k in ("description", "quantity", "unit_price", "amount")}})
        result = doc.result or {}
        for issue in result.get("validation_issues", []):
            issues.append({"document_id": doc.id, "filename": doc.filename, **issue})
        for f in result.get("fields", []):
            if f.get("confidence", 1) < threshold and not f.get("human_verified"):
                low_conf.append({"document_id": doc.id, "filename": doc.filename, "field": f["path"],
                                 "value": json.dumps(f.get("value")) if isinstance(f.get("value"), (dict, list))
                                 else f.get("value"), "confidence": f.get("confidence")})

    frames = {"Summary": pd.DataFrame(summary)}
    for doc_type, rows in per_type.items():
        frames[SHEET_NAMES.get(doc_type, doc_type.title())] = pd.DataFrame(rows)
    frames["Line Items"] = pd.DataFrame(items, columns=[
        "document_id", "filename", "document_type", "line", "description", "quantity", "unit_price", "amount"])
    frames["Validation Issues"] = pd.DataFrame(issues, columns=[
        "document_id", "filename", "field", "rule", "severity", "message"])
    frames["Low Confidence Fields"] = pd.DataFrame(low_conf, columns=[
        "document_id", "filename", "field", "value", "confidence"])
    return frames


def to_csv(docs: list[Document], dataset: str = "documents") -> bytes:
    """``documents`` = one row per document (summary + all fields); ``line_items`` = one row per item."""
    frames = build_frames(docs)
    if dataset == "line_items":
        df = frames["Line Items"]
    else:
        rows = []
        for doc in docs:
            row = _summary_row(doc)
            row.update({f"field.{k}": v for k, v in _flat_fields(doc.final_data or {}).items()})
            rows.append(row)
        df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8-sig")  # BOM so Excel opens UTF-8 correctly


def to_json(docs: list[Document]) -> bytes:
    payload = [
        {**_summary_row(doc), "data": doc.final_data,
         "validation_issues": (doc.result or {}).get("validation_issues", [])}
        for doc in docs
    ]
    return json.dumps(payload, indent=2, default=str).encode("utf-8")


def to_excel(docs: list[Document]) -> bytes:
    from openpyxl.formatting.rule import CellIsRule
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    frames = build_frames(docs)
    buf = io.BytesIO()
    header_fill = PatternFill("solid", fgColor="1F3A5F")
    red, amber, green = (PatternFill("solid", fgColor=c) for c in ("F8D7DA", "FFF3CD", "D4EDDA"))
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in frames.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
            ws = writer.sheets[name[:31]]
            ws.freeze_panes = "A2"
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = header_fill
                cell.alignment = Alignment(vertical="center", wrap_text=True)
            for idx, col in enumerate(df.columns, start=1):
                width = max([len(str(col))] + [len(str(v)) for v in df[col].head(200).tolist()]) + 2
                ws.column_dimensions[get_column_letter(idx)].width = min(max(width, 10), 60)
                if col == "confidence" and len(df):
                    letter = get_column_letter(idx)
                    rng = f"{letter}2:{letter}{len(df) + 1}"
                    for r in range(2, len(df) + 2):
                        ws[f"{letter}{r}"].number_format = "0.0%"
                    ws.conditional_formatting.add(rng, CellIsRule(operator="lessThan", formula=["0.6"], fill=red))
                    ws.conditional_formatting.add(rng, CellIsRule(operator="between", formula=["0.6", "0.8"], fill=amber))
                    ws.conditional_formatting.add(rng, CellIsRule(operator="greaterThan", formula=["0.8"], fill=green))
            if len(df):
                ws.auto_filter.ref = ws.dimensions
    return buf.getvalue()
