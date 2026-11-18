"""Human-in-the-loop review: corrections, approval, rejection and audit trail."""

from __future__ import annotations

import copy
from typing import Any

from sqlalchemy.orm import Session

from app.processing.pipeline import rescore
from app.schemas.api import ReviewDecision, ReviewSubmission
from app.schemas.documents import DocumentType
from app.storage.models import Document, DocumentStatus, ReviewAction
from app.utils import flatten, set_by_path, utcnow


class ReviewError(ValueError):
    pass


def _diff(old: dict[str, Any], new: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    a, b = flatten(old or {}), flatten(new or {})
    changes = []
    for path in sorted(set(a) | set(b)):
        if a.get(path) != b.get(path):
            changes.append((path, a.get(path), b.get(path)))
    return changes


def submit_review(db: Session, doc: Document, payload: ReviewSubmission) -> Document:
    if doc.status in (DocumentStatus.QUEUED.value, DocumentStatus.PROCESSING.value, DocumentStatus.FAILED.value):
        raise ReviewError(f"Document is '{doc.status}' and cannot be reviewed.")
    previous = doc.final_data or {}
    new_data = copy.deepcopy(payload.data if payload.data is not None else previous)
    for path, value in payload.corrections.items():
        set_by_path(new_data, path, value)

    doc_type = DocumentType(payload.document_type or doc.document_type or "generic")
    if doc.document_type != doc_type.value:
        db.add(ReviewAction(document_id=doc.id, action="change_type", field_path="document_type",
                            old_value=doc.document_type, new_value=doc_type.value,
                            reviewer=payload.reviewer, comment=payload.comment))
        doc.document_type = doc_type.value

    result = rescore(doc.result or {}, new_data, doc.raw_text or "", set(), doc_type=doc_type)
    changes = _diff(previous, result.data)
    changed_paths = {p for p, _, _ in changes}
    verified = set(flatten(result.data)) if payload.approve else changed_paths
    result = rescore(doc.result or {}, result.data, doc.raw_text or "", verified, doc_type=doc_type)

    for path, old, new in changes:
        db.add(ReviewAction(document_id=doc.id, action="correct", field_path=path, old_value=old,
                            new_value=new, reviewer=payload.reviewer, comment=payload.comment))

    doc.corrected_data = result.data
    doc.result = result.model_dump(mode="json")
    doc.overall_confidence = result.overall_confidence
    doc.reviewed_by = payload.reviewer
    doc.reviewed_at = utcnow()
    if payload.comment:
        doc.review_notes = payload.comment
    if payload.approve:
        doc.status = DocumentStatus.APPROVED.value
        doc.needs_review = False
        db.add(ReviewAction(document_id=doc.id, action="approve", reviewer=payload.reviewer,
                            comment=payload.comment))
    else:
        doc.status = DocumentStatus.NEEDS_REVIEW.value
        doc.needs_review = True
    db.commit()
    db.refresh(doc)
    return doc


def approve(db: Session, doc: Document, decision: ReviewDecision) -> Document:
    return submit_review(db, doc, ReviewSubmission(reviewer=decision.reviewer, comment=decision.comment,
                                                   approve=True))


def reject(db: Session, doc: Document, decision: ReviewDecision) -> Document:
    if doc.status in (DocumentStatus.QUEUED.value, DocumentStatus.PROCESSING.value):
        raise ReviewError("Document is still being processed.")
    doc.status = DocumentStatus.REJECTED.value
    doc.needs_review = False
    doc.reviewed_by = decision.reviewer
    doc.reviewed_at = utcnow()
    doc.review_notes = decision.comment
    db.add(ReviewAction(document_id=doc.id, action="reject", reviewer=decision.reviewer, comment=decision.comment))
    db.commit()
    db.refresh(doc)
    return doc
