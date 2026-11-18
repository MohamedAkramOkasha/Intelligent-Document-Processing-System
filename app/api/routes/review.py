from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_document
from app.api.routes.documents import to_detail
from app.schemas.api import DocumentDetail, DocumentSummary, ReviewActionOut, ReviewDecision, ReviewSubmission
from app.services import review as svc
from app.storage.database import get_db
from app.storage.models import Document, DocumentStatus

router = APIRouter(prefix="/review", tags=["human review"])


@router.get("/queue", response_model=list[DocumentSummary], summary="Documents awaiting human review")
def review_queue(
    document_type: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
) -> list[DocumentSummary]:
    stmt = select(Document).where(Document.status == DocumentStatus.NEEDS_REVIEW.value)
    if document_type:
        stmt = stmt.where(Document.document_type == document_type)
    # least confident first: that's where a reviewer adds the most value
    stmt = stmt.order_by(Document.overall_confidence.asc(), Document.created_at).limit(limit)
    return [DocumentSummary.model_validate(d) for d in db.scalars(stmt)]


def _run(fn, *args) -> DocumentDetail:
    try:
        return to_detail(fn(*args))
    except svc.ReviewError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


@router.post("/{document_id}", response_model=DocumentDetail, summary="Submit corrections (and optionally approve)")
def submit_review(payload: ReviewSubmission, doc: Document = Depends(get_document),
                  db: Session = Depends(get_db)) -> DocumentDetail:
    return _run(svc.submit_review, db, doc, payload)


@router.post("/{document_id}/approve", response_model=DocumentDetail, summary="Approve as-is")
def approve(decision: ReviewDecision, doc: Document = Depends(get_document),
            db: Session = Depends(get_db)) -> DocumentDetail:
    return _run(svc.approve, db, doc, decision)


@router.post("/{document_id}/reject", response_model=DocumentDetail, summary="Reject the document")
def reject(decision: ReviewDecision, doc: Document = Depends(get_document),
           db: Session = Depends(get_db)) -> DocumentDetail:
    return _run(svc.reject, db, doc, decision)


@router.get("/{document_id}/history", response_model=list[ReviewActionOut], summary="Audit trail")
def history(doc: Document = Depends(get_document)) -> list[ReviewActionOut]:
    return [ReviewActionOut.model_validate(a) for a in doc.review_actions]
