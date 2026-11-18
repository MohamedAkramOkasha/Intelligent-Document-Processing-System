from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.processing.ocr import ocr_available
from app.schemas.api import StatsOut
from app.storage.database import get_db
from app.storage.models import Batch, Document, DocumentStatus

router = APIRouter(tags=["system"])


@router.get("/health", summary="Liveness / configuration check")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "version": s.app_version,
        "llm": {
            "enabled": s.llm_available,
            "provider": s.llm_providers[0] if s.llm_available else None,
            "model": s.model_for(s.llm_providers[0]) if s.llm_available else None,
            "chain": s.llm_label or None,  # e.g. "groq:openai/gpt-oss-120b -> gemini:gemini-3.8-flash"
        },
        "ocr": {"enabled": ocr_available()},
    }


@router.get("/stats", response_model=StatsOut, summary="Processing statistics")
def stats(db: Session = Depends(get_db)) -> StatsOut:
    s = get_settings()
    by_status = dict(db.execute(select(Document.status, func.count()).group_by(Document.status)).all())
    by_type = dict(db.execute(
        select(Document.document_type, func.count()).where(Document.document_type.is_not(None))
        .group_by(Document.document_type)).all())
    avg = db.scalar(select(func.avg(Document.overall_confidence)))
    auto = by_status.get(DocumentStatus.AUTO_APPROVED.value, 0)
    decided = auto + sum(by_status.get(x.value, 0) for x in (
        DocumentStatus.NEEDS_REVIEW, DocumentStatus.APPROVED, DocumentStatus.REJECTED))
    return StatsOut(
        total_documents=sum(by_status.values()),
        by_status=by_status,
        by_type=by_type,
        average_confidence=round(avg, 3) if avg is not None else None,
        review_queue=by_status.get(DocumentStatus.NEEDS_REVIEW.value, 0),
        auto_approval_rate=round(auto / decided, 3) if decided else None,
        total_batches=db.scalar(select(func.count(Batch.id))) or 0,
        llm_enabled=s.llm_available,
        ocr_enabled=ocr_available(),
        model=s.llm_label if s.llm_available else None,
    )
