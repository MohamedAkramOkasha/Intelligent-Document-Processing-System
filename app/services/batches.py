"""Batch processing: many files submitted together, processed concurrently."""

from __future__ import annotations

from collections import Counter

from sqlalchemy.orm import Session

from app.schemas.api import BatchOut, DocumentSummary
from app.storage.models import Batch, DocumentStatus

_PENDING = {DocumentStatus.QUEUED.value, DocumentStatus.PROCESSING.value}


def create_batch(db: Session, name: str | None) -> Batch:
    batch = Batch(name=name)
    db.add(batch)
    db.flush()
    return batch


def batch_out(batch: Batch, include_documents: bool = False) -> BatchOut:
    docs = sorted(batch.documents, key=lambda d: d.created_at)
    counts = Counter(d.status for d in docs)
    total = len(docs)
    completed = sum(n for s, n in counts.items() if s not in _PENDING)
    if total == 0:
        status = "empty"
    elif completed < total:
        status = "processing"
    elif counts.get(DocumentStatus.FAILED.value) == total:
        status = "failed"
    elif counts.get(DocumentStatus.NEEDS_REVIEW.value):
        status = "needs_review"
    else:
        status = "completed"
    return BatchOut(
        id=batch.id, name=batch.name, created_at=batch.created_at, total=total, completed=completed,
        status=status, progress=round(completed / total, 3) if total else 1.0, counts=dict(counts),
        documents=[DocumentSummary.model_validate(d) for d in docs] if include_documents else [],
    )
