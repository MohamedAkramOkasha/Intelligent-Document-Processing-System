from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.schemas.api import BatchOut
from app.schemas.documents import DocumentType
from app.services import documents as doc_svc
from app.services.batches import batch_out, create_batch
from app.storage.database import get_db
from app.storage.models import Batch

router = APIRouter(prefix="/batches", tags=["batch processing"])


@router.post("", response_model=BatchOut, status_code=status.HTTP_202_ACCEPTED,
             summary="Upload many documents; they are processed concurrently in the background")
def create(
    files: list[UploadFile] = File(...),
    name: str | None = Form(None),
    document_type: DocumentType | None = Form(None),
    db: Session = Depends(get_db),
) -> BatchOut:
    batch = create_batch(db, name)
    rejected: list[dict[str, str]] = []
    doc_ids: list[str] = []
    for f in files:
        try:
            doc = doc_svc.create_document(db, f.file.read(), f.filename or "document", f.content_type,
                                          batch_id=batch.id, requested_type=document_type)
            doc_ids.append(doc.id)
        except doc_svc.UploadError as exc:
            rejected.append({"filename": f.filename or "?", "error": str(exc)})
    db.commit()
    if not doc_ids:
        db.delete(batch)
        db.commit()
        raise HTTPException(422, {"message": "No valid files", "rejected": rejected})
    for doc_id in doc_ids:
        doc_svc.submit(doc_id)
    db.refresh(batch)
    out = batch_out(batch, include_documents=True)
    out.rejected = rejected
    return out


@router.get("", response_model=list[BatchOut], summary="List batches")
def list_batches(limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)) -> list[BatchOut]:
    batches = db.scalars(select(Batch).order_by(Batch.created_at.desc()).limit(limit))
    return [batch_out(b) for b in batches]


@router.get("/{batch_id}", response_model=BatchOut, summary="Batch progress and documents")
def read_batch(batch_id: str, db: Session = Depends(get_db)) -> BatchOut:
    batch = db.get(Batch, batch_id)
    if batch is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Batch not found")
    return batch_out(batch, include_documents=True)
