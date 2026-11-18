from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_document
from app.schemas.api import DocumentDetail, DocumentList, DocumentSummary
from app.schemas.documents import DocumentType
from app.services import documents as svc
from app.storage.database import get_db
from app.storage.models import Document

router = APIRouter(prefix="/documents", tags=["documents"])


def to_detail(doc: Document, include_text: bool = False) -> DocumentDetail:
    detail = DocumentDetail.model_validate(doc)
    if not include_text:
        detail.raw_text = None
    return detail


@router.post("", response_model=DocumentDetail, status_code=status.HTTP_201_CREATED,
             summary="Upload and process a single document")
def upload_document(
    file: UploadFile = File(..., description="PDF, image (PNG/JPG/TIFF), DOCX or TXT"),
    document_type: DocumentType | None = Form(None, description="Skip classification and force a type"),
    wait: bool = Query(True, description="Process synchronously; false = queue and return immediately"),
    db: Session = Depends(get_db),
) -> DocumentDetail:
    try:
        doc = svc.create_document(db, file.file.read(), file.filename or "document", file.content_type,
                                  requested_type=document_type)
    except svc.UploadError as exc:
        raise HTTPException(422, str(exc)) from exc
    if wait:
        svc.process_document(doc.id)
    else:
        svc.submit(doc.id)
    db.refresh(doc)
    return to_detail(doc)


@router.get("", response_model=DocumentList, summary="List documents")
def list_documents(
    status_: list[str] | None = Query(None, alias="status"),
    document_type: DocumentType | None = None,
    batch_id: str | None = None,
    q: str | None = Query(None, description="Filename contains"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> DocumentList:
    stmt = select(Document)
    if status_:
        stmt = stmt.where(Document.status.in_(status_))
    if document_type:
        stmt = stmt.where(Document.document_type == document_type.value)
    if batch_id:
        stmt = stmt.where(Document.batch_id == batch_id)
    if q:
        stmt = stmt.where(Document.filename.ilike(f"%{q}%"))
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.scalars(stmt.order_by(Document.created_at.desc()).limit(limit).offset(offset))
    return DocumentList(items=[DocumentSummary.model_validate(d) for d in rows], total=total,
                        limit=limit, offset=offset)


@router.get("/{document_id}", response_model=DocumentDetail, summary="Get a document with its extraction")
def read_document(include_text: bool = False, doc: Document = Depends(get_document)) -> DocumentDetail:
    return to_detail(doc, include_text)


@router.get("/{document_id}/file", summary="Download the original file")
def download_file(doc: Document = Depends(get_document)) -> FileResponse:
    path = Path(doc.file_path)
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Original file is missing")
    return FileResponse(path, filename=doc.filename, media_type=doc.content_type or "application/octet-stream")


@router.get("/{document_id}/pages/{page}", summary="Rendered page preview (PNG)")
def page_preview(page: int, doc: Document = Depends(get_document)) -> FileResponse:
    path = svc.preview_path(doc, page)
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No preview for this page")
    return FileResponse(path, media_type="image/png")


@router.post("/{document_id}/reprocess", response_model=DocumentDetail, summary="Re-run extraction")
def reprocess_document(
    document_type: DocumentType | None = Query(None),
    wait: bool = True,
    reviewer: str | None = None,
    doc: Document = Depends(get_document),
    db: Session = Depends(get_db),
) -> DocumentDetail:
    doc = db.merge(doc)
    svc.reprocess(db, doc, document_type, reviewer)
    if wait:
        svc.process_document(doc.id)
    else:
        svc.submit(doc.id)
    db.refresh(doc)
    return to_detail(doc)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a document")
def delete_document(doc: Document = Depends(get_document), db: Session = Depends(get_db)) -> None:
    svc.delete_document(db, db.merge(doc))
