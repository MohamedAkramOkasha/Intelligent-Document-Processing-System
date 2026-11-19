from __future__ import annotations

import hmac

from fastapi import Depends, HTTPException, Security, UploadFile, status
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from app.config import get_settings
from app.storage.database import get_db
from app.storage.models import Document

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str | None = Security(_api_key_header)) -> None:
    expected = get_settings().api_key
    if expected and not hmac.compare_digest((key or "").encode(), expected.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Key header")


def read_upload(file: UploadFile) -> bytes:
    """Read an upload, stopping one byte past MAX_UPLOAD_MB so oversized files are never fully loaded.

    ``create_document`` rejects anything longer than the limit with the usual error message.
    """
    return file.file.read(get_settings().max_upload_mb * 1024 * 1024 + 1)


def get_document(document_id: str, db: Session = Depends(get_db)) -> Document:
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Document '{document_id}' not found")
    return doc
