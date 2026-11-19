"""Document lifecycle: storage, background processing, reprocessing, deletion."""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import get_settings
from app.processing.parsers import SUPPORTED_EXTENSIONS, UnsupportedDocumentError
from app.processing.pipeline import get_pipeline
from app.schemas.documents import DocumentType
from app.storage.database import session_scope
from app.storage.models import Document, DocumentStatus, ReviewAction
from app.utils import utcnow

logger = logging.getLogger(__name__)
settings = get_settings()

_executor: ThreadPoolExecutor | None = None


def executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=settings.batch_concurrency, thread_name_prefix="idp-worker")
    return _executor


def shutdown_executor() -> None:
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None


class UploadError(ValueError):
    pass


def safe_filename(name: str) -> str:
    name = Path(name or "document").name
    return re.sub(r"[^\w.\- ()]", "_", name)[:200] or "document"


def create_document(
    db: Session,
    data: bytes,
    filename: str,
    content_type: str | None = None,
    batch_id: str | None = None,
    requested_type: DocumentType | None = None,
) -> Document:
    filename = safe_filename(filename)
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UploadError(f"Unsupported file type '{ext or '(none)'}'.")
    if not data:
        raise UploadError(f"'{filename}' is empty.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise UploadError(f"'{filename}' exceeds the {settings.max_upload_mb} MB limit.")

    doc = Document(
        filename=filename, content_type=content_type, file_size=len(data),
        file_hash=hashlib.sha256(data).hexdigest(), batch_id=batch_id,
        requested_type=requested_type.value if requested_type else None,
        status=DocumentStatus.QUEUED.value, file_path="",
    )
    db.add(doc)
    db.flush()  # assigns id
    path = settings.upload_dir / f"{doc.id}{ext}"
    path.write_bytes(data)
    doc.file_path = str(path)
    db.commit()
    return doc


def process_document(doc_id: str) -> None:
    """Run the pipeline for one stored document and persist the result. Thread-safe."""
    with session_scope() as db:
        doc = db.get(Document, doc_id)
        if doc is None:
            return
        doc.status = DocumentStatus.PROCESSING.value
        doc.error = None
        path, filename, content_type = Path(doc.file_path), doc.filename, doc.content_type
        requested = DocumentType(doc.requested_type) if doc.requested_type else None

    try:
        output = get_pipeline().process(path.read_bytes(), filename, content_type, requested)
    except UnsupportedDocumentError as exc:
        _mark_failed(doc_id, str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("Processing failed for %s", doc_id)
        _mark_failed(doc_id, f"{type(exc).__name__}: {exc}")
        return

    preview_dir = settings.preview_dir / doc_id
    shutil.rmtree(preview_dir, ignore_errors=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(output.parsed.preview_images, start=1):
        img.save(preview_dir / f"page_{i}.png", format="PNG", optimize=True)

    result = output.result
    with session_scope() as db:
        doc = db.get(Document, doc_id)
        if doc is None:  # deleted while the pipeline ran
            shutil.rmtree(preview_dir, ignore_errors=True)
            return
        doc.document_type = result.document_type.value
        doc.extractor = result.extractor
        doc.overall_confidence = result.overall_confidence
        doc.needs_review = result.needs_review
        doc.page_count = result.page_count
        doc.preview_count = len(output.parsed.preview_images)
        doc.extracted_data = result.data
        doc.corrected_data = None
        doc.result = result.model_dump(mode="json")
        doc.raw_text = output.parsed.text
        doc.processed_at = utcnow()
        doc.status = (DocumentStatus.NEEDS_REVIEW if result.needs_review else DocumentStatus.AUTO_APPROVED).value
    logger.info("Processed %s (%s) -> %s, confidence %.2f", filename, result.document_type.value,
                "needs_review" if result.needs_review else "auto_approved", result.overall_confidence)


def _mark_failed(doc_id: str, error: str) -> None:
    with session_scope() as db:
        doc = db.get(Document, doc_id)
        if doc:
            doc.status = DocumentStatus.FAILED.value
            doc.error = error[:2000]
            doc.processed_at = utcnow()


def submit(doc_id: str) -> None:
    executor().submit(process_document, doc_id)


def reprocess(db: Session, doc: Document, document_type: DocumentType | None, reviewer: str | None) -> None:
    doc.status = DocumentStatus.QUEUED.value
    if document_type:
        doc.requested_type = document_type.value
    db.add(ReviewAction(document_id=doc.id, action="reprocess", reviewer=reviewer,
                        new_value=document_type.value if document_type else None))
    db.commit()


def delete_document(db: Session, doc: Document) -> None:
    Path(doc.file_path).unlink(missing_ok=True)
    shutil.rmtree(settings.preview_dir / doc.id, ignore_errors=True)
    db.delete(doc)
    db.commit()


def preview_path(doc: Document, page: int) -> Path:
    return settings.preview_dir / doc.id / f"page_{page}.png"
