"""ORM models: documents, batches and the human-review audit trail."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.storage.database import Base
from app.utils import utcnow


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return utcnow()


class DocumentStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    NEEDS_REVIEW = "needs_review"
    AUTO_APPROVED = "auto_approved"
    APPROVED = "approved"
    REJECTED = "rejected"
    FAILED = "failed"


FINAL_STATUSES = {DocumentStatus.AUTO_APPROVED, DocumentStatus.APPROVED,
                  DocumentStatus.REJECTED, DocumentStatus.FAILED, DocumentStatus.NEEDS_REVIEW}
APPROVED_STATUSES = {DocumentStatus.AUTO_APPROVED, DocumentStatus.APPROVED}


class Batch(Base):
    __tablename__ = "batches"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(), default=_now)

    documents: Mapped[list[Document]] = relationship(back_populates="batch")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    batch_id: Mapped[str | None] = mapped_column(ForeignKey("batches.id", ondelete="SET NULL"), index=True)
    filename: Mapped[str] = mapped_column(String(512))
    content_type: Mapped[str | None] = mapped_column(String(128))
    file_path: Mapped[str] = mapped_column(String(1024))
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    file_hash: Mapped[str] = mapped_column(String(64), index=True)

    status: Mapped[str] = mapped_column(String(32), default=DocumentStatus.QUEUED.value, index=True)
    requested_type: Mapped[str | None] = mapped_column(String(32))
    document_type: Mapped[str | None] = mapped_column(String(32), index=True)
    extractor: Mapped[str | None] = mapped_column(String(128))
    overall_confidence: Mapped[float | None] = mapped_column(Float)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    preview_count: Mapped[int] = mapped_column(Integer, default=0)

    extracted_data: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    corrected_data: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    raw_text: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime(), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(), default=_now, onupdate=_now)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime())
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime())
    reviewed_by: Mapped[str | None] = mapped_column(String(128))
    review_notes: Mapped[str | None] = mapped_column(Text)

    batch: Mapped[Batch | None] = relationship(back_populates="documents")
    review_actions: Mapped[list[ReviewAction]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="ReviewAction.created_at"
    )

    @property
    def final_data(self) -> dict[str, Any] | None:
        return self.corrected_data if self.corrected_data is not None else self.extracted_data


class ReviewAction(Base):
    __tablename__ = "review_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    action: Mapped[str] = mapped_column(String(32))  # correct | approve | reject | reprocess
    field_path: Mapped[str | None] = mapped_column(String(255))
    old_value: Mapped[Any | None] = mapped_column(JSON)
    new_value: Mapped[Any | None] = mapped_column(JSON)
    reviewer: Mapped[str | None] = mapped_column(String(128))
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(), default=_now)

    document: Mapped[Document] = relationship(back_populates="review_actions")
