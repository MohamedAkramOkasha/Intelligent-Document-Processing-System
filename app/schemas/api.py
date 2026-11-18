"""Request/response models for the REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.extraction import ExtractionResult


class DocumentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    batch_id: str | None = None
    filename: str
    status: str
    document_type: str | None = None
    overall_confidence: float | None = None
    needs_review: bool = False
    page_count: int = 0
    extractor: str | None = None
    error: str | None = None
    created_at: datetime
    processed_at: datetime | None = None
    reviewed_at: datetime | None = None
    reviewed_by: str | None = None


class DocumentDetail(DocumentSummary):
    content_type: str | None = None
    file_size: int = 0
    preview_count: int = 0
    extracted_data: dict[str, Any] | None = None
    corrected_data: dict[str, Any] | None = None
    final_data: dict[str, Any] | None = None
    result: ExtractionResult | None = None
    review_notes: str | None = None
    raw_text: str | None = None


class DocumentList(BaseModel):
    items: list[DocumentSummary]
    total: int
    limit: int
    offset: int


class BatchOut(BaseModel):
    id: str
    name: str | None = None
    created_at: datetime
    total: int
    completed: int
    status: str
    progress: float
    counts: dict[str, int]
    documents: list[DocumentSummary] = Field(default_factory=list)
    rejected: list[dict[str, str]] = Field(default_factory=list, description="Files that were not accepted")


class ReviewSubmission(BaseModel):
    """Corrections from a human reviewer.

    Either send ``data`` (full corrected document) or ``corrections`` (``{"field.path": value}``),
    or both (``corrections`` are applied on top of ``data``).
    """

    data: dict[str, Any] | None = None
    corrections: dict[str, Any] = Field(default_factory=dict)
    document_type: str | None = Field(None, description="Change the document type")
    reviewer: str = "reviewer"
    comment: str | None = None
    approve: bool = True


class ReviewDecision(BaseModel):
    reviewer: str = "reviewer"
    comment: str | None = None


class ReviewActionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    action: str
    field_path: str | None = None
    old_value: Any = None
    new_value: Any = None
    reviewer: str | None = None
    comment: str | None = None
    created_at: datetime


class StatsOut(BaseModel):
    total_documents: int
    by_status: dict[str, int]
    by_type: dict[str, int]
    average_confidence: float | None
    review_queue: int
    auto_approval_rate: float | None
    total_batches: int
    llm_enabled: bool
    ocr_enabled: bool
    model: str | None
