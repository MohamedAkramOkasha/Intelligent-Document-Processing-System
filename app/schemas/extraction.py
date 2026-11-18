"""Models describing the output of the extraction pipeline."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.documents import DocumentType


class RawExtraction(BaseModel):
    """What an extractor (LLM or heuristic) returns before validation & scoring."""

    data: dict[str, Any] = Field(default_factory=dict)
    field_confidence: dict[str, float] = Field(default_factory=dict)
    evidence: dict[str, str] = Field(default_factory=dict)


class ValidationIssue(BaseModel):
    field: str | None = None
    rule: str
    message: str
    severity: Literal["error", "warning"] = "error"


class FieldConfidence(BaseModel):
    path: str
    value: Any = None
    confidence: float = Field(ge=0, le=1)
    llm_confidence: float | None = None
    evidence_score: float | None = None
    ocr_confidence: float | None = None
    evidence: str | None = None
    required: bool = False
    needs_review: bool = False
    issues: list[str] = Field(default_factory=list)
    human_verified: bool = False


class ExtractionResult(BaseModel):
    document_type: DocumentType
    classification_confidence: float
    data: dict[str, Any]
    fields: list[FieldConfidence] = Field(default_factory=list)
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    overall_confidence: float = 0.0
    needs_review: bool = True
    review_reasons: list[str] = Field(default_factory=list)
    extractor: str = "unknown"
    page_count: int = 0
    ocr_used: bool = False
    ocr_confidence: float | None = None
    processing_time_ms: int = 0
    warnings: list[str] = Field(default_factory=list)

    def field_map(self) -> dict[str, FieldConfidence]:
        return {f.path: f for f in self.fields}
