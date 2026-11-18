"""End-to-end processing pipeline.

    file bytes -> parse (text layer / OCR) -> classify -> extract (Gemini or heuristic)
               -> schema coercion (Pydantic) -> business-rule validation
               -> confidence scoring -> review routing
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from app.config import Settings, get_settings
from app.processing import confidence as conf
from app.processing.extractors import Extractor, GeminiExtractor, HeuristicExtractor
from app.processing.parsers import ParsedDocument, parse_document
from app.processing.validation import coerce, validate
from app.schemas.documents import DocumentType
from app.schemas.extraction import ExtractionResult, RawExtraction

logger = logging.getLogger(__name__)


@dataclass
class PipelineOutput:
    result: ExtractionResult
    parsed: ParsedDocument


class DocumentPipeline:
    def __init__(self, settings: Settings | None = None, extractor: Extractor | None = None):
        self.settings = settings or get_settings()
        self.fallback = HeuristicExtractor()
        if extractor is not None:
            self.extractor: Extractor = extractor
        elif self.settings.llm_available:
            self.extractor = GeminiExtractor(self.settings)
        else:
            logger.warning("No GEMINI_API_KEY configured - using the heuristic extractor.")
            self.extractor = self.fallback

    # ------------------------------------------------------------------ public API
    def process(
        self,
        data: bytes,
        filename: str,
        content_type: str | None = None,
        document_type: DocumentType | None = None,
    ) -> PipelineOutput:
        started = time.perf_counter()
        parsed = parse_document(data, filename, content_type)
        warnings = list(parsed.warnings)
        extractor = self.extractor
        if not parsed.text.strip() and not isinstance(extractor, GeminiExtractor):
            warnings.append("Nothing to extract: no text and no vision model available.")

        # 1. classification
        if document_type is not None:
            doc_type, class_conf = DocumentType(document_type), 1.0
        else:
            try:
                doc_type, class_conf = extractor.classify(parsed)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Classification failed with %s", extractor.name)
                warnings.append(f"LLM classification failed ({exc}); used keyword classifier.")
                doc_type, class_conf = self.fallback.classify(parsed)

        # 2. extraction (with automatic fallback)
        extractor_name = extractor.name
        try:
            raw = extractor.extract(parsed, doc_type)
        except Exception as exc:  # noqa: BLE001
            if extractor is self.fallback:
                raise
            logger.exception("Extraction failed with %s", extractor.name)
            warnings.append(f"LLM extraction failed ({exc}); used rule-based fallback.")
            raw = self.fallback.extract(parsed, doc_type)
            extractor_name = f"{self.fallback.name} (fallback)"

        result = self.finalize(
            doc_type=doc_type, classification_confidence=class_conf, raw=raw, parsed=parsed,
            extractor_name=extractor_name, warnings=warnings,
        )
        result.processing_time_ms = int((time.perf_counter() - started) * 1000)
        return PipelineOutput(result=result, parsed=parsed)

    def finalize(
        self,
        *,
        doc_type: DocumentType,
        classification_confidence: float,
        raw: RawExtraction,
        parsed: ParsedDocument | None = None,
        text: str | None = None,
        extractor_name: str = "unknown",
        warnings: list[str] | None = None,
        verified_paths: set[str] | None = None,
        ocr_confidence: float | None = None,
    ) -> ExtractionResult:
        """Coerce, validate and score extracted data. Also used to re-score human corrections."""
        s = self.settings
        data, schema_issues = coerce(doc_type, raw.data)
        issues = schema_issues + validate(doc_type, data)
        ctx = conf.ScoringContext(
            text=parsed.text if parsed else (text or ""),
            ocr_words=parsed.ocr_words if parsed else [],
            ocr_confidence=parsed.ocr_confidence if parsed else ocr_confidence,
        )
        fields = conf.score_fields(
            doc_type, data, raw.field_confidence, raw.evidence, issues, ctx,
            s.field_confidence_threshold, verified_paths,
        )
        overall = conf.document_confidence(fields, classification_confidence)
        reasons = conf.review_reasons(
            fields, issues, overall, classification_confidence,
            doc_threshold=s.confidence_threshold, field_threshold=s.field_confidence_threshold,
            class_threshold=s.classification_threshold,
        )
        if extractor_name.startswith("heuristic") and s.llm_available:
            reasons.append("Extracted by rule-based fallback instead of the LLM")
        return ExtractionResult(
            document_type=doc_type,
            classification_confidence=round(classification_confidence, 3),
            data=data,
            fields=fields,
            validation_issues=issues,
            overall_confidence=overall,
            needs_review=bool(reasons),
            review_reasons=reasons,
            extractor=extractor_name,
            page_count=parsed.page_count if parsed else 0,
            ocr_used=parsed.ocr_used if parsed else False,
            ocr_confidence=round(ctx.ocr_confidence, 3) if ctx.ocr_confidence is not None else None,
            warnings=warnings or [],
        )


_pipeline: DocumentPipeline | None = None


def get_pipeline() -> DocumentPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DocumentPipeline()
    return _pipeline


def set_pipeline(pipeline: DocumentPipeline | None) -> None:
    """Override the global pipeline (used by tests)."""
    global _pipeline
    _pipeline = pipeline


def rescore(
    previous: dict[str, Any],
    data: dict[str, Any],
    text: str,
    verified_paths: set[str],
    doc_type: DocumentType | None = None,
) -> ExtractionResult:
    """Re-run validation/scoring after a reviewer edits the data."""
    if not previous:
        previous = {"document_type": (doc_type or DocumentType.GENERIC).value,
                    "classification_confidence": 1.0, "data": {}}
    prev = ExtractionResult.model_validate(previous)
    doc_type = doc_type or prev.document_type
    raw = RawExtraction(
        data=data,
        field_confidence={f.path: f.llm_confidence for f in prev.fields if f.llm_confidence is not None},
        evidence={f.path: f.evidence for f in prev.fields if f.evidence},
    )
    result = get_pipeline().finalize(
        doc_type=doc_type, classification_confidence=1.0, raw=raw, text=text,
        extractor_name=prev.extractor, warnings=prev.warnings, verified_paths=verified_paths,
        ocr_confidence=prev.ocr_confidence,
    )
    result.page_count, result.ocr_used = prev.page_count, prev.ocr_used
    result.processing_time_ms = prev.processing_time_ms
    return result
