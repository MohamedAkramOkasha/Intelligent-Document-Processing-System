"""Extractor implementations: LLM (Gemini or OpenAI-compatible) and heuristic (rule-based)."""

from __future__ import annotations

import json
import logging
from typing import Protocol

from pydantic import ValidationError

from app.config import Settings
from app.processing import heuristics
from app.processing.llm import Attachment, BaseLLMClient, make_client
from app.processing.parsers import ParsedDocument
from app.processing.prompts import CLASSIFY_PROMPT, REPAIR_PROMPT, build_extract_prompt
from app.schemas.documents import DocumentType, model_for
from app.schemas.extraction import RawExtraction

logger = logging.getLogger(__name__)


class Extractor(Protocol):
    name: str

    def classify(self, doc: ParsedDocument) -> tuple[DocumentType, float]: ...

    def extract(self, doc: ParsedDocument, doc_type: DocumentType) -> RawExtraction: ...


class HeuristicExtractor:
    name = "heuristic"

    def classify(self, doc: ParsedDocument) -> tuple[DocumentType, float]:
        return heuristics.classify(doc.text)

    def extract(self, doc: ParsedDocument, doc_type: DocumentType) -> RawExtraction:
        return heuristics.extract(doc.text, doc_type)


class LLMExtractor:
    def __init__(self, settings: Settings, client: BaseLLMClient | None = None):
        self.settings = settings
        self.client = client or make_client(settings)
        self.name = f"{settings.llm_provider}:{settings.active_model}"
        # Whether the model gets to see the original file (PDF/image), not just the text.
        self.sees_files = settings.llm_send_files and (
            settings.llm_provider == "gemini" or settings.llm_vision
        )

    def _attachments(self, doc: ParsedDocument) -> list[Attachment]:
        if not (self.sees_files and doc.is_visual):
            return []
        att = BaseLLMClient.prepare_attachment(doc.raw_bytes, doc.mime_type)
        return [att] if att else []

    def _text(self, doc: ParsedDocument, limit: int) -> str:
        text = doc.text
        return text if len(text) <= limit else text[:limit] + "\n...[truncated]"

    def classify(self, doc: ParsedDocument) -> tuple[DocumentType, float]:
        prompt = CLASSIFY_PROMPT.format(text=self._text(doc, 6000))
        # Only attach the file when there is too little text to decide on.
        attachments = self._attachments(doc) if len(doc.text.strip()) < 200 else []
        out = self.client.generate_json(prompt, attachments)
        try:
            doc_type = DocumentType(str(out.get("document_type", "generic")).lower())
        except ValueError:
            doc_type = DocumentType.GENERIC
        return doc_type, float(out.get("confidence", 0.5))

    def extract(self, doc: ParsedDocument, doc_type: DocumentType) -> RawExtraction:
        prompt = build_extract_prompt(doc_type, self._text(doc, self.settings.llm_max_text_chars))
        attachments = self._attachments(doc)
        out = self.client.generate_json(prompt, attachments)
        model = model_for(doc_type)
        try:
            model.model_validate(out.get("data") or {})
        except ValidationError as exc:
            # One self-repair round: show the model its validation errors.
            logger.info("Extraction failed schema validation, asking the LLM to repair: %s", exc.error_count())
            errors = "\n".join(f"- {'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            repair = REPAIR_PROMPT.format(errors=errors, previous=json.dumps(out)[:20000])
            try:
                out = self.client.generate_json(prompt + "\n\n" + repair, attachments)
            except Exception:  # noqa: BLE001 - keep the first answer; coercion handles the rest
                logger.warning("Repair round failed; continuing with original output")
        return RawExtraction(
            data=out.get("data") or {},
            field_confidence=_floats(out.get("field_confidence")),
            evidence={k: str(v) for k, v in (out.get("evidence") or {}).items() if v},
        )


GeminiExtractor = LLMExtractor  # backwards-compatible name


def _floats(value) -> dict[str, float]:
    result: dict[str, float] = {}
    for k, v in (value or {}).items():
        try:
            result[k] = max(0.0, min(float(v), 1.0))
        except (TypeError, ValueError):
            continue
    return result
