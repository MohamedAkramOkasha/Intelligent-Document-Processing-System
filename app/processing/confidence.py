"""Confidence scoring.

Each extracted field gets a score in ``[0, 1]`` blended from independent signals:

* **LLM self-assessment** - the model's own per-field confidence (calibrated by the prompt rubric).
* **Evidence grounding** - is the value actually present in the document text? (fuzzy match for
  strings, format-aware match for numbers and dates). Catches hallucinations.
* **OCR quality** - Tesseract word confidences of the tokens supporting the value (scanned docs only).

Business-rule violations then penalise the fields they touch. The document score combines the
weighted mean of all fields with the weakest *required* field, so one bad total can't hide
behind twenty good line-item descriptions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from rapidfuzz import fuzz

from app.processing.heuristics import DATE
from app.processing.ocr import OCRWord
from app.schemas.documents import REQUIRED_FIELDS, DocumentType
from app.schemas.extraction import FieldConfidence, ValidationIssue
from app.utils import flatten, parse_date, path_ancestors

WEIGHTS = {"llm": 0.45, "evidence": 0.35, "ocr": 0.20}
ERROR_PENALTY = 0.5
WARNING_PENALTY = 0.85
DEFAULT_LLM_CONFIDENCE = 0.7
_WORD = re.compile(r"[\w@.\-/]+", re.U)


@dataclass
class ScoringContext:
    text: str
    ocr_words: list[OCRWord]
    ocr_confidence: float | None

    def __post_init__(self) -> None:
        self.norm_text = re.sub(r"\s+", " ", self.text.lower())
        self.compact_text = re.sub(r"[\s,]", "", self.norm_text)
        self.dates: set[date] = set()
        for m in re.finditer(DATE, self.text):
            try:
                self.dates.add(parse_date(m.group(1)))
            except ValueError:
                continue
        self.word_conf: dict[str, list[float]] = {}
        for w in self.ocr_words:
            self.word_conf.setdefault(_norm_token(w.text), []).append(w.confidence)


def _norm_token(s: str) -> str:
    return s.lower().strip(".,:;()[]{}\"'$€£")


# --------------------------------------------------------------------------- signals
def evidence_score(value: Any, ctx: ScoringContext) -> float | None:
    if value is None or isinstance(value, bool) or not ctx.norm_text.strip():
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        variants = {f"{v:.2f}", f"{v:g}", f"{v:,.2f}".replace(",", ""), f"{v:.2f}".replace(".", ",")}
        if v.is_integer():
            variants.add(str(int(v)))
        return 1.0 if any(var in ctx.compact_text for var in variants) else 0.25
    s = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        try:
            return 1.0 if date.fromisoformat(s) in ctx.dates else 0.3
        except ValueError:
            return 0.0
    if not s:
        return None
    return fuzz.partial_ratio(s.lower(), ctx.norm_text) / 100.0


def ocr_score(value: Any, evidence: str | None, ctx: ScoringContext) -> float | None:
    if not ctx.ocr_words or value is None:
        return None
    source = evidence or (f"{value:,.2f}" if isinstance(value, float) else str(value))
    confs = [c for tok in _WORD.findall(source) for c in ctx.word_conf.get(_norm_token(tok), [])]
    if confs:
        return sum(confs) / len(confs)
    return (ctx.ocr_confidence or 0.5) * 0.9


def _lookup(mapping: dict[str, Any], path: str) -> Any:
    for key in path_ancestors(path):
        if key in mapping:
            return mapping[key]
    return None


def _issue_touches(issue: ValidationIssue, path: str) -> bool:
    f = issue.field
    return bool(f) and (path == f or path.startswith(f + ".") or path.startswith(f + "["))


# --------------------------------------------------------------------------- scoring
def score_fields(
    doc_type: DocumentType,
    data: dict[str, Any],
    llm_confidence: dict[str, float],
    evidence: dict[str, str],
    issues: list[ValidationIssue],
    ctx: ScoringContext,
    field_threshold: float,
    verified_paths: set[str] | None = None,
) -> list[FieldConfidence]:
    verified_paths = verified_paths or set()
    required = REQUIRED_FIELDS[doc_type]
    flat = {p: v for p, v in flatten(data).items() if v not in (None, "", [])}
    fields: list[FieldConfidence] = []

    for path, value in flat.items():
        is_required = any(path == r or path.startswith(r + ".") or path.startswith(r + "[") for r in required)
        if path in verified_paths:
            fields.append(FieldConfidence(path=path, value=value, confidence=1.0, required=is_required,
                                          human_verified=True))
            continue
        ev_text = _lookup(evidence, path)
        llm = _lookup(llm_confidence, path)
        llm = DEFAULT_LLM_CONFIDENCE if llm is None else float(llm)
        evid = evidence_score(value, ctx)
        ocr = ocr_score(value, ev_text if isinstance(ev_text, str) else None, ctx)
        parts = {"llm": llm, "evidence": evid, "ocr": ocr}
        weight = sum(WEIGHTS[k] for k, v in parts.items() if v is not None)
        score = sum(WEIGHTS[k] * v for k, v in parts.items() if v is not None) / weight
        notes: list[str] = []
        for issue in issues:
            if _issue_touches(issue, path):
                score *= ERROR_PENALTY if issue.severity == "error" else WARNING_PENALTY
                notes.append(issue.message)
        if evid is not None and evid < 0.5:
            notes.append("Value not found verbatim in document text")
        score = round(max(0.0, min(score, 1.0)), 3)
        fields.append(FieldConfidence(
            path=path, value=value, confidence=score, llm_confidence=round(llm, 3),
            evidence_score=None if evid is None else round(evid, 3),
            ocr_confidence=None if ocr is None else round(ocr, 3),
            evidence=ev_text if isinstance(ev_text, str) else None, required=is_required,
            needs_review=score < field_threshold, issues=notes,
        ))

    for req in required:
        if not any(f.path == req or f.path.startswith(req + ".") or f.path.startswith(req + "[") for f in fields):
            fields.append(FieldConfidence(path=req, value=None, confidence=0.0, required=True, needs_review=True,
                                          issues=["Required field is missing"]))
    return fields


def document_confidence(fields: list[FieldConfidence], classification_confidence: float) -> float:
    if not fields:
        return 0.0
    weights = [3.0 if f.required else 1.0 for f in fields]
    mean = sum(w * f.confidence for w, f in zip(weights, fields)) / sum(weights)
    required = [f.confidence for f in fields if f.required]
    score = 0.6 * mean + 0.4 * min(required) if required else mean
    # an uncertain document type caps how confident we can be in the rest
    score *= min(1.0, 0.5 + classification_confidence / 2 + 0.05)
    return round(max(0.0, min(score, 1.0)), 3)


def review_reasons(
    fields: list[FieldConfidence],
    issues: list[ValidationIssue],
    overall: float,
    classification_confidence: float,
    *,
    doc_threshold: float,
    field_threshold: float,
    class_threshold: float,
) -> list[str]:
    reasons: list[str] = []
    if overall < doc_threshold:
        reasons.append(f"Overall confidence {overall:.0%} is below the {doc_threshold:.0%} threshold")
    if classification_confidence < class_threshold:
        reasons.append(f"Document type is uncertain ({classification_confidence:.0%})")
    low = sorted({f.path for f in fields if f.required and f.confidence < field_threshold})
    if low:
        reasons.append("Low-confidence required fields: " + ", ".join(low))
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        reasons.append(f"{len(errors)} validation error(s): " + "; ".join(i.message for i in errors[:3]))
    return reasons
