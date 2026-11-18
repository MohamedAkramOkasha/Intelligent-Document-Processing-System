from app.config import get_settings
from app.processing.confidence import ScoringContext, evidence_score
from app.processing.extractors import GeminiExtractor
from app.processing.llm import LLMError
from app.processing.pipeline import DocumentPipeline
from app.schemas.documents import DocumentType
from tests.conftest import FakeGemini


def _pipeline(fake: FakeGemini) -> DocumentPipeline:
    return DocumentPipeline(extractor=GeminiExtractor(get_settings(), client=fake))


def test_evidence_score_detects_hallucinations(invoice_text):
    ctx = ScoringContext(text=invoice_text, ocr_words=[], ocr_confidence=None)
    assert evidence_score(550.0, ctx) == 1.0
    assert evidence_score("2026-03-03", ctx) == 1.0  # printed as "March 3, 2026"
    assert evidence_score("Sunrise Office Supplies", ctx) > 0.95
    assert evidence_score(9999.99, ctx) < 0.5
    assert evidence_score("Totally Made Up Corp", ctx) < 0.7


def test_heuristic_pipeline_extracts_invoice(invoice_text):
    out = DocumentPipeline().process(invoice_text.encode(), "inv.txt")
    r = out.result
    assert r.document_type == DocumentType.INVOICE
    assert r.data["invoice_number"] == "SOS-10593"
    assert r.data["total_amount"] == 550.0
    assert len(r.data["line_items"]) == 2
    assert r.extractor == "heuristic"
    assert r.needs_review  # modest heuristic confidence on a text doc stays below the threshold


def test_gemini_pipeline_auto_approves_confident_extraction(invoice_text, gemini_invoice_response):
    fake = FakeGemini([{"document_type": "invoice", "confidence": 0.98}, gemini_invoice_response])
    r = _pipeline(fake).process(invoice_text.encode(), "inv.txt").result
    assert r.document_type == DocumentType.INVOICE
    assert r.data["currency"] == "USD" and r.data["total_amount"] == 550.0
    assert not r.validation_issues
    assert r.overall_confidence >= 0.85
    assert not r.needs_review, r.review_reasons
    assert len(fake.calls) == 2


def test_gemini_hallucinated_total_triggers_review(invoice_text, gemini_invoice_response):
    gemini_invoice_response["data"]["total_amount"] = 5500
    fake = FakeGemini([gemini_invoice_response])
    r = _pipeline(fake).process(invoice_text.encode(), "inv.txt", document_type=DocumentType.INVOICE).result
    total = r.field_map()["total_amount"]
    assert total.needs_review and total.evidence_score < 0.5
    assert r.needs_review
    assert any(i.rule == "total_reconciliation" for i in r.validation_issues)


def test_schema_repair_round(invoice_text, gemini_invoice_response):
    bad = {"data": {"invoice_date": "sometime", "line_items": "oops"}}
    fake = FakeGemini([bad, gemini_invoice_response])
    r = _pipeline(fake).process(invoice_text.encode(), "inv.txt", document_type=DocumentType.INVOICE).result
    assert "Invoice validation failed" not in str(r.warnings)
    assert r.data["invoice_date"] == "2026-03-03"
    assert "failed validation" in fake.calls[1][0]


def test_falls_back_to_heuristics_when_llm_fails(invoice_text):
    fake = FakeGemini([LLMError("quota exceeded"), LLMError("quota exceeded")])
    r = _pipeline(fake).process(invoice_text.encode(), "inv.txt").result
    assert r.extractor == "heuristic (fallback)"
    assert r.data["invoice_number"] == "SOS-10593"
    assert any("fallback" in w or "keyword" in w for w in r.warnings)
    assert r.needs_review
