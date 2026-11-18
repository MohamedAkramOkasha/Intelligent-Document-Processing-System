"""Turn uploaded files (PDF, images, DOCX, text) into text + page images.

Digital PDFs use their embedded text layer; pages with little or no text are rasterised
and sent through OCR. Images are always OCR'd. Rendered page previews are kept so the
review UI can show the original next to the extracted values.
"""

from __future__ import annotations

import io
import logging
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

from app.config import get_settings
from app.processing.ocr import OCRWord, ocr_available, run_ocr

logger = logging.getLogger(__name__)

PDF_EXT = {".pdf"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
DOCX_EXT = {".docx"}
TEXT_EXT = {".txt", ".md", ".text", ".csv", ".json", ".xml", ".html", ".htm"}
SUPPORTED_EXTENSIONS = PDF_EXT | IMAGE_EXT | DOCX_EXT | TEXT_EXT


class UnsupportedDocumentError(ValueError):
    pass


@dataclass
class PageContent:
    number: int
    text: str
    ocr_used: bool = False
    ocr_confidence: float | None = None


@dataclass
class ParsedDocument:
    filename: str
    mime_type: str
    raw_bytes: bytes
    pages: list[PageContent] = field(default_factory=list)
    ocr_words: list[OCRWord] = field(default_factory=list)
    preview_images: list[Image.Image] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        if len(self.pages) == 1:
            return self.pages[0].text
        return "\n\n".join(f"--- Page {p.number} ---\n{p.text}" for p in self.pages)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def ocr_used(self) -> bool:
        return any(p.ocr_used for p in self.pages)

    @property
    def ocr_confidence(self) -> float | None:
        if not self.ocr_words:
            return None
        return sum(w.confidence for w in self.ocr_words) / len(self.ocr_words)

    @property
    def is_visual(self) -> bool:
        """True when the source is a PDF or image (worth sending to a multimodal LLM)."""
        return self.mime_type == "application/pdf" or self.mime_type.startswith("image/")


def guess_mime(filename: str, content_type: str | None = None) -> str:
    if content_type and content_type not in {"application/octet-stream", ""}:
        return content_type
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


def parse_document(data: bytes, filename: str, content_type: str | None = None) -> ParsedDocument:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    doc = ParsedDocument(filename=filename, mime_type=guess_mime(filename, content_type), raw_bytes=data)
    if ext in PDF_EXT:
        doc.mime_type = "application/pdf"
        _parse_pdf(doc)
    elif ext in IMAGE_EXT:
        _parse_image(doc)
    elif ext in DOCX_EXT:
        _parse_docx(doc)
    else:
        doc.pages = [PageContent(number=1, text=_decode(data))]
    if not doc.text.strip():
        doc.warnings.append("No text could be extracted from the document.")
    return doc


# --------------------------------------------------------------------------- formats
def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _ocr_page(doc: ParsedDocument, number: int, image: Image.Image) -> PageContent:
    result = run_ocr(image)
    doc.ocr_words.extend(result.words)
    return PageContent(number=number, text=result.text, ocr_used=True, ocr_confidence=result.mean_confidence)


def _parse_pdf(doc: ParsedDocument) -> None:
    import pypdfium2 as pdfium

    settings = get_settings()
    try:
        pdf = pdfium.PdfDocument(doc.raw_bytes)
    except pdfium.PdfiumError as exc:
        raise UnsupportedDocumentError(f"Could not open PDF: {exc}") from exc
    try:
        n_pages = len(pdf)
        if n_pages > settings.max_pages:
            doc.warnings.append(f"PDF has {n_pages} pages; only the first {settings.max_pages} were processed.")
        warned_no_ocr = False
        for i in range(min(n_pages, settings.max_pages)):
            page = pdf[i]
            textpage = page.get_textpage()
            text = textpage.get_text_range().replace("\r\n", "\n").replace("\r", "\n")
            textpage.close()
            if i < settings.max_preview_pages:
                doc.preview_images.append(page.render(scale=1.5).to_pil().convert("RGB"))
            if len(text.strip()) < settings.min_text_chars_per_page:
                if ocr_available():
                    image = page.render(scale=settings.ocr_dpi / 72).to_pil()
                    content = _ocr_page(doc, i + 1, image)
                    if len(content.text.strip()) > len(text.strip()):
                        doc.pages.append(content)
                        page.close()
                        continue
                elif not warned_no_ocr:
                    doc.warnings.append("Scanned PDF pages detected but Tesseract is not available.")
                    warned_no_ocr = True
            doc.pages.append(PageContent(number=i + 1, text=text))
            page.close()
    finally:
        pdf.close()


def _parse_image(doc: ParsedDocument) -> None:
    settings = get_settings()
    try:
        img = Image.open(io.BytesIO(doc.raw_bytes))
    except Exception as exc:  # noqa: BLE001
        raise UnsupportedDocumentError(f"Could not open image: {exc}") from exc
    frames = getattr(img, "n_frames", 1)
    if not ocr_available():
        doc.warnings.append("Tesseract is not available; image text relies on the LLM vision model.")
    for i in range(min(frames, settings.max_pages)):
        img.seek(i)
        frame = ImageOps.exif_transpose(img.copy()).convert("RGB")
        if i < settings.max_preview_pages:
            preview = frame.copy()
            preview.thumbnail((1600, 2200))
            doc.preview_images.append(preview)
        doc.pages.append(_ocr_page(doc, i + 1, frame) if ocr_available() else PageContent(i + 1, ""))


def _parse_docx(doc: ParsedDocument) -> None:
    import docx

    try:
        document = docx.Document(io.BytesIO(doc.raw_bytes))
    except Exception as exc:  # noqa: BLE001
        raise UnsupportedDocumentError(f"Could not open DOCX: {exc}") from exc
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        parts.append("")
        for row in table.rows:
            cells: list[str] = []
            for cell in row.cells:
                value = cell.text.strip()
                if not cells or cells[-1] != value:  # merged cells repeat their text
                    cells.append(value)
            parts.append(" | ".join(cells))
    doc.pages = [PageContent(number=1, text="\n".join(parts).strip())]
