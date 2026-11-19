"""Tesseract OCR wrapper with image pre-processing and word-level confidences."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

from app.config import get_settings

logger = logging.getLogger(__name__)

_WINDOWS_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

try:  # pytesseract is optional at import time so the app still runs without OCR
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None


@dataclass
class OCRWord:
    text: str
    confidence: float  # 0..1


@dataclass
class OCRResult:
    text: str
    words: list[OCRWord] = field(default_factory=list)

    @property
    def mean_confidence(self) -> float | None:
        if not self.words:
            return None
        return sum(w.confidence for w in self.words) / len(self.words)


@lru_cache
def ocr_available() -> bool:
    settings = get_settings()
    if not settings.ocr_enabled or pytesseract is None:
        return False
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    elif not shutil.which("tesseract") and Path(_WINDOWS_DEFAULT).is_file():
        # The Windows installer doesn't add Tesseract to PATH.
        pytesseract.pytesseract.tesseract_cmd = _WINDOWS_DEFAULT
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001
        logger.warning("Tesseract binary not found - OCR disabled (LLM vision can still be used).")
        return False


def preprocess(image: Image.Image) -> Image.Image:
    """Grayscale, upscale small scans, normalise contrast and lightly sharpen."""
    img = ImageOps.exif_transpose(image)
    img = img.convert("L")
    if img.width < 1400:
        scale = 1400 / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.filter(ImageFilter.SHARPEN)
    return img


def _auto_rotate(img: Image.Image) -> Image.Image:
    try:
        osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
        angle = int(osd.get("rotate", 0))
        if angle:
            return img.rotate(-angle, expand=True, fillcolor=255)
    except Exception:  # noqa: BLE001 - OSD needs osd.traineddata and enough text
        pass
    return img


def run_ocr(image: Image.Image, languages: str | None = None) -> OCRResult:
    if not ocr_available():
        return OCRResult(text="")
    languages = languages or get_settings().ocr_languages
    img = _auto_rotate(preprocess(image))
    data = pytesseract.image_to_data(
        img, lang=languages, config="--oem 3 --psm 3", output_type=pytesseract.Output.DICT
    )
    lines: dict[tuple[int, int, int], list[str]] = {}
    words: list[OCRWord] = []
    for i, raw in enumerate(data["text"]):
        token = (raw or "").strip()
        conf = float(data["conf"][i])
        if not token or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(token)
        words.append(OCRWord(text=token, confidence=max(0.0, min(conf / 100.0, 1.0))))
    text_lines: list[str] = []
    last_block = None
    for (block, _par, _line), tokens in lines.items():
        if last_block is not None and block != last_block:
            text_lines.append("")
        text_lines.append(" ".join(tokens))
        last_block = block
    return OCRResult(text="\n".join(text_lines), words=words)
