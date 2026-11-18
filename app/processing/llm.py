"""Thin wrapper around the Google Gen AI SDK (Gemini) returning parsed JSON."""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


@dataclass
class Attachment:
    data: bytes
    mime_type: str


_GEMINI_MIME = {"application/pdf", "image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"}


def _retry_delay_s(exc: Exception) -> float | None:
    """Server-suggested retry delay of a 429 (e.g. ``retryDelay: '57711s'``), if any."""
    match = re.search(r"retryDelay'?\"?:\s*'?\"?(\d+(?:\.\d+)?)s", str(exc))
    return float(match.group(1)) if match else None


def _strip_fences(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    return match.group(1).strip() if match else text


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.8-flash",
        temperature: float = 0.0,
        max_retries: int = 3,
        timeout_s: int = 120,
    ):
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=timeout_s * 1000)
        )
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        # Circuit breaker: after a long quota block (e.g. daily limit) fail fast until it lifts,
        # so documents fall back to rules immediately instead of each one waiting on retries.
        self._blocked_until = 0.0
        self._blocked_reason = ""

    @staticmethod
    def prepare_attachment(data: bytes, mime_type: str) -> Attachment | None:
        """Convert unsupported image formats (TIFF/BMP/GIF) to PNG; skip oversized files."""
        if len(data) > 18 * 1024 * 1024:
            return None
        if mime_type in _GEMINI_MIME:
            return Attachment(data, mime_type)
        if mime_type.startswith("image/"):
            import io

            from PIL import Image

            buf = io.BytesIO()
            Image.open(io.BytesIO(data)).convert("RGB").save(buf, format="PNG")
            return Attachment(buf.getvalue(), "image/png")
        return None

    def generate_json(self, prompt: str, attachments: list[Attachment] | None = None) -> dict[str, Any]:
        if time.time() < self._blocked_until:
            raise LLMError(f"Gemini temporarily unavailable: {self._blocked_reason}")
        types = self._types
        contents: list[Any] = [
            types.Part.from_bytes(data=a.data, mime_type=a.mime_type) for a in attachments or []
        ]
        contents.append(prompt)
        config = types.GenerateContentConfig(
            temperature=self.temperature, response_mime_type="application/json"
        )
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
                text = response.text or ""
                parsed = json.loads(_strip_fences(text))
                if not isinstance(parsed, dict):
                    raise LLMError("LLM returned JSON that is not an object")
                return parsed
            except Exception as exc:  # noqa: BLE001 - network, quota, JSON errors
                last_exc = exc
                code = getattr(exc, "code", None)
                if isinstance(code, int) and 400 <= code < 500 and code not in (408, 429):
                    break  # permanent error (bad key, unknown model, ...) - retrying won't help
                delay_hint = _retry_delay_s(exc) if code == 429 else None
                if delay_hint and delay_hint > 60:
                    self._blocked_until = time.time() + delay_hint
                    self._blocked_reason = f"quota exhausted, retry in {delay_hint / 3600:.1f}h"
                    logger.error("Gemini quota exhausted; using fallback for %.0fs", delay_hint)
                    break
                if attempt < self.max_retries:
                    delay = min(2 ** attempt + random.random(), 30)
                    logger.warning("Gemini call failed (attempt %s/%s): %s - retrying in %.1fs",
                                   attempt, self.max_retries, exc, delay)
                    time.sleep(delay)
        raise LLMError(f"Gemini request failed: {last_exc}")
