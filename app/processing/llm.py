"""LLM clients (Gemini SDK or any OpenAI-compatible API) returning parsed JSON."""

from __future__ import annotations

import base64
import io
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


class APIError(Exception):
    """HTTP error from an OpenAI-compatible endpoint (mirrors google-genai's ``.code``)."""

    def __init__(self, code: int, message: str, retry_after: float | None = None):
        super().__init__(f"{code} {message}")
        self.code = code
        self.retry_after = retry_after


@dataclass
class Attachment:
    data: bytes
    mime_type: str


_GEMINI_MIME = {"application/pdf", "image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"}


def _retry_delay_s(exc: Exception) -> float | None:
    """Server-suggested retry delay of a 429 (e.g. ``retryDelay: '57711s'``), if any."""
    hint = getattr(exc, "retry_after", None)
    if hint:
        return float(hint)
    match = re.search(r"retryDelay'?\"?:\s*'?\"?(\d+(?:\.\d+)?)s", str(exc))
    return float(match.group(1)) if match else None


def _strip_fences(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    return match.group(1).strip() if match else text


class BaseLLMClient:
    """Shared JSON parsing, retries with backoff and the daily-quota circuit breaker."""

    provider = "llm"

    def __init__(self, model: str, temperature: float = 0.0, max_retries: int = 3, timeout_s: int = 120):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout_s = timeout_s
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
            from PIL import Image

            buf = io.BytesIO()
            Image.open(io.BytesIO(data)).convert("RGB").save(buf, format="PNG")
            return Attachment(buf.getvalue(), "image/png")
        return None

    def _request(self, prompt: str, attachments: list[Attachment]) -> str:
        raise NotImplementedError

    def generate_json(self, prompt: str, attachments: list[Attachment] | None = None) -> dict[str, Any]:
        name = self.provider.capitalize()
        if time.time() < self._blocked_until:
            raise LLMError(f"{name} temporarily unavailable: {self._blocked_reason}")
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                parsed = json.loads(_strip_fences(self._request(prompt, list(attachments or [])) or ""))
                if not isinstance(parsed, dict):
                    raise LLMError("LLM returned JSON that is not an object")
                return parsed
            except Exception as exc:  # noqa: BLE001 - network, quota, JSON errors
                last_exc = exc
                code = getattr(exc, "code", None)
                if isinstance(code, int) and 400 <= code < 500 and code not in (408, 429):
                    break  # permanent error (bad key, unknown model, ...) - retrying won't help
                if code == 429 and "request too large" in str(exc).lower():
                    break  # a single request exceeds the per-minute limit - it can never succeed
                delay_hint = _retry_delay_s(exc) if code == 429 else None
                if delay_hint and delay_hint > 60:
                    self._blocked_until = time.time() + delay_hint
                    when = (f"{delay_hint / 3600:.1f}h" if delay_hint >= 3600
                            else f"{delay_hint / 60:.0f}min")
                    self._blocked_reason = f"quota exhausted, retry in {when}"
                    logger.error("%s quota exhausted; using fallback for %.0fs", name, delay_hint)
                    break
                if attempt < self.max_retries:
                    delay = delay_hint or min(2 ** attempt + random.random(), 30)
                    logger.warning("%s call failed (attempt %s/%s): %s - retrying in %.1fs",
                                   name, attempt, self.max_retries, exc, delay)
                    time.sleep(delay)
        raise LLMError(f"{name} request failed: {last_exc}")


class GeminiClient(BaseLLMClient):
    provider = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.8-flash",
        temperature: float = 0.0,
        max_retries: int = 3,
        timeout_s: int = 120,
    ):
        super().__init__(model, temperature, max_retries, timeout_s)
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=timeout_s * 1000)
        )

    def _request(self, prompt: str, attachments: list[Attachment]) -> str:
        types = self._types
        contents: list[Any] = [
            types.Part.from_bytes(data=a.data, mime_type=a.mime_type) for a in attachments
        ]
        contents.append(prompt)
        config = types.GenerateContentConfig(
            temperature=self.temperature, response_mime_type="application/json"
        )
        response = self.client.models.generate_content(model=self.model, contents=contents, config=config)
        return response.text or ""


class OpenAICompatibleClient(BaseLLMClient):
    """Chat Completions client for Groq, OpenRouter, Ollama, LM Studio, OpenAI, ...

    Uses JSON mode (``response_format={"type": "json_object"}``). Images are sent as data URLs
    only when ``vision`` is enabled; PDFs are never attached (their text / OCR is in the prompt).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        provider: str = "openai",
        vision: bool = False,
        temperature: float = 0.0,
        max_retries: int = 3,
        timeout_s: int = 120,
    ):
        super().__init__(model, temperature, max_retries, timeout_s)
        import requests

        self.provider = provider
        self.vision = vision
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def _content(self, prompt: str, attachments: list[Attachment]) -> str | list[dict[str, Any]]:
        images = [a for a in attachments if self.vision and a.mime_type.startswith("image/")]
        if not images:
            return prompt
        parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for a in images:
            url = f"data:{a.mime_type};base64,{base64.b64encode(a.data).decode()}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        return parts

    def _request(self, prompt: str, attachments: list[Attachment]) -> str:
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "You are a precise document data extraction engine. "
                                              "Always answer with a single valid JSON object."},
                {"role": "user", "content": self._content(prompt, attachments)},
            ],
        }
        resp = self.session.post(self.url, json=body, timeout=self.timeout_s)
        if resp.status_code >= 400:
            try:
                retry_after = float(resp.headers.get("retry-after", ""))
            except ValueError:
                retry_after = None
            raise APIError(resp.status_code, resp.text[:1000], retry_after)
        return resp.json()["choices"][0]["message"].get("content") or ""


def make_client(settings) -> BaseLLMClient:
    """Build the LLM client selected by ``LLM_PROVIDER``."""
    common = dict(
        temperature=settings.llm_temperature,
        max_retries=settings.llm_max_retries,
        timeout_s=settings.llm_timeout_s,
    )
    if settings.llm_provider == "gemini":
        return GeminiClient(api_key=settings.gemini_api_key or "", model=settings.gemini_model, **common)
    return OpenAICompatibleClient(
        base_url=settings.active_base_url,
        model=settings.active_model,
        api_key=settings.llm_api_key,
        provider=settings.llm_provider,
        vision=settings.llm_vision,
        **common,
    )
