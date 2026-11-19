"""LLM clients (Gemini SDK or any OpenAI-compatible API) returning parsed JSON."""

from __future__ import annotations

import base64
import io
import json
import logging
import random
import re
import threading
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
    if match:
        return float(match.group(1))
    # Groq / OpenAI style: "Please try again in 6.72s" / "in 1h2m3.5s" / "in 450ms"
    match = re.search(r"try again in (\d+(?:\.\d+)?ms|(?:\d+h)?(?:\d+m(?!s))?(?:\d+(?:\.\d+)?s)?)\b", str(exc))
    if not match or not match.group(1):
        return None
    text = match.group(1)
    if text.endswith("ms"):
        return float(text[:-2]) / 1000
    units = {"h": 3600, "m": 60, "s": 1}
    total = sum(float(n) * units[u] for n, u in re.findall(r"(\d+(?:\.\d+)?)([hms])", text))
    return total or None


_UNAVAILABLE_COOLDOWN_S = 60


def _is_timeout(exc: Exception) -> bool:
    """requests / httpx / google-genai timeouts all carry "Timeout" in the class name or message."""
    return any("timeout" in t.__name__.lower() for t in type(exc).__mro__) or "timed out" in str(exc).lower()


def _strip_fences(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    return match.group(1).strip() if match else text


class BaseLLMClient:
    """Shared JSON parsing, retries, rate-limit pausing and the daily-quota circuit breaker.

    One instance is shared by all worker threads, so its state acts on the whole queue:

    * ``max_concurrent`` caps simultaneous requests (0 = unlimited).
    * A short rate limit (429 with a retry delay <= 60s) pauses *every* request until the
      limit resets. Those waits do not use up ``max_retries``; a request gives up only after
      waiting ``rate_limit_max_wait_s`` in total, so the caller can switch to a fallback.
    * A long block (daily quota) opens a circuit breaker: calls fail fast until it lifts.
    """

    provider = "llm"
    sees_files = False  # whether attachments (PDF/images) reach the model

    def __init__(self, model: str, temperature: float = 0.0, max_retries: int = 3, timeout_s: int = 120,
                 max_concurrent: int = 0, rate_limit_max_wait_s: float = 90.0):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.rate_limit_max_wait_s = rate_limit_max_wait_s
        self._slots = threading.BoundedSemaphore(max_concurrent) if max_concurrent > 0 else None
        self._lock = threading.Lock()
        self._paused_until = 0.0
        self._blocked_until = 0.0
        self._blocked_reason = ""

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def last_label(self) -> str:
        return self.label

    def _pause(self, seconds: float) -> None:
        with self._lock:
            self._paused_until = max(self._paused_until, time.time() + seconds)

    def _call(self, prompt: str, attachments: list[Attachment]) -> str:
        if self._slots is None:
            return self._request(prompt, attachments)
        with self._slots:
            return self._request(prompt, attachments)

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
        attempt = rate_limits = 0
        rate_waited = 0.0
        while True:
            pause = self._paused_until - time.time()
            if pause > 0:
                if rate_waited + pause > self.rate_limit_max_wait_s:
                    raise LLMError(f"{name} rate limited (waited {rate_waited:.0f}s): {last_exc}")
                time.sleep(pause)
                rate_waited += pause
            try:
                parsed = json.loads(_strip_fences(self._call(prompt, list(attachments or [])) or ""))
                if not isinstance(parsed, dict):
                    raise LLMError("LLM returned JSON that is not an object")
                return parsed
            except Exception as exc:  # noqa: BLE001 - network, quota, JSON errors
                last_exc = exc
                code = getattr(exc, "code", None)
                if isinstance(code, int) and 400 <= code < 500 and code not in (408, 429):
                    break  # permanent error (bad key, unknown model, ...) - retrying won't help
                if code == 503 or _is_timeout(exc):
                    # Overloaded or hanging: retrying (each try can take the full timeout) only
                    # delays the fallback. Skip this provider briefly so queued calls fail fast too.
                    with self._lock:
                        self._blocked_until = max(self._blocked_until, time.time() + _UNAVAILABLE_COOLDOWN_S)
                    self._blocked_reason = "overloaded or not responding"
                    logger.warning("%s unavailable (%s); skipping it for %ds", name, str(exc)[:120],
                                   _UNAVAILABLE_COOLDOWN_S)
                    break
                if code == 429:
                    if "request too large" in str(exc).lower():
                        break  # one request exceeds the per-minute limit - it can never succeed
                    delay_hint = _retry_delay_s(exc)
                    if delay_hint and delay_hint > 60:
                        with self._lock:
                            self._blocked_until = time.time() + delay_hint
                        when = (f"{delay_hint / 3600:.1f}h" if delay_hint >= 3600
                                else f"{delay_hint / 60:.0f}min")
                        self._blocked_reason = f"quota exhausted, retry in {when}"
                        logger.error("%s quota exhausted; using fallback for %.0fs", name, delay_hint)
                        break
                    # Per-minute limit: pause the whole queue, don't count it as a retry.
                    rate_limits += 1
                    delay = (delay_hint or min(2 ** rate_limits, 30)) + random.uniform(0.2, 1.0)
                    self._pause(delay)
                    logger.info("%s rate limited - pausing requests for %.1fs", name, delay)
                    continue
                attempt += 1
                if attempt >= self.max_retries:
                    break
                delay = min(2 ** attempt + random.random(), 30)
                logger.warning("%s call failed (attempt %s/%s): %s - retrying in %.1fs",
                               name, attempt, self.max_retries, exc, delay)
                time.sleep(delay)
        raise LLMError(f"{name} request failed: {last_exc}")


class GeminiClient(BaseLLMClient):
    provider = "gemini"
    sees_files = True

    def __init__(self, api_key: str, model: str = "gemini-3.8-flash", temperature: float = 0.0,
                 max_retries: int = 3, timeout_s: int = 120, **limits: Any):
        super().__init__(model, temperature, max_retries, timeout_s, **limits)
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
        **limits: Any,
    ):
        super().__init__(model, temperature, max_retries, timeout_s, **limits)
        import requests

        self.provider = provider
        self.vision = self.sees_files = vision
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


class FallbackClient:
    """Tries each client in order; the first one that answers wins.

    A client that is rate limited past its wait budget, over its daily quota or failing
    raises ``LLMError`` and the next one is tried. ``last_label`` (per thread) says which
    client produced the most recent answer.
    """

    def __init__(self, clients: list[BaseLLMClient]):
        if not clients:
            raise ValueError("FallbackClient needs at least one client")
        self.clients = clients
        self._local = threading.local()

    @property
    def provider(self) -> str:
        return self.clients[0].provider

    @property
    def model(self) -> str:
        return self.clients[0].model

    @property
    def label(self) -> str:
        return " -> ".join(c.label for c in self.clients)

    @property
    def sees_files(self) -> bool:
        return any(c.sees_files for c in self.clients)

    @property
    def last_label(self) -> str:
        return getattr(self._local, "label", self.clients[0].label)

    def generate_json(self, prompt: str, attachments: list[Attachment] | None = None,
                      files_only: bool = False) -> dict[str, Any]:
        """``files_only``: the content is only in the attachments (no text), so skip clients
        that can't see files - a text-only model would answer, but with nothing to go on."""
        clients = [c for c in self.clients if c.sees_files] if files_only else []
        clients = clients or self.clients
        errors: list[str] = []
        for i, client in enumerate(clients):
            try:
                out = client.generate_json(prompt, attachments)
            except LLMError as exc:
                errors.append(str(exc))
                if i + 1 < len(clients):
                    logger.warning("%s failed (%s) - falling back to %s",
                                   client.label, str(exc)[:200], clients[i + 1].label)
                continue
            self._local.label = client.label
            return out
        raise LLMError(" | ".join(errors))


def make_provider_client(settings, provider: str) -> BaseLLMClient:
    """Build the client for one provider from settings."""
    common: dict[str, Any] = dict(
        temperature=settings.llm_temperature,
        max_retries=settings.llm_max_retries,
        timeout_s=settings.llm_timeout_s,
        max_concurrent=settings.concurrency_for(provider),
        rate_limit_max_wait_s=settings.llm_rate_limit_max_wait_s,
    )
    if provider == "gemini":
        return GeminiClient(api_key=settings.gemini_api_key or "", model=settings.gemini_model, **common)
    return OpenAICompatibleClient(
        base_url=settings.base_url_for(provider),
        model=settings.model_for(provider),
        api_key=settings.llm_api_key,
        provider=provider,
        vision=settings.llm_vision,
        **common,
    )


def make_client(settings) -> BaseLLMClient | FallbackClient:
    """Client for ``LLM_PROVIDER``, wrapped with ``LLM_FALLBACK_PROVIDER`` when configured."""
    providers = settings.llm_providers or [settings.llm_provider]
    clients = [make_provider_client(settings, p) for p in providers]
    return clients[0] if len(clients) == 1 else FallbackClient(clients)
