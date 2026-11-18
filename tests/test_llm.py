from types import SimpleNamespace

import pytest

from app.processing.llm import GeminiClient, LLMError


class FakeAPIError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(f"{code} {message}")
        self.code = code


def _client(side_effects: list) -> tuple[GeminiClient, list]:
    client = GeminiClient(api_key="test-key", max_retries=3)
    calls: list = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        item = side_effects.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(text=item)

    client.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return client, calls


def test_parses_json_and_strips_fences():
    client, _ = _client(['```json\n{"a": 1}\n```'])
    assert client.generate_json("p") == {"a": 1}


def test_permanent_error_is_not_retried():
    client, calls = _client([FakeAPIError(404, "model not found")])
    with pytest.raises(LLMError, match="model not found"):
        client.generate_json("p")
    assert len(calls) == 1


def test_daily_quota_opens_circuit_breaker():
    quota = FakeAPIError(429, "RESOURCE_EXHAUSTED {'retryDelay': '57711s'}")
    client, calls = _client([quota])
    with pytest.raises(LLMError):
        client.generate_json("p")
    with pytest.raises(LLMError, match="temporarily unavailable"):
        client.generate_json("p")  # fails fast, no API call
    assert len(calls) == 1


def test_transient_error_is_retried(monkeypatch):
    monkeypatch.setattr("app.processing.llm.time.sleep", lambda _: None)
    client, calls = _client([FakeAPIError(503, "unavailable"), '{"ok": true}'])
    assert client.generate_json("p") == {"ok": True}
    assert len(calls) == 2


# ---------------------------------------------------------------- OpenAI-compatible (Groq, ...)
from app.config import Settings  # noqa: E402
from app.processing.llm import Attachment, OpenAICompatibleClient, make_client  # noqa: E402


def _http_client(responses: list, vision: bool = False) -> tuple[OpenAICompatibleClient, list]:
    client = OpenAICompatibleClient(base_url="https://api.example/v1/", model="m", api_key="k",
                                    provider="groq", vision=vision, max_retries=3)
    calls: list = []

    def post(url, json=None, timeout=None):
        calls.append((url, json))
        status, body, headers = responses.pop(0)
        return SimpleNamespace(status_code=status, text=str(body), headers=headers,
                               json=lambda: {"choices": [{"message": {"content": body}}]})

    client.session.post = post
    return client, calls


def test_openai_client_json_mode_and_auth():
    client, calls = _http_client([(200, '{"a": 1}', {})])
    assert client.generate_json("p") == {"a": 1}
    url, body = calls[0]
    assert url == "https://api.example/v1/chat/completions"
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][-1]["content"] == "p"
    assert client.session.headers["Authorization"] == "Bearer k"


def test_openai_client_images_only_with_vision():
    img = Attachment(b"png", "image/png")
    client, calls = _http_client([(200, "{}", {})], vision=False)
    client.generate_json("p", [img])
    assert calls[0][1]["messages"][-1]["content"] == "p"

    client, calls = _http_client([(200, "{}", {})], vision=True)
    client.generate_json("p", [img, Attachment(b"%PDF", "application/pdf")])
    parts = calls[0][1]["messages"][-1]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_openai_client_bad_key_not_retried():
    client, calls = _http_client([(401, "invalid api key", {})])
    with pytest.raises(LLMError, match="Groq request failed: 401"):
        client.generate_json("p")
    assert len(calls) == 1


def test_openai_client_long_retry_after_opens_breaker():
    client, calls = _http_client([(429, "rate limit", {"retry-after": "3600"})])
    with pytest.raises(LLMError):
        client.generate_json("p")
    with pytest.raises(LLMError, match="temporarily unavailable"):
        client.generate_json("p")
    assert len(calls) == 1


def test_openai_client_short_retry_after_is_honoured(monkeypatch):
    sleeps: list = []
    monkeypatch.setattr("app.processing.llm.time.sleep", sleeps.append)
    client, calls = _http_client([(429, "tpm", {"retry-after": "7"}), (200, '{"ok": 1}', {})])
    assert client.generate_json("p") == {"ok": 1}
    assert sleeps == [7.0]


def test_groq_settings_and_factory():
    s = Settings(_env_file=None, llm_provider="groq", LLM_API_KEY="k")
    assert s.llm_available
    assert s.active_base_url == "https://api.groq.com/openai/v1"
    client = make_client(s)
    assert isinstance(client, OpenAICompatibleClient)
    assert client.model == s.active_model
    assert not Settings(_env_file=None, llm_provider="groq").llm_available


def test_openai_client_request_too_large_not_retried():
    client, calls = _http_client([(429, "Request too large for model on tokens per minute", {})])
    with pytest.raises(LLMError):
        client.generate_json("p")
    assert len(calls) == 1
