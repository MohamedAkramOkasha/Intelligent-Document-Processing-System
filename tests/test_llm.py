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
