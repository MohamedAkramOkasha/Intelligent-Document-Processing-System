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
    client, calls = _client([FakeAPIError(500, "internal error"), '{"ok": true}'])
    assert client.generate_json("p") == {"ok": True}
    assert len(calls) == 2


@pytest.mark.parametrize("error", [
    FakeAPIError(503, "UNAVAILABLE: This model is currently experiencing high demand"),
    TimeoutError("The read operation timed out"),
])
def test_overloaded_or_hanging_provider_fails_fast(error):
    """No retries (each can take the full timeout), and the next call skips the provider."""
    client, calls = _client([error, '{"ok": true}'])
    with pytest.raises(LLMError):
        client.generate_json("p")
    with pytest.raises(LLMError, match="temporarily unavailable: overloaded"):
        client.generate_json("p")
    assert len(calls) == 1


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
    assert len(sleeps) == 1 and 6.5 < sleeps[0] < 8.1  # retry-after + jitter


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


# ---------------------------------------------------------------- rate limits & fallback
from app.processing.llm import FallbackClient, _retry_delay_s  # noqa: E402


def test_retry_delay_parsing():
    assert _retry_delay_s(Exception("Please try again in 6.72s.")) == 6.72
    assert _retry_delay_s(Exception("try again in 1h2m3s")) == 3723
    assert _retry_delay_s(Exception("try again in 2m30.5s")) == 150.5
    assert _retry_delay_s(Exception("try again in 450ms")) == 0.45
    assert _retry_delay_s(Exception("no hint")) is None


def test_rate_limits_do_not_consume_retries(monkeypatch):
    monkeypatch.setattr("app.processing.llm.time.sleep", lambda _: None)
    monkeypatch.setattr("app.processing.llm.time.time", lambda: 0.0)  # pauses expire instantly
    tpm = (429, "Rate limit reached. Please try again in 2s", {})
    client, calls = _http_client([tpm, tpm, tpm, tpm, (200, '{"ok": 1}', {})])  # max_retries=3
    assert client.generate_json("p") == {"ok": 1}
    assert len(calls) == 5


def test_rate_limit_wait_budget_then_give_up(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("app.processing.llm.time.time", lambda: clock[0])
    monkeypatch.setattr("app.processing.llm.time.sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    tpm = (429, "Please try again in 30s", {})
    client, calls = _http_client([tpm] * 10)
    client.rate_limit_max_wait_s = 60
    with pytest.raises(LLMError, match="rate limited"):
        client.generate_json("p")
    assert len(calls) == 2  # waited ~30s once; a second ~30s wait would exceed the budget


def test_rate_limit_pause_is_shared(monkeypatch):
    """A 429 seen by one request delays the next request on the same client."""
    clock = [0.0]
    sleeps: list = []
    monkeypatch.setattr("app.processing.llm.time.time", lambda: clock[0])

    def sleep(s):
        sleeps.append(s)
        clock[0] += s

    monkeypatch.setattr("app.processing.llm.time.sleep", sleep)
    client, _ = _http_client([(429, "Please try again in 5s", {}), (200, "{}", {}), (200, "{}", {})])
    client._pause(5)  # as if another worker just hit the limit
    client.generate_json("p")
    assert sleeps and sleeps[0] == 5


def test_concurrency_limit():
    import threading as th

    client = OpenAICompatibleClient(base_url="https://x/v1", model="m", max_concurrent=1)
    active, peak, lock = [0], [0], th.Lock()

    def request(prompt, attachments):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        import time as t
        t.sleep(0.02)
        with lock:
            active[0] -= 1
        return "{}"

    client._request = request
    threads = [th.Thread(target=client.generate_json, args=("p",)) for _ in range(4)]
    for t_ in threads:
        t_.start()
    for t_ in threads:
        t_.join()
    assert peak[0] == 1


class _Stub:
    def __init__(self, label, result, sees_files=False):
        self.label, self.result, self.sees_files, self.calls = label, result, sees_files, 0

    def generate_json(self, prompt, attachments=None):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_fallback_client_uses_next_provider():
    groq = _Stub("groq:m", LLMError("Groq rate limited"))
    gemini = _Stub("gemini:g", {"ok": 1}, sees_files=True)
    fb = FallbackClient([groq, gemini])
    assert fb.generate_json("p") == {"ok": 1}
    assert fb.last_label == "gemini:g"
    assert fb.label == "groq:m -> gemini:g" and fb.sees_files


def test_fallback_client_all_fail():
    fb = FallbackClient([_Stub("a", LLMError("A down")), _Stub("b", LLMError("B down"))])
    with pytest.raises(LLMError, match="A down .* B down"):
        fb.generate_json("p")


def test_fallback_settings_chain():
    s = Settings(_env_file=None, llm_provider="groq", LLM_API_KEY="k", GEMINI_API_KEY="g",
                 llm_fallback_provider="gemini")
    assert s.llm_providers == ["groq", "gemini"]
    assert s.llm_label == "groq:openai/gpt-oss-120b -> gemini:gemini-3.8-flash"
    client = make_client(s)
    assert isinstance(client, FallbackClient)
    assert client.clients[0]._slots is not None  # groq: 1 request at a time by default
    assert client.clients[1]._slots is None
    # fallback without its key is dropped; two OpenAI-compatible providers are not chained
    assert Settings(_env_file=None, llm_provider="groq", LLM_API_KEY="k",
                    llm_fallback_provider="gemini").llm_providers == ["groq"]
    assert Settings(_env_file=None, llm_provider="groq", LLM_API_KEY="k",
                    llm_fallback_provider="openai").llm_providers == ["groq"]


def test_fallback_client_files_only_skips_text_only_providers():
    groq = _Stub("groq:m", {"empty": True})
    gemini = _Stub("gemini:g", {"ok": 1}, sees_files=True)
    fb = FallbackClient([groq, gemini])
    assert fb.generate_json("p", files_only=True) == {"ok": 1}
    assert groq.calls == 0 and fb.last_label == "gemini:g"
    # no provider sees files: try them all as usual rather than failing
    fb = FallbackClient([_Stub("a", {"a": 1}), _Stub("b", {"b": 1})])
    assert fb.generate_json("p", files_only=True) == {"a": 1}
