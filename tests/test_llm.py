import pytest

from quaketriage.agent.llm import LLM, LLMError, is_retryable, make_llm


class FakeError(Exception):
    def __init__(self, status):
        super().__init__(f"status {status}")
        self.status_code = status


class Fake:
    def __init__(self, provider, script):
        self.provider, self.model, self.script, self.n = provider, "m", list(script), 0

    def generate(self, prompt, system, temperature):
        self.n += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _llm(backends, tmp_path, **kw):
    delays = []
    llm = LLM(backends, cache_dir=tmp_path, max_retries=2, base_delay=1.0, temperature=0.0,
              sleep=delays.append, **kw)
    return llm, delays


def test_retries_429_with_exponential_backoff_then_succeeds(tmp_path):
    b = Fake("gemini", [FakeError(429), FakeError(503), "ok"])
    llm, delays = _llm([b], tmp_path)
    assert llm.complete("hi") == "ok"
    assert delays == [1.0, 2.0] and llm.calls == 3 and b.n == 3


def test_cache_serves_identical_request_without_a_call(tmp_path):
    b = Fake("gemini", ["first"])
    llm, _ = _llm([b], tmp_path)
    assert llm.complete("q", system="s") == "first"
    assert llm.complete("q", system="s") == "first"
    assert llm.calls == 1 and llm.cache_hits == 1
    b2 = Fake("gemini", ["other"])
    assert _llm([b2], tmp_path)[0].complete("q2", system="s") == "other"  # different prompt


def test_falls_back_to_second_provider_after_rate_limit(tmp_path):
    primary = Fake("gemini", [FakeError(429)] * 3)
    backup = Fake("groq", ["from groq"])
    llm, _ = _llm([primary, backup], tmp_path)
    assert llm.complete("x") == "from groq"
    assert llm.last_backend == "groq/m" and primary.n == 3


def test_non_retryable_error_fails_fast_without_leaking_details(tmp_path):
    b = Fake("gemini", [FakeError(400)])
    llm, delays = _llm([b], tmp_path)
    with pytest.raises(LLMError, match="status 400"):
        llm.complete("x")
    assert b.n == 1 and delays == []


def test_all_backends_exhausted_raises(tmp_path):
    llm, _ = _llm([Fake("gemini", [FakeError(429)] * 3)], tmp_path)
    with pytest.raises(LLMError, match="all LLM backends failed"):
        llm.complete("x")


def test_is_retryable_classification():
    assert is_retryable(FakeError(429)) and is_retryable(FakeError(500))
    assert not is_retryable(FakeError(401)) and is_retryable(ConnectionError())
    assert not is_retryable(ValueError("bug"))


def test_make_llm_uses_env_and_rejects_unknown_provider(monkeypatch, tmp_path):
    from quaketriage import config
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "groq")
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "other-model")
    config._load.cache_clear()
    llm = make_llm(cache_dir=tmp_path)
    assert [(b.provider, b.model) for b in llm.backends] == [("gemini", "some-model"),
                                                              ("groq", "other-model")]
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    config._load.cache_clear()
    with pytest.raises(LLMError, match="unknown LLM provider"):
        make_llm(cache_dir=tmp_path)
    config._load.cache_clear()  # do not leak the fake env into later tests


@pytest.fixture(autouse=True)
def _fresh_config():
    from quaketriage import config
    config._load.cache_clear()
    yield
    config._load.cache_clear()
