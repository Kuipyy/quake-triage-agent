"""LLM adapter: Gemini / Groq behind one interface, with retry, provider fallback and a disk cache.

Provider and model names come from ``.env`` (``LLM_PROVIDER``/``LLM_MODEL``, optional
``LLM_FALLBACK_PROVIDER``/``LLM_FALLBACK_MODEL``) via ``configs/default.yaml``; API keys are read
from the environment only and never logged. Identical requests are served from ``cache_dir``
(not counted as LLM calls). Rate limits (429) and transient server errors are retried with
exponential backoff; if a backend stays rate-limited the next one is tried.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from quaketriage.config import get

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
ROOT = Path(__file__).parents[3]


class LLMError(RuntimeError):
    """Raised when no backend could answer (message never contains secrets)."""


class Backend(Protocol):
    provider: str
    model: str

    def generate(self, prompt: str, system: str | None, temperature: float) -> str: ...


def status_of(exc: BaseException) -> int | None:
    """HTTP status carried by an SDK exception (google-genai: ``code``; groq: ``status_code``)."""
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def is_retryable(exc: BaseException) -> bool:
    status = status_of(exc)
    if status is not None:
        return status in RETRYABLE_STATUS
    return isinstance(exc, (TimeoutError, ConnectionError)) or "Connection" in type(exc).__name__


class GeminiBackend:
    provider = "gemini"

    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self._key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._client = None

    def generate(self, prompt: str, system: str | None, temperature: float) -> str:
        if self._client is None:
            from google import genai

            if not self._key:
                raise LLMError("GEMINI_API_KEY is not set (put it in .env)")
            self._client = genai.Client(api_key=self._key)
        from google.genai import types

        cfg = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
        return self._client.models.generate_content(
            model=self.model, contents=prompt, config=cfg).text or ""


class GroqBackend:
    provider = "groq"

    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self._key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._client = None

    def generate(self, prompt: str, system: str | None, temperature: float) -> str:
        if self._client is None:
            import groq

            if not self._key:
                raise LLMError("GROQ_API_KEY is not set (put it in .env)")
            self._client = groq.Groq(api_key=self._key)
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}]
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature)
        return resp.choices[0].message.content or ""


_BACKENDS = {"gemini": GeminiBackend, "groq": GroqBackend}


class LLM:
    """Ordered backends + retry + cache. ``calls`` counts real backend requests (not cache hits)."""

    def __init__(self, backends: list[Backend], cache_dir: str | Path | None = None,
                 max_retries: int | None = None, base_delay: float | None = None,
                 temperature: float | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        if not backends:
            raise LLMError("no LLM backend configured (set LLM_PROVIDER and LLM_MODEL in .env)")
        self.backends = backends
        self.cache_dir = Path(cache_dir) if cache_dir else ROOT / get("llm.cache_dir")
        self.max_retries = get("llm.max_retries") if max_retries is None else max_retries
        self.base_delay = get("llm.retry_base_delay") if base_delay is None else base_delay
        self.temperature = get("llm.temperature") if temperature is None else temperature
        self._sleep = sleep
        self.calls = 0
        self.cache_hits = 0
        self.last_backend: str | None = None

    def _key(self, prompt: str, system: str | None) -> str:
        b = self.backends[0]
        raw = json.dumps([b.provider, b.model, system, prompt, self.temperature])
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def complete(self, prompt: str, system: str | None = None) -> str:
        path = self.cache_dir / f"{self._key(prompt, system)}.json"
        if path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text())["text"]
        errors = []
        for backend in self.backends:
            for attempt in range(self.max_retries + 1):
                try:
                    self.calls += 1
                    text = backend.generate(prompt, system, self.temperature)
                except LLMError:
                    raise
                except Exception as exc:  # SDK-specific types; classified by status code
                    if not is_retryable(exc):
                        raise LLMError(f"{backend.provider} request failed "
                                       f"(status {status_of(exc)}): {type(exc).__name__}") from exc
                    errors.append(f"{backend.provider}:{status_of(exc)}")
                    if attempt < self.max_retries:
                        self._sleep(self.base_delay * 2 ** attempt)
                    continue
                self.last_backend = f"{backend.provider}/{backend.model}"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"text": text, "backend": self.last_backend}))
                return text
        raise LLMError(f"all LLM backends failed after retries ({', '.join(errors)})")


def make_llm(**overrides) -> LLM:
    """Build the client from config/.env: primary backend, plus the fallback if one is set."""
    backends: list[Backend] = []
    for prefix in ("", "fallback_"):
        provider = get(f"llm.{prefix}provider") if prefix else get("llm.provider")
        model = get(f"llm.{prefix}model") if prefix else get("llm.model")
        if not provider or not model:
            continue
        if provider not in _BACKENDS:
            raise LLMError(f"unknown LLM provider {provider!r}; use one of {sorted(_BACKENDS)}")
        backends.append(_BACKENDS[provider](model))
    return LLM(backends, **overrides)
