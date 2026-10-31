"""Gemini OpenAI-compatible calls with retry and deterministic local caching."""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

from graphkit.llm import make_client, model_name


def normalise_message(message):
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    if isinstance(message, dict):
        return message
    raise TypeError(f"Unsupported provider message type: {type(message).__name__}")


class ResponseCache:
    def __init__(self, path, enabled=True):
        self.path = Path(path)
        self.enabled = enabled
        self._values = None

    def _load(self):
        if self._values is None:
            if self.enabled and self.path.is_file():
                try:
                    self._values = json.loads(self.path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    self._values = {}
            else:
                self._values = {}
        return self._values

    def get(self, key):
        return self._load().get(key) if self.enabled else None

    def put(self, key, value):
        if not self.enabled:
            return
        self._load()[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self._values, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)


class GeminiProvider:
    def __init__(self, *, client=None, model=None, cache_path=None, cache_enabled=True,
                 max_attempts=3, base_delay=1.0, sleep=time.sleep):
        self.client = client or make_client()
        self.model = model or model_name()
        self.cache = ResponseCache(cache_path or Path(".orchestrator") / "llm_cache.json", cache_enabled)
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.sleep = sleep

    @staticmethod
    def _key(payload):
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _transient(exc):
        status = getattr(exc, "status_code", None)
        if status == 429 or isinstance(status, int) and status >= 500:
            return True
        name = type(exc).__name__.lower()
        return any(token in name for token in ("timeout", "ratelimit", "connection", "internalserver"))

    def complete(self, *, messages, tools, tool_choice="auto", purpose="llm", trace=None):
        payload = {"model": self.model, "messages": messages, "tools": tools,
                   "tool_choice": tool_choice, "purpose": purpose}
        key = self._key(payload)
        cached = self.cache.get(key)
        if cached is not None:
            if trace:
                trace.event("llm_call", model=self.model, purpose=purpose, attempt=0,
                            cached=True, response=cached)
            return cached
        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, tools=tools, tool_choice=tool_choice,
                )
                message = normalise_message(response.choices[0].message)
                self.cache.put(key, message)
                if trace:
                    trace.event("llm_call", model=self.model, purpose=purpose, attempt=attempt,
                                cached=False, response=message)
                return message
            except Exception as exc:
                last_error = exc
                if trace:
                    trace.event("llm_error", model=self.model, purpose=purpose, attempt=attempt,
                                transient=self._transient(exc), error=f"{type(exc).__name__}: {exc}")
                if attempt >= self.max_attempts or not self._transient(exc):
                    raise
                delay = self.base_delay * (2 ** (attempt - 1)) + random.uniform(0, self.base_delay / 4)
                self.sleep(delay)
        raise last_error
