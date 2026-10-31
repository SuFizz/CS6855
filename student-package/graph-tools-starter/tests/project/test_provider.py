from pathlib import Path
from types import SimpleNamespace

import pytest

from orchestrator.provider import GeminiProvider


class RateLimitError(Exception):
    status_code = 429


class Client:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            model_dump=lambda exclude_none=True: outcome,
        ))])


def test_provider_retries_transient_errors_and_caches(tmp_path):
    client = Client([RateLimitError("slow down"), {"role": "assistant", "content": "ok"}])
    provider = GeminiProvider(client=client, model="test", cache_path=tmp_path / "cache.json",
                              max_attempts=3, base_delay=0, sleep=lambda _: None)
    arguments = {"messages": [{"role": "user", "content": "x"}], "tools": [], "purpose": "test"}
    first = provider.complete(**arguments)
    second = provider.complete(**arguments)
    assert first == second
    assert client.calls == 2


def test_provider_does_not_retry_non_transient_errors(tmp_path):
    client = Client([ValueError("bad request")])
    provider = GeminiProvider(client=client, model="test", cache_path=tmp_path / "cache.json",
                              max_attempts=3, base_delay=0, sleep=lambda _: None)
    with pytest.raises(ValueError):
        provider.complete(messages=[], tools=[], purpose="test")
    assert client.calls == 1
