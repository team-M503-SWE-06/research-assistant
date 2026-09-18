"""Tests for researcher.services.ai_service.AIService (retries, logging)."""

from __future__ import annotations

import logging

import httpx
import pytest

from ai.providers.base import ProviderError
from researcher.services import ai_service as ai_service_module
from researcher.services.ai_service import AIService
from researcher.services.search_terms import wikipedia_search_candidates

# The candidate list itself is search_terms' business and is tested in
# tests/test_search_terms.py. These tests derive the expectation from it rather
# than hard-coding phrases, so that tuning the search-term algorithm does not
# break tests that are really about how AIService walks the candidates.
QUESTION = "What is photosynthesis and what are its main stages?"
CANDIDATES = wikipedia_search_candidates(QUESTION)
MATCHING_TERM = "photosynthesis"


@pytest.mark.asyncio
async def test_fetch_wikipedia_retries_then_succeeds(monkeypatch, settings_factory, sample_sources):
    calls = {"n": 0}

    async def flaky(query, *, max_results=3, client=None):
        calls["n"] += 1
        if calls["n"] < 2:
            raise ProviderError("transient failure")
        return sample_sources

    monkeypatch.setattr(ai_service_module.ai_sources, "fetch_wikipedia", flaky)
    service = AIService(settings_factory(retry_max_attempts=3, retry_backoff_seconds=0.001))

    result = await service.fetch_wikipedia("q", max_results=3)

    assert result == sample_sources
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_fetch_wikipedia_raises_after_exhausting_retries(monkeypatch, settings_factory):
    calls = {"n": 0}

    async def always_fails(query, *, max_results=3, client=None):
        calls["n"] += 1
        raise ProviderError("permanent failure")

    monkeypatch.setattr(ai_service_module.ai_sources, "fetch_wikipedia", always_fails)
    service = AIService(settings_factory(retry_max_attempts=2, retry_backoff_seconds=0.001))

    with pytest.raises(ProviderError):
        await service.fetch_wikipedia("q", max_results=3)
    # "q" yields a single candidate, so every call here is a retry of the same
    # term. Asserting the count is what distinguishes giving up after the
    # configured number of attempts from never retrying at all.
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_fetch_wikipedia_retries_on_httpx_error(monkeypatch, settings_factory, sample_sources):
    """The retry policy covers transport failures, not only ProviderError."""
    calls = {"n": 0}

    async def flaky(query, *, max_results=3, client=None):
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.ConnectError("connection refused")
        return sample_sources

    monkeypatch.setattr(ai_service_module.ai_sources, "fetch_wikipedia", flaky)
    service = AIService(settings_factory(retry_max_attempts=3, retry_backoff_seconds=0.001))

    result = await service.fetch_wikipedia("q", max_results=3)

    assert result == sample_sources
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_fetch_wikipedia_does_not_retry_on_value_error(monkeypatch, settings_factory):
    calls = {"n": 0}

    async def bad_input(query, *, max_results=3, client=None):
        calls["n"] += 1
        raise ValueError("bad input")

    monkeypatch.setattr(ai_service_module.ai_sources, "fetch_wikipedia", bad_input)
    service = AIService(settings_factory(retry_max_attempts=3, retry_backoff_seconds=0.001))

    with pytest.raises(ValueError):
        await service.fetch_wikipedia("q", max_results=3)
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_fetch_wikipedia_falls_back_to_shorter_query_until_match(
    monkeypatch, settings_factory, sample_sources
):
    queried: list[str] = []

    async def title_search(query, *, max_results=3, client=None):
        queried.append(query)
        return sample_sources if query == MATCHING_TERM else []

    monkeypatch.setattr(ai_service_module.ai_sources, "fetch_wikipedia", title_search)
    service = AIService(settings_factory())

    result = await service.fetch_wikipedia(QUESTION, max_results=3)

    assert result == sample_sources
    # Walks the candidates in order and stops at the first one that matches.
    assert queried == CANDIDATES[: CANDIDATES.index(MATCHING_TERM) + 1]


@pytest.mark.asyncio
async def test_fetch_wikipedia_returns_empty_when_no_candidate_matches(monkeypatch, settings_factory):
    queried: list[str] = []

    async def no_titles(query, *, max_results=3, client=None):
        queried.append(query)
        return []

    monkeypatch.setattr(ai_service_module.ai_sources, "fetch_wikipedia", no_titles)
    service = AIService(settings_factory())

    result = await service.fetch_wikipedia(QUESTION, max_results=3)

    assert result == []
    # Nothing matched, so every candidate was tried.
    assert queried == CANDIDATES


@pytest.mark.asyncio
async def test_fetch_wikipedia_provider_error_propagates_without_trying_next_candidate(
    monkeypatch, settings_factory
):
    queried: list[str] = []

    async def down(query, *, max_results=3, client=None):
        queried.append(query)
        raise ProviderError("Wikipedia search failed")

    monkeypatch.setattr(ai_service_module.ai_sources, "fetch_wikipedia", down)
    service = AIService(settings_factory(retry_max_attempts=1))

    with pytest.raises(ProviderError):
        await service.fetch_wikipedia(QUESTION, max_results=3)
    # The error surfaces instead of being masked by moving to the next candidate.
    assert queried == CANDIDATES[:1]


@pytest.mark.asyncio
async def test_fetch_logs_never_include_api_key(monkeypatch, settings_factory, sample_sources, caplog):
    async def fake(query, *, max_results=3, client=None):
        return sample_sources

    monkeypatch.setattr(ai_service_module.ai_sources, "fetch_wikipedia", fake)
    service = AIService(settings_factory(tavily_api_key="super-secret-key"))

    with caplog.at_level(logging.DEBUG):
        await service.fetch_wikipedia("q", max_results=3)

    assert "super-secret-key" not in caplog.text


def test_synthesize_retries_transient_provider_error(
    monkeypatch, settings_factory, sample_sources, fake_llm
):
    """The synthesis path is wrapped in the same retry policy as the fetch path."""
    from ai.synthesizer import synthesize as real_synthesize

    calls = {"n": 0}

    def flaky(question, sources, *, llm=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProviderError("transient failure")
        return real_synthesize(question, sources, llm=fake_llm)

    monkeypatch.setattr(ai_service_module.ai_synth, "synthesize", flaky)
    service = AIService(settings_factory(retry_max_attempts=3, retry_backoff_seconds=0.001))

    answer = service.synthesize("What is photosynthesis?", sample_sources)

    assert calls["n"] == 3
    assert answer.citations


def test_synthesize_does_not_retry_on_value_error(monkeypatch, settings_factory, sample_sources):
    """Bad input is permanent: it must fail on the first attempt, like fetch does."""
    calls = {"n": 0}

    def bad_input(question, sources, *, llm=None):
        calls["n"] += 1
        raise ValueError("no sources to synthesize from")

    monkeypatch.setattr(ai_service_module.ai_synth, "synthesize", bad_input)
    service = AIService(settings_factory(retry_max_attempts=3, retry_backoff_seconds=0.001))

    with pytest.raises(ValueError):
        service.synthesize("q", sample_sources)
    assert calls["n"] == 1


def test_synthesize_wraps_ai_synthesizer(monkeypatch, settings_factory, sample_sources, fake_llm):
    from ai.synthesizer import synthesize as real_synthesize

    def synth(question, sources, *, llm=None):
        return real_synthesize(question, sources, llm=fake_llm)

    monkeypatch.setattr(ai_service_module.ai_synth, "synthesize", synth)
    service = AIService(settings_factory())

    answer = service.synthesize("What is photosynthesis?", sample_sources)

    assert answer.question == "What is photosynthesis?"
    assert answer.citations
