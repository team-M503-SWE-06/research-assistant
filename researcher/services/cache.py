"""TTL-aware cache of `(source, query) -> list[Source]` results.

Caches raw source lists, not final synthesized answers: LLM output isn't
deterministic per call, and the assignment only requires source-level caching.
"""

from __future__ import annotations

import hashlib
import re
import time

from pydantic import ValidationError

from ai.schemas import Source
from researcher.models import CacheEntry
from researcher.storage.cache_store import CacheBackend


def canonicalize_query(query: str) -> str:
    """Lowercase, strip, and collapse internal whitespace so equivalent queries share a key."""
    return re.sub(r"\s+", " ", query.strip().lower())


def cache_key(source: str, query: str) -> str:
    """A filesystem/dict-safe key for a (source, query) pair."""
    normalized = f"{source}:{canonicalize_query(query)}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class SourceCache:
    """TTL-aware cache in front of a `CacheBackend`."""

    def __init__(self, backend: CacheBackend, ttl_seconds: float) -> None:
        self._backend = backend
        self._ttl = ttl_seconds

    async def get(self, source: str, query: str) -> list[Source] | None:
        payload = await self._backend.read(cache_key(source, query))
        if payload is None:
            return None
        try:
            entry = CacheEntry.model_validate(payload)
        except ValidationError:
            # A payload written by an older, incompatible version is a miss,
            # not a crash -- same treatment as a corrupted file.
            return None
        if entry.is_expired:
            return None  # stale entry -> treated as a miss (lazy expiry, no sweeper)
        return list(entry.sources)

    async def set(self, source: str, query: str, sources: list[Source]) -> None:
        now = time.time()
        entry = CacheEntry(
            source=source,
            query=canonicalize_query(query),
            cached_at=now,
            expires_at=now + self._ttl,
            sources=sources,
        )
        await self._backend.write(cache_key(source, query), entry.model_dump(mode="json"))
