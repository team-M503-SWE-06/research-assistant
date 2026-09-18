"""Domain models for the researcher package. No naked dicts cross module boundaries."""

from __future__ import annotations

import time
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from ai.schemas import AnswerWithCitations, Source


class CacheEntry(BaseModel):
    """One cached `(source, query)` result, as stored by `SourceCache`.

    Typed rather than a bare dict so the cache's on-disk shape is declared in
    one place: a renamed field breaks here, not silently at a string-keyed
    lookup in whichever module happens to read it next.
    """

    source: str
    query: str
    cached_at: float
    expires_at: float
    sources: list[Source]

    @property
    def is_expired(self) -> bool:
        """True once `expires_at` has passed. Expiry is lazy, checked on read."""
        return time.time() > self.expires_at


class SourceOutcome(BaseModel):
    """The result of trying to fetch one source (wikipedia/arxiv/web) for a question."""

    origin: str
    source_count: int
    error: str | None = None
    elapsed_seconds: float
    from_cache: bool = False


class ResearchSession(BaseModel):
    """A completed research question: the synthesized answer plus per-source outcomes."""

    question: str
    answer: AnswerWithCitations
    outcomes: list[SourceOutcome]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def degraded(self) -> bool:
        """True if at least one requested source failed to produce results."""
        return any(o.error for o in self.outcomes)
