"""Configuration and data models for CacheMind-AI."""

from typing import Any, Dict, List, Literal, Optional
import time
from pydantic import BaseModel, Field


class CacheConfig(BaseModel):
    """Runtime configuration for SemanticCache."""

    similarity_threshold: float = Field(
        default=0.88,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity required to trigger a semantic cache hit.",
    )
    exact_match_l1: bool = Field(
        default=True,
        description="Enable L1 O(1) hash lookup before computing vector embeddings.",
    )
    storage_backend: Literal["memory", "sqlite"] = Field(
        default="memory",
        description="Underlying storage medium for cache entries.",
    )
    sqlite_path: str = Field(
        default="cachemind_store.db",
        description="File path for SQLite persistence when storage_backend='sqlite'.",
    )
    max_entries: int = Field(
        default=1000,
        gt=0,
        description="Maximum entries stored before applying the eviction policy.",
    )
    eviction_policy: Literal["lru", "lfu", "fifo"] = Field(
        default="lru",
        description="Eviction strategy when cache reaches max_entries capacity.",
    )
    ttl_seconds: Optional[int] = Field(
        default=None,
        description="Optional time-to-live in seconds for cached responses.",
    )
    embedding_provider: Literal["tfidf", "gemini"] = Field(
        default="tfidf",
        description="Embedding model provider for semantic similarity.",
    )
    pricing_model: str = Field(
        default="gemini-2.0-flash",
        description="LLM pricing tier used for token cost arbitrage calculations.",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or domain isolation namespace.",
    )


class CacheEntry(BaseModel):
    """A single cached prompt-response record with vector embedding and telemetry."""

    key: str
    prompt: str
    response: str
    embedding: List[float] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    last_accessed: float = Field(default_factory=time.time)
    access_count: int = 1
    prompt_tokens: int = 0
    completion_tokens: int = 0
    namespace: str = "default"
    ttl_seconds: Optional[int] = None

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Determines if this entry has passed its TTL expiration."""
        if self.ttl_seconds is None:
            return False
        now = current_time if current_time is not None else time.time()
        return (now - self.created_at) > self.ttl_seconds

    def touch(self) -> None:
        """Updates last_accessed timestamp and increments access_count."""
        self.last_accessed = time.time()
        self.access_count += 1


class CacheLookupResult(BaseModel):
    """Result of a cache query containing hit status, latency, and savings metrics."""

    hit: bool
    hit_type: Optional[Literal["exact", "semantic"]] = None
    similarity: float = 0.0
    response: Optional[str] = None
    matched_prompt: Optional[str] = None
    latency_ms: float = 0.0
    tokens_saved: int = 0
    cost_saved_usd: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    entry_key: Optional[str] = None
