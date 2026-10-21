"""Core SemanticCache engine orchestrating L1 exact match, L2 vector search, storage, and telemetry."""

import hashlib
import time
from typing import Any, Callable, Dict, Optional, Tuple

from cachemind.config import CacheConfig, CacheEntry, CacheLookupResult
from cachemind.embeddings.base import BaseEmbeddingProvider
from cachemind.embeddings.gemini import GeminiEmbeddingProvider
from cachemind.embeddings.local_tfidf import LocalTFIDFEmbedder
from cachemind.metrics.pricing import calculate_token_cost
from cachemind.metrics.tracker import MetricsTracker
from cachemind.similarity.index import VectorIndex
from cachemind.storage.base import BaseStorage
from cachemind.storage.memory import InMemoryStorage
from cachemind.storage.sqlite_store import SQLiteStorage


class SemanticCache:
    """Enterprise-grade semantic caching engine for LLMs and AI agent workflows."""

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        storage: Optional[BaseStorage] = None,
        embedder: Optional[BaseEmbeddingProvider] = None,
        **kwargs,
    ):
        if config is None:
            self.config = CacheConfig(**kwargs) if kwargs else CacheConfig()
        else:
            self.config = config

        # 1. Initialize Embedder
        if embedder is not None:
            self.embedder = embedder
        elif self.config.embedding_provider == "gemini":
            self.embedder = GeminiEmbeddingProvider()
        else:
            self.embedder = LocalTFIDFEmbedder()

        # 2. Initialize Vector Index
        self.index = VectorIndex(dimension=self.embedder.dimension)

        # 3. Initialize Storage Backend
        if storage is not None:
            self.storage = storage
        elif self.config.storage_backend == "sqlite":
            self.storage = SQLiteStorage(
                db_path=self.config.sqlite_path,
                max_entries=self.config.max_entries,
                eviction_policy=self.config.eviction_policy,
            )
        else:
            self.storage = InMemoryStorage(
                max_entries=self.config.max_entries,
                eviction_policy=self.config.eviction_policy,
            )

        # 4. Initialize Metrics Tracker
        self.metrics = MetricsTracker(default_pricing_model=self.config.pricing_model)

        # Warm index if storage already contains entries (e.g. SQLite reopen)
        self._warm_index_from_storage()

    def _warm_index_from_storage(self) -> None:
        """Loads existing entries into the in-memory vector index."""
        try:
            entries = self.storage.all_entries(namespace=self.config.namespace)
            for e in entries:
                if e.embedding:
                    self.index.add(e.key, e.embedding)
        except Exception:
            pass

    @staticmethod
    def normalize_prompt(prompt: str) -> str:
        """Strips surrounding whitespace and standardizes casing for exact hashing."""
        return prompt.strip()

    def _make_key(self, prompt: str, namespace: str) -> str:
        """Deterministic SHA256 key for L1 exact match."""
        normalized = self.normalize_prompt(prompt).lower()
        seed = f"{namespace}:{normalized}"
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()

    def _estimate_tokens(self, text: str) -> int:
        """Heuristic estimation of token count (~4 characters per token)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def get(
        self,
        prompt: str,
        namespace: Optional[str] = None,
    ) -> CacheLookupResult:
        """Queries the cache with L1 exact match and L2 semantic vector similarity."""
        t0 = time.perf_counter()
        ns = namespace or self.config.namespace
        clean_prompt = self.normalize_prompt(prompt)

        if not clean_prompt:
            lat = (time.perf_counter() - t0) * 1000.0
            self.metrics.record_miss(lat)
            return CacheLookupResult(hit=False, latency_ms=lat)

        # --- TIER 1: L1 Exact Hash Lookup ---
        if self.config.exact_match_l1:
            exact_key = self._make_key(clean_prompt, ns)
            entry = self.storage.get(exact_key)
            if entry and not entry.is_expired():
                lat = (time.perf_counter() - t0) * 1000.0
                tokens_saved = entry.prompt_tokens + entry.completion_tokens
                cost_saved = calculate_token_cost(
                    entry.prompt_tokens,
                    entry.completion_tokens,
                    self.config.pricing_model,
                )
                self.metrics.record_hit(
                    hit_type="exact",
                    latency_ms=lat,
                    prompt_tokens=entry.prompt_tokens,
                    completion_tokens=entry.completion_tokens,
                )
                return CacheLookupResult(
                    hit=True,
                    hit_type="exact",
                    similarity=1.0,
                    response=entry.response,
                    matched_prompt=entry.prompt,
                    latency_ms=lat,
                    tokens_saved=tokens_saved,
                    cost_saved_usd=cost_saved,
                    metadata=entry.metadata,
                    entry_key=entry.key,
                )

        # --- TIER 2: L2 Semantic Vector Search ---
        query_vec = self.embedder.embed_query(clean_prompt)
        matches = self.index.search(
            query_vec,
            top_k=1,
            threshold=self.config.similarity_threshold,
        )

        if matches:
            matched_key, similarity = matches[0]
            entry = self.storage.get(matched_key)
            if entry and not entry.is_expired() and entry.namespace == ns:
                lat = (time.perf_counter() - t0) * 1000.0
                tokens_saved = entry.prompt_tokens + entry.completion_tokens
                cost_saved = calculate_token_cost(
                    entry.prompt_tokens,
                    entry.completion_tokens,
                    self.config.pricing_model,
                )
                self.metrics.record_hit(
                    hit_type="semantic",
                    latency_ms=lat,
                    prompt_tokens=entry.prompt_tokens,
                    completion_tokens=entry.completion_tokens,
                )
                return CacheLookupResult(
                    hit=True,
                    hit_type="semantic",
                    similarity=round(float(similarity), 4),
                    response=entry.response,
                    matched_prompt=entry.prompt,
                    latency_ms=lat,
                    tokens_saved=tokens_saved,
                    cost_saved_usd=cost_saved,
                    metadata=entry.metadata,
                    entry_key=entry.key,
                )

        # --- TIER 3: Cache Miss ---
        lat = (time.perf_counter() - t0) * 1000.0
        self.metrics.record_miss(lat)
        return CacheLookupResult(
            hit=False,
            latency_ms=lat,
            similarity=0.0,
        )

    def set(
        self,
        prompt: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        ttl_seconds: Optional[int] = None,
    ) -> CacheEntry:
        """Stores a prompt-response pair with dense vector embedding into storage and index."""
        ns = namespace or self.config.namespace
        clean_prompt = self.normalize_prompt(prompt)
        key = self._make_key(clean_prompt, ns)

        # Embed prompt
        embedding = self.embedder.embed_query(clean_prompt)

        p_toks = prompt_tokens if prompt_tokens is not None else self._estimate_tokens(clean_prompt)
        c_toks = completion_tokens if completion_tokens is not None else self._estimate_tokens(response)
        ttl = ttl_seconds if ttl_seconds is not None else self.config.ttl_seconds

        entry = CacheEntry(
            key=key,
            namespace=ns,
            prompt=clean_prompt,
            response=response,
            embedding=embedding,
            metadata=metadata or {},
            prompt_tokens=p_toks,
            completion_tokens=c_toks,
            ttl_seconds=ttl,
        )

        # Update storage and vector index
        self.storage.set(entry)
        self.index.add(key, embedding)
        return entry

    def call_or_cache(
        self,
        prompt: str,
        llm_callable: Callable[[str], str],
        namespace: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> Tuple[str, CacheLookupResult]:
        """Looks up cached answer; executes callable on miss and caches response transparently."""
        lookup = self.get(prompt, namespace=namespace)
        if lookup.hit and lookup.response is not None:
            return lookup.response, lookup

        # Execute upstream LLM
        t0 = time.perf_counter()
        raw_response = llm_callable(prompt)
        llm_latency_ms = (time.perf_counter() - t0) * 1000.0

        meta = metadata or {}
        meta["llm_latency_ms"] = round(llm_latency_ms, 2)

        self.set(
            prompt=prompt,
            response=raw_response,
            metadata=meta,
            namespace=namespace,
            ttl_seconds=ttl_seconds,
        )

        return raw_response, lookup

    def delete(self, key: str) -> bool:
        """Removes an entry from storage and index."""
        self.index.remove(key)
        return self.storage.delete(key)

    def clear(self, namespace: Optional[str] = None) -> None:
        """Clears all or namespaced entries."""
        self.storage.clear(namespace=namespace)
        if namespace is None or namespace == self.config.namespace:
            self.index.clear()
        else:
            self._warm_index_from_storage()

    def stats(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Retrieves aggregated metrics, latency percentiles, and cost savings."""
        summary = self.metrics.summary(model=model or self.config.pricing_model)
        summary["active_entries"] = self.storage.count(namespace=self.config.namespace)
        summary["storage_backend"] = self.config.storage_backend
        summary["similarity_threshold"] = self.config.similarity_threshold
        summary["embedding_dimension"] = self.embedder.dimension
        return summary

    def export_dashboard(self, output_path: str = "cachemind_dashboard.html") -> str:
        """Renders an interactive, standalone HTML analytics dashboard."""
        from cachemind.visualizer import generate_dashboard_html

        entries = self.storage.all_entries(namespace=self.config.namespace)
        stats = self.stats()
        return generate_dashboard_html(stats=stats, entries=entries, output_path=output_path)
