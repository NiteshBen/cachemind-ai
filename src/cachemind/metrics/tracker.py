"""Observability and metrics tracker for cache performance, latency distributions, and savings."""

import threading
from typing import Any, Dict, List
import numpy as np

from cachemind.metrics.pricing import calculate_token_cost


class MetricsTracker:
    """Thread-safe metrics engine recording query telemetry and economic impact."""

    def __init__(self, default_pricing_model: str = "gemini-2.0-flash"):
        self.default_pricing_model = default_pricing_model
        self._exact_hits = 0
        self._semantic_hits = 0
        self._misses = 0
        self._bypasses = 0
        self._tokens_saved_prompt = 0
        self._tokens_saved_completion = 0
        self._latencies_l1: List[float] = []
        self._latencies_l2: List[float] = []
        self._latencies_miss: List[float] = []
        self._lock = threading.RLock()

    def record_hit(
        self,
        hit_type: str,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Records a cache hit event."""
        with self._lock:
            if hit_type == "exact":
                self._exact_hits += 1
                self._latencies_l1.append(latency_ms)
            else:
                self._semantic_hits += 1
                self._latencies_l2.append(latency_ms)
            self._tokens_saved_prompt += prompt_tokens
            self._tokens_saved_completion += completion_tokens

    def record_miss(self, latency_ms: float) -> None:
        """Records a cache miss event."""
        with self._lock:
            self._misses += 1
            self._latencies_miss.append(latency_ms)

    def record_bypass(self) -> None:
        """Records a bypass event."""
        with self._lock:
            self._bypasses += 1

    @property
    def total_queries(self) -> int:
        with self._lock:
            return self._exact_hits + self._semantic_hits + self._misses

    @property
    def total_hits(self) -> int:
        with self._lock:
            return self._exact_hits + self._semantic_hits

    @property
    def hit_rate(self) -> float:
        with self._lock:
            t = self.total_queries
            return (self.total_hits / t) if t > 0 else 0.0

    @property
    def exact_hit_rate(self) -> float:
        with self._lock:
            t = self.total_queries
            return (self._exact_hits / t) if t > 0 else 0.0

    @property
    def semantic_hit_rate(self) -> float:
        with self._lock:
            t = self.total_queries
            return (self._semantic_hits / t) if t > 0 else 0.0

    def get_latency_percentiles(self, tier: str = "all") -> Dict[str, float]:
        """Calculates p50, p90, p95, and p99 latency in milliseconds."""
        with self._lock:
            if tier == "l1":
                pool = self._latencies_l1
            elif tier == "l2":
                pool = self._latencies_l2
            elif tier == "miss":
                pool = self._latencies_miss
            else:
                pool = self._latencies_l1 + self._latencies_l2

            if not pool:
                return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}

            arr = np.array(pool, dtype=np.float32)
            return {
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "mean": float(np.mean(arr)),
            }

    def total_cost_saved_usd(self, model: str = None) -> float:
        """Calculates total dollar savings achieved through avoided LLM tokens."""
        with self._lock:
            target_model = model or self.default_pricing_model
            return calculate_token_cost(
                prompt_tokens=self._tokens_saved_prompt,
                completion_tokens=self._tokens_saved_completion,
                model=target_model,
            )

    def summary(self, model: str = None) -> Dict[str, Any]:
        """Returns structured JSON-serializable observability summary."""
        with self._lock:
            lat_all = self.get_latency_percentiles("all")
            lat_l1 = self.get_latency_percentiles("l1")
            lat_l2 = self.get_latency_percentiles("l2")
            total_tokens = self._tokens_saved_prompt + self._tokens_saved_completion
            saved_usd = self.total_cost_saved_usd(model)

            return {
                "total_queries": self.total_queries,
                "total_hits": self.total_hits,
                "exact_hits": self._exact_hits,
                "semantic_hits": self._semantic_hits,
                "misses": self._misses,
                "bypasses": self._bypasses,
                "hit_rate_pct": round(self.hit_rate * 100, 2),
                "exact_hit_rate_pct": round(self.exact_hit_rate * 100, 2),
                "semantic_hit_rate_pct": round(self.semantic_hit_rate * 100, 2),
                "tokens_saved": {
                    "prompt_tokens": self._tokens_saved_prompt,
                    "completion_tokens": self._tokens_saved_completion,
                    "total_tokens": total_tokens,
                },
                "estimated_cost_saved_usd": round(saved_usd, 5),
                "pricing_tier": model or self.default_pricing_model,
                "latency_ms": {
                    "overall": lat_all,
                    "l1_exact": lat_l1,
                    "l2_semantic": lat_l2,
                },
            }

    def reset(self) -> None:
        """Resets all metrics."""
        with self._lock:
            self._exact_hits = 0
            self._semantic_hits = 0
            self._misses = 0
            self._bypasses = 0
            self._tokens_saved_prompt = 0
            self._tokens_saved_completion = 0
            self._latencies_l1.clear()
            self._latencies_l2.clear()
            self._latencies_miss.clear()
