"""CacheMind-AI: Production Semantic Caching & Cost Optimization Engine for LLMs & AI Agents.

Sub-millisecond semantic vector caching, cost arbitrage, exact L1 + semantic L2 tiering,
multi-backend storage, and real-time observability.
"""

from cachemind.config import CacheConfig, CacheEntry, CacheLookupResult
from cachemind.core import SemanticCache
from cachemind.decorators import cached
from cachemind.embeddings.base import BaseEmbeddingProvider
from cachemind.embeddings.local_tfidf import LocalTFIDFEmbedder
from cachemind.metrics.tracker import MetricsTracker

__version__ = "0.1.0"
__author__ = "Nitesh Beniwal"

__all__ = [
    "SemanticCache",
    "CacheConfig",
    "CacheEntry",
    "CacheLookupResult",
    "cached",
    "BaseEmbeddingProvider",
    "LocalTFIDFEmbedder",
    "MetricsTracker",
]
