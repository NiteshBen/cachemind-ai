"""Similarity search and metrics module for CacheMind-AI."""

from cachemind.similarity.metrics import (
    cosine_similarity,
    batch_cosine_similarity,
    token_jaccard_similarity,
    hybrid_similarity,
)
from cachemind.similarity.index import VectorIndex

__all__ = [
    "cosine_similarity",
    "batch_cosine_similarity",
    "token_jaccard_similarity",
    "hybrid_similarity",
    "VectorIndex",
]
