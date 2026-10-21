"""Similarity and distance computation metrics for vector space and lexical overlap."""

import re
from typing import List, Set
import numpy as np


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two float vectors.

    Assumes vectors are already L2 normalized for maximum performance,
    falling back to full norm calculation if needed.
    """
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0

    a = np.asarray(v1, dtype=np.float32)
    b = np.asarray(v2, dtype=np.float32)

    dot = float(np.dot(a, b))
    # If already unit length (dot product within [-1.0, 1.0]), clamp and return
    if 0.98 <= float(np.linalg.norm(a)) <= 1.02 and 0.98 <= float(np.linalg.norm(b)) <= 1.02:
        return max(-1.0, min(1.0, dot))

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0

    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def batch_cosine_similarity(query_vec: List[float], matrix: np.ndarray) -> np.ndarray:
    """Computes vector matrix inner product for batch similarity search."""
    if matrix.size == 0:
        return np.array([], dtype=np.float32)

    q = np.asarray(query_vec, dtype=np.float32)
    # If matrix is (N, D) and q is (D,), matrix @ q gives (N,)
    scores = np.dot(matrix, q)
    return np.clip(scores, -1.0, 1.0)


def token_jaccard_similarity(text1: str, text2: str) -> float:
    """Calculates word-token Jaccard set intersection over union."""
    if not text1 or not text2:
        return 0.0

    tokens1: Set[str] = set(re.findall(r"\w+", text1.lower()))
    tokens2: Set[str] = set(re.findall(r"\w+", text2.lower()))

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return float(intersection) / float(union) if union > 0 else 0.0


def hybrid_similarity(
    vec1: List[float],
    vec2: List[float],
    text1: str,
    text2: str,
    alpha: float = 0.85,
) -> float:
    """Combines dense vector cosine similarity with sparse lexical Jaccard overlap."""
    cos_sim = cosine_similarity(vec1, vec2)
    lex_sim = token_jaccard_similarity(text1, text2)
    # Cosine is in [-1, 1], normalize to [0, 1] for blending
    cos_norm = max(0.0, cos_sim)
    return float(alpha * cos_norm + (1.0 - alpha) * lex_sim)
