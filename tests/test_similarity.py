"""Tests for vector similarity metrics and VectorIndex."""

import numpy as np
import pytest
from cachemind.similarity.metrics import (
    cosine_similarity,
    batch_cosine_similarity,
    token_jaccard_similarity,
    hybrid_similarity,
)
from cachemind.similarity.index import VectorIndex


def test_cosine_similarity_identical():
    v = [0.6, 0.8]
    assert pytest.approx(cosine_similarity(v, v), rel=1e-3) == 1.0


def test_cosine_similarity_orthogonal():
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    assert pytest.approx(cosine_similarity(v1, v2), abs=1e-5) == 0.0


def test_cosine_similarity_opposite():
    v1 = [1.0, 0.0]
    v2 = [-1.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v2), rel=1e-3) == -1.0


def test_cosine_similarity_zero_vector():
    v1 = [0.0, 0.0]
    v2 = [1.0, 2.0]
    assert cosine_similarity(v1, v2) == 0.0


def test_batch_cosine_similarity():
    matrix = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.7071, 0.7071],
    ], dtype=np.float32)
    q = [1.0, 0.0]
    scores = batch_cosine_similarity(q, matrix)
    assert len(scores) == 3
    assert pytest.approx(scores[0], rel=1e-3) == 1.0
    assert pytest.approx(scores[1], abs=1e-5) == 0.0
    assert pytest.approx(scores[2], rel=1e-2) == 0.7071


def test_token_jaccard_similarity():
    t1 = "How do I cancel my order"
    t2 = "Can I cancel my order please"
    sim = token_jaccard_similarity(t1, t2)
    assert 0.4 <= sim <= 0.8

    assert token_jaccard_similarity("hello world", "hello world") == 1.0
    assert token_jaccard_similarity("alpha beta", "gamma delta") == 0.0
    assert token_jaccard_similarity("", "something") == 0.0


def test_hybrid_similarity():
    v1 = [1.0, 0.0]
    v2 = [0.7071, 0.7071]
    t1 = "cancel subscription"
    t2 = "cancel order subscription"
    h_sim = hybrid_similarity(v1, v2, t1, t2, alpha=0.8)
    assert 0.0 < h_sim <= 1.0


def test_vector_index_lifecycle():
    idx = VectorIndex(dimension=4)
    assert len(idx) == 0

    idx.add("key1", [1.0, 0.0, 0.0, 0.0])
    idx.add("key2", [0.0, 1.0, 0.0, 0.0])
    idx.add("key3", [0.7071, 0.7071, 0.0, 0.0])

    assert len(idx) == 3

    # Search closest to key1
    matches = idx.search([1.0, 0.0, 0.0, 0.0], top_k=2, threshold=0.5)
    assert len(matches) == 2
    assert matches[0][0] == "key1"
    assert pytest.approx(matches[0][1], rel=1e-3) == 1.0
    assert matches[1][0] == "key3"

    # Search with high threshold
    matches_high = idx.search([1.0, 0.0, 0.0, 0.0], top_k=5, threshold=0.99)
    assert len(matches_high) == 1
    assert matches_high[0][0] == "key1"

    # Remove key
    assert idx.remove("key1") is True
    assert len(idx) == 2
    assert idx.remove("nonexistent") is False

    # Clear
    idx.clear()
    assert len(idx) == 0
    assert idx.search([1.0, 0.0, 0.0, 0.0]) == []
