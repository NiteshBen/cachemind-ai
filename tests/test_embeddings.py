"""Tests for embedding providers."""

import numpy as np
import pytest
from cachemind.embeddings.local_tfidf import LocalTFIDFEmbedder
from cachemind.embeddings.gemini import GeminiEmbeddingProvider
from cachemind.similarity.metrics import cosine_similarity


def test_tfidf_embedder_dimension():
    embedder = LocalTFIDFEmbedder(dim=256)
    assert embedder.dimension == 256
    vec = embedder.embed_query("Hello world")
    assert len(vec) == 256


def test_tfidf_embedder_normalization():
    embedder = LocalTFIDFEmbedder(dim=384)
    vec = embedder.embed_query("Semantic caching in Python for AI agents")
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_tfidf_embedder_empty_string():
    embedder = LocalTFIDFEmbedder(dim=128)
    vec = embedder.embed_query("")
    assert len(vec) == 128
    assert all(v == 0.0 for v in vec)

    vec_space = embedder.embed_query("   ")
    assert all(v == 0.0 for v in vec_space)


def test_tfidf_embedder_batch():
    embedder = LocalTFIDFEmbedder(dim=128)
    texts = ["How do I cancel my order?", "Can I refund my purchase?"]
    batch_vecs = embedder.embed_batch(texts)
    assert len(batch_vecs) == 2
    assert len(batch_vecs[0]) == 128
    assert len(batch_vecs[1]) == 128


def test_tfidf_semantic_similarity_paraphrase():
    embedder = LocalTFIDFEmbedder(dim=384)
    q1 = embedder.embed_query("How do I cancel my subscription?")
    q2 = embedder.embed_query("Can I cancel my subscription?")
    q3 = embedder.embed_query("What is the recipe for chocolate brownies?")

    sim_related = cosine_similarity(q1, q2)
    sim_unrelated = cosine_similarity(q1, q3)

    assert sim_related > 0.70
    assert sim_unrelated < 0.30
    assert sim_related > sim_unrelated


def test_gemini_embedder_fallback_without_key():
    embedder = GeminiEmbeddingProvider(api_key=None, dimension=256)
    vec = embedder.embed_query("Testing Gemini fallback")
    assert len(vec) == 256
    assert pytest.approx(np.linalg.norm(vec), rel=1e-3) == 1.0
