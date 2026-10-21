"""End-to-end tests for the core SemanticCache engine."""

import time
import pytest
from cachemind.config import CacheConfig
from cachemind.core import SemanticCache


def test_exact_l1_hit():
    cache = SemanticCache()
    prompt = "What is the capital of France?"
    response = "The capital of France is Paris."

    cache.set(prompt=prompt, response=response, prompt_tokens=10, completion_tokens=15)

    lookup = cache.get(prompt)
    assert lookup.hit is True
    assert lookup.hit_type == "exact"
    assert lookup.similarity == 1.0
    assert lookup.response == response
    assert lookup.matched_prompt == prompt
    assert lookup.latency_ms < 1.0  # Ultra-fast sub-millisecond
    assert lookup.tokens_saved == 25
    assert lookup.cost_saved_usd > 0.0


def test_semantic_l2_hit_paraphrase():
    # Lower threshold slightly to allow robust n-gram paraphrase matching
    cfg = CacheConfig(similarity_threshold=0.72)
    cache = SemanticCache(config=cfg)

    stored_prompt = "How do I cancel my monthly subscription?"
    stored_response = "Navigate to Billing Settings and click Cancel Subscription."
    cache.set(stored_prompt, stored_response)

    paraphrased_query = "Can I cancel my subscription monthly?"
    lookup = cache.get(paraphrased_query)

    assert lookup.hit is True
    assert lookup.hit_type == "semantic"
    assert lookup.similarity >= 0.72
    assert lookup.response == stored_response
    assert lookup.matched_prompt == stored_prompt
    assert lookup.latency_ms < 10.0


def test_cache_miss_unrelated():
    cache = SemanticCache(config=CacheConfig(similarity_threshold=0.85))
    cache.set("What is Python programming language?", "Python is a high-level interpreted programming language.")

    lookup = cache.get("How to bake sourdough bread at home?")
    assert lookup.hit is False
    assert lookup.response is None
    assert lookup.similarity == 0.0


def test_call_or_cache_wrapper():
    cache = SemanticCache()
    call_count = 0

    def mock_llm_api(query: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"LLM Response to: {query}"

    # 1. First invocation -> cache miss -> calls LLM
    ans1, lookup1 = cache.call_or_cache("Tell me a joke", mock_llm_api)
    assert call_count == 1
    assert lookup1.hit is False
    assert ans1 == "LLM Response to: Tell me a joke"

    # 2. Second invocation -> cache hit -> does NOT call LLM
    ans2, lookup2 = cache.call_or_cache("Tell me a joke", mock_llm_api)
    assert call_count == 1
    assert lookup2.hit is True
    assert lookup2.hit_type == "exact"
    assert ans2 == "LLM Response to: Tell me a joke"


def test_namespace_isolation():
    cache = SemanticCache()
    prompt = "How do I change my billing address?"

    # Cache under namespace "finance"
    cache.set(prompt, "Visit finance.example.com", namespace="finance")

    # Query from "support" namespace should miss
    lookup_support = cache.get(prompt, namespace="support")
    assert lookup_support.hit is False

    # Query from "finance" namespace should hit
    lookup_finance = cache.get(prompt, namespace="finance")
    assert lookup_finance.hit is True
    assert lookup_finance.response == "Visit finance.example.com"


def test_sqlite_backend_persistence(tmp_path):
    db_file = str(tmp_path / "sqlite_cache.db")
    cfg = CacheConfig(storage_backend="sqlite", sqlite_path=db_file, similarity_threshold=0.75)
    cache = SemanticCache(config=cfg)

    cache.set("Where is the user documentation?", "Documentation is at docs.example.com")
    st = cache.stats()
    assert st["active_entries"] == 1

    # Create new cache pointing to same database
    cache2 = SemanticCache(config=cfg)
    lookup = cache2.get("Where is the user documentation?")
    assert lookup.hit is True
    assert lookup.response == "Documentation is at docs.example.com"


def test_clear_and_stats():
    cache = SemanticCache()
    cache.set("Query 1", "Response 1")
    cache.set("Query 2", "Response 2")
    assert cache.stats()["active_entries"] == 2

    cache.clear()
    assert cache.stats()["active_entries"] == 0
    assert cache.get("Query 1").hit is False
