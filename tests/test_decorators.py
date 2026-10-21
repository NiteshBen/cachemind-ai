"""Tests for function caching decorator."""

import pytest
from cachemind.core import SemanticCache
from cachemind.decorators import cached


def test_bare_cached_decorator():
    call_count = 0

    @cached
    def ask_ai(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"Echo: {prompt}"

    # First call: misses and calls ask_ai
    r1 = ask_ai("Explain quantum computing")
    assert call_count == 1
    assert r1 == "Echo: Explain quantum computing"

    # Second call: hits cache
    r2 = ask_ai("Explain quantum computing")
    assert call_count == 1
    assert r2 == "Echo: Explain quantum computing"


def test_parameterized_cached_decorator():
    custom_cache = SemanticCache()
    call_count = 0

    @cached(cache=custom_cache, threshold=0.70, namespace="test_ns")
    def lookup_weather(query: str) -> str:
        nonlocal call_count
        call_count += 1
        return "Sunny and 24C"

    # Initial query
    ans1 = lookup_weather("What is the weather today?")
    assert call_count == 1
    assert ans1 == "Sunny and 24C"

    # Paraphrased query hitting semantic cache
    ans2 = lookup_weather("What is today's weather?")
    assert call_count == 1
    assert ans2 == "Sunny and 24C"


def test_cached_bypass_non_string():
    call_count = 0

    @cached
    def compute_sum(a: int, b: int) -> int:
        nonlocal call_count
        call_count += 1
        return a + b

    res1 = compute_sum(5, 10)
    assert res1 == 15
    assert call_count == 1

    res2 = compute_sum(5, 10)
    assert res2 == 15
    assert call_count == 2  # bypassed because no string prompt
