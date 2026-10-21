"""Tests for metrics tracking and pricing calculations."""

import pytest
from cachemind.metrics.pricing import PRICING_TABLE, calculate_token_cost
from cachemind.metrics.tracker import MetricsTracker


def test_calculate_token_cost():
    # 1,000,000 prompt tokens + 1,000,000 completion tokens on gemini-2.0-flash
    cost = calculate_token_cost(1_000_000, 1_000_000, model="gemini-2.0-flash")
    # 0.10 + 0.40 = 0.50
    assert pytest.approx(cost, rel=1e-3) == 0.50

    cost_gpt4o = calculate_token_cost(100_000, 50_000, model="gpt-4o")
    # (0.1 * 2.50) + (0.05 * 10.00) = 0.25 + 0.50 = 0.75
    assert pytest.approx(cost_gpt4o, rel=1e-3) == 0.75


def test_metrics_tracker_hits_and_misses():
    tracker = MetricsTracker(default_pricing_model="gemini-2.0-flash")

    # Record 2 exact hits
    tracker.record_hit(hit_type="exact", latency_ms=0.08, prompt_tokens=100, completion_tokens=200)
    tracker.record_hit(hit_type="exact", latency_ms=0.12, prompt_tokens=100, completion_tokens=200)

    # Record 1 semantic hit
    tracker.record_hit(hit_type="semantic", latency_ms=1.45, prompt_tokens=150, completion_tokens=250)

    # Record 1 miss
    tracker.record_miss(latency_ms=1850.0)

    assert tracker.total_queries == 4
    assert tracker.total_hits == 3
    assert tracker.hit_rate == 0.75
    assert tracker.exact_hit_rate == 0.50
    assert tracker.semantic_hit_rate == 0.25

    # Check tokens saved
    total_toks = (100 + 100 + 150) + (200 + 200 + 250)
    assert tracker._tokens_saved_prompt + tracker._tokens_saved_completion == total_toks

    # Summary dict
    summ = tracker.summary()
    assert summ["hit_rate_pct"] == 75.0
    assert summ["total_queries"] == 4
    assert summ["tokens_saved"]["total_tokens"] == total_toks
    assert summ["estimated_cost_saved_usd"] > 0.0

    # Latencies percentiles
    p_exact = tracker.get_latency_percentiles("l1")
    assert p_exact["mean"] < 0.2
    p_sem = tracker.get_latency_percentiles("l2")
    assert p_sem["mean"] > 1.0


def test_metrics_tracker_reset():
    tracker = MetricsTracker()
    tracker.record_hit("exact", 0.05, 10, 20)
    tracker.record_miss(1000.0)
    assert tracker.total_queries == 2

    tracker.reset()
    assert tracker.total_queries == 0
    assert tracker.total_hits == 0
    assert tracker.hit_rate == 0.0
