"""Tests for HTML dashboard visualizer."""

import os
from cachemind.config import CacheEntry
from cachemind.visualizer import generate_dashboard_html


def test_generate_dashboard_html(tmp_path):
    stats = {
        "total_queries": 100,
        "total_hits": 82,
        "exact_hits": 50,
        "semantic_hits": 32,
        "misses": 18,
        "hit_rate_pct": 82.0,
        "exact_hit_rate_pct": 50.0,
        "semantic_hit_rate_pct": 32.0,
        "tokens_saved": {"total_tokens": 12500},
        "estimated_cost_saved_usd": 0.0525,
        "pricing_tier": "gemini-2.0-flash",
        "latency_ms": {
            "overall": {"p50": 0.12, "mean": 0.45},
            "l1_exact": {"mean": 0.08},
            "l2_semantic": {"mean": 1.25},
        },
    }

    entries = [
        CacheEntry(
            key="k1",
            prompt="How to reset password?",
            response="Click on forgot password link.",
            access_count=5,
            prompt_tokens=10,
            completion_tokens=20,
        ),
        CacheEntry(
            key="k2",
            prompt="What is refund window?",
            response="30 days from date of purchase.",
            access_count=3,
            prompt_tokens=12,
            completion_tokens=15,
        ),
    ]

    out_file = str(tmp_path / "dashboard.html")
    res_path = generate_dashboard_html(stats=stats, entries=entries, output_path=out_file)

    assert os.path.exists(res_path)
    with open(res_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "<!DOCTYPE html>" in content
    assert "CacheMind-AI Observability Hub" in content
    assert "82.0%" in content
    assert "12,500" in content
    assert "How to reset password?" in content
    assert "simulateQuery" in content
