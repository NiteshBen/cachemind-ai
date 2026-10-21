"""Pricing registry and token cost arbitrage calculator for modern LLMs."""

from typing import Dict, Tuple

# Pricing per 1,000,000 tokens: (prompt_usd, completion_usd)
PRICING_TABLE: Dict[str, Tuple[float, float]] = {
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-pro": (1.25, 5.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    "llama-3.3-70b": (0.35, 0.40),
}


def calculate_token_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str = "gemini-2.0-flash",
) -> float:
    """Calculates dollar cost for consumed tokens based on model tier."""
    pricing = PRICING_TABLE.get(model.lower(), PRICING_TABLE["gemini-2.0-flash"])
    prompt_cost = (prompt_tokens / 1_000_000.0) * pricing[0]
    completion_cost = (completion_tokens / 1_000_000.0) * pricing[1]
    return float(prompt_cost + completion_cost)
