"""Decorator utilities for transparent function and LLM semantic caching."""

from functools import wraps
from typing import Any, Callable, Optional, Union
from cachemind.core import SemanticCache

_GLOBAL_DEFAULT_CACHE: Optional[SemanticCache] = None


def get_default_cache() -> SemanticCache:
    """Returns singleton global SemanticCache instance."""
    global _GLOBAL_DEFAULT_CACHE
    if _GLOBAL_DEFAULT_CACHE is None:
        _GLOBAL_DEFAULT_CACHE = SemanticCache()
    return _GLOBAL_DEFAULT_CACHE


def set_default_cache(cache: SemanticCache) -> None:
    """Sets singleton global SemanticCache instance."""
    global _GLOBAL_DEFAULT_CACHE
    _GLOBAL_DEFAULT_CACHE = cache


def cached(
    func: Optional[Callable] = None,
    *,
    cache: Optional[SemanticCache] = None,
    threshold: Optional[float] = None,
    namespace: Optional[str] = None,
    ttl_seconds: Optional[int] = None,
):
    """Decorator to semantically cache LLM and text-processing function responses.

    Can be applied bare:
        @cached
        def call_gemini(prompt: str) -> str:
            ...

    Or with parameters:
        @cached(threshold=0.85, namespace="customer_support")
        def answer_support(query: str) -> str:
            ...
    """

    def decorator(fn: Callable) -> Callable:
        target_cache = cache or get_default_cache()

        @wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            # Extract prompt argument: inspect kwargs or first positional arg
            prompt: Optional[str] = None
            if "prompt" in kwargs and isinstance(kwargs["prompt"], str):
                prompt = kwargs["prompt"]
            elif "query" in kwargs and isinstance(kwargs["query"], str):
                prompt = kwargs["query"]
            elif "text" in kwargs and isinstance(kwargs["text"], str):
                prompt = kwargs["text"]
            elif len(args) > 0 and isinstance(args[0], str):
                prompt = args[0]
            elif len(args) > 1 and isinstance(args[1], str):
                # e.g., self.method(prompt)
                prompt = args[1]

            # If no string prompt could be extracted, bypass cache
            if prompt is None:
                target_cache.metrics.record_bypass()
                return fn(*args, **kwargs)

            # Override threshold temporarily if specified
            original_thresh = target_cache.config.similarity_threshold
            if threshold is not None:
                target_cache.config.similarity_threshold = threshold

            try:
                res, lookup = target_cache.call_or_cache(
                    prompt=prompt,
                    llm_callable=lambda p: fn(*args, **kwargs),
                    namespace=namespace,
                    ttl_seconds=ttl_seconds,
                )
                return res
            finally:
                if threshold is not None:
                    target_cache.config.similarity_threshold = original_thresh

        # Attach cache reference to the wrapped function for inspection
        wrapper.__cache__ = target_cache
        return wrapper

    if func is not None and callable(func):
        return decorator(func)
    return decorator
