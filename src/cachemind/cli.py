"""Command-line interface for CacheMind-AI management, benchmarking, and telemetry."""

import json
import time
import click

from cachemind.config import CacheConfig
from cachemind.core import SemanticCache


def _get_cache(sqlite: bool = False, db_path: str = "cachemind_store.db") -> SemanticCache:
    backend = "sqlite" if sqlite else "memory"
    cfg = CacheConfig(storage_backend=backend, sqlite_path=db_path)
    return SemanticCache(config=cfg)


@click.group()
@click.version_option(version="0.1.0", prog_name="CacheMind-AI")
def main():
    """CacheMind-AI: Production Semantic Caching & Cost Optimization Engine."""
    pass


@main.command()
@click.option("--sqlite", is_flag=True, help="Use persistent SQLite storage backend.")
@click.option("--db-path", default="cachemind_store.db", help="Path to SQLite database.")
@click.option("--model", default="gemini-2.0-flash", help="LLM pricing tier for cost calculations.")
def stats(sqlite: bool, db_path: str, model: str):
    """Display real-time cache telemetry, hit rates, and economic savings."""
    cache = _get_cache(sqlite, db_path)
    s = cache.stats(model=model)
    click.echo("=" * 60)
    click.echo("  CACHEMIND-AI REAL-TIME OBSERVABILITY STATS")
    click.echo("=" * 60)
    click.echo(f"Active Entries:       {s['active_entries']}")
    click.echo(f"Storage Backend:      {s['storage_backend']}")
    click.echo(f"Similarity Threshold: {s['similarity_threshold']}")
    click.echo(f"Total Queries:        {s['total_queries']}")
    click.echo(f"Total Hits:           {s['total_hits']} ({s['hit_rate_pct']}%)")
    click.echo(f"  - Exact (L1):       {s['exact_hits']} ({s['exact_hit_rate_pct']}%)")
    click.echo(f"  - Semantic (L2):    {s['semantic_hits']} ({s['semantic_hit_rate_pct']}%)")
    click.echo(f"Misses:               {s['misses']}")
    click.echo(f"Tokens Saved:         {s['tokens_saved']['total_tokens']:,}")
    click.echo(f"Est. Cost Saved:      ${s['estimated_cost_saved_usd']:.4f} USD ({model})")
    click.echo("=" * 60)


@main.command()
@click.argument("query_text")
@click.option("--sqlite", is_flag=True, help="Use persistent SQLite storage.")
@click.option("--db-path", default="cachemind_store.db", help="Path to SQLite database.")
@click.option("--threshold", default=0.85, type=float, help="Similarity threshold.")
def query(query_text: str, sqlite: bool, db_path: str, threshold: float):
    """Query cache with exact L1 lookup and L2 semantic similarity search."""
    cache = _get_cache(sqlite, db_path)
    cache.config.similarity_threshold = threshold
    res = cache.get(query_text)

    if res.hit:
        click.secho(f"\n[HIT - {res.hit_type.upper()}] Similarity: {res.similarity:.4f} ({res.latency_ms:.2f}ms)", fg="green", bold=True)
        click.echo(f"Matched Prompt: {res.matched_prompt}")
        click.echo(f"Cached Response:\n{res.response}")
        click.echo(f"Tokens Saved:   {res.tokens_saved} | Est. Cost Saved: ${res.cost_saved_usd:.6f}")
    else:
        click.secho(f"\n[MISS] ({res.latency_ms:.2f}ms) No cache entry met threshold {threshold}", fg="yellow")


@main.command(name="set")
@click.argument("prompt")
@click.argument("response")
@click.option("--sqlite", is_flag=True, help="Use persistent SQLite storage.")
@click.option("--db-path", default="cachemind_store.db", help="Path to SQLite database.")
@click.option("--namespace", default="default", help="Namespace partition.")
def set_entry(prompt: str, response: str, sqlite: bool, db_path: str, namespace: str):
    """Store a prompt and response pair into the cache."""
    cache = _get_cache(sqlite, db_path)
    entry = cache.set(prompt=prompt, response=response, namespace=namespace)
    click.secho(f"Successfully cached entry '{entry.key[:12]}...' [{namespace}]", fg="green")


@main.command()
@click.option("--sqlite", is_flag=True, help="Use persistent SQLite storage.")
@click.option("--db-path", default="cachemind_store.db", help="Path to SQLite database.")
@click.option("--limit", default=20, type=int, help="Maximum entries to display.")
def inspect(sqlite: bool, db_path: str, limit: int):
    """List active cached prompts and response snippets."""
    cache = _get_cache(sqlite, db_path)
    entries = cache.storage.all_entries(namespace=cache.config.namespace)
    click.echo(f"\nTotal Active Entries: {len(entries)}")
    click.echo("-" * 75)
    for i, e in enumerate(entries[:limit]):
        preview_resp = e.response[:60] + "..." if len(e.response) > 60 else e.response
        click.echo(f"[{i+1}] Prompt:   {e.prompt}")
        click.echo(f"    Response: {preview_resp}")
        click.echo(f"    Hits: {e.access_count} | Tokens: {e.prompt_tokens + e.completion_tokens} | Key: {e.key[:10]}...")
        click.echo("-" * 75)


@main.command()
@click.option("--sqlite", is_flag=True, help="Use persistent SQLite storage.")
@click.option("--db-path", default="cachemind_store.db", help="Path to SQLite database.")
@click.option("--namespace", default=None, help="Specific namespace to clear.")
def clear(sqlite: bool, db_path: str, namespace: str):
    """Purge all or namespaced cache entries."""
    cache = _get_cache(sqlite, db_path)
    cache.clear(namespace=namespace)
    target = f"in namespace '{namespace}'" if namespace else "all namespaces"
    click.secho(f"Cleared cache entries {target}.", fg="red")


@main.command()
@click.option("--sqlite", is_flag=True, help="Use persistent SQLite storage.")
@click.option("--db-path", default="cachemind_store.db", help="Path to SQLite database.")
@click.option("--output", default="cachemind_dashboard.html", help="Output HTML file path.")
def dashboard(sqlite: bool, db_path: str, output: str):
    """Export the standalone interactive HTML analytics dashboard."""
    cache = _get_cache(sqlite, db_path)
    out_file = cache.export_dashboard(output_path=output)
    click.secho(f"Executive dashboard generated successfully at: {out_file}", fg="cyan", bold=True)


@main.command()
@click.option("--queries", default=20, help="Number of benchmark iterations.")
def benchmark(queries: int):
    """Run an automated performance benchmark simulating realistic agent traffic."""
    cache = SemanticCache(config=CacheConfig(similarity_threshold=0.80))
    click.echo("=" * 65)
    click.echo(f"  RUNNING CACHEMIND BENCHMARK ({queries} queries)")
    click.echo("=" * 65)

    # Seed FAQ corpus
    seed_data = [
        ("How do I cancel my subscription?", "To cancel, go to Settings > Billing and click Cancel Subscription."),
        ("What is the company refund policy?", "Full refunds are granted within 30 days of purchase."),
        ("Where can I reset my password?", "Click 'Forgot Password' on the login screen to receive a reset link."),
        ("How to install CacheMind?", "Run pip install cachemind-ai or clone the GitHub repo."),
    ]
    for p, r in seed_data:
        cache.set(p, r, prompt_tokens=25, completion_tokens=40)

    # Test synthetic variations (exact, paraphrased, misses)
    test_queries = [
        ("How do I cancel my subscription?", "exact"),
        ("Can I cancel my subscription?", "semantic"),
        ("What is the company refund policy?", "exact"),
        ("Can I get a refund on my purchase?", "semantic"),
        ("Where can I reset my password?", "exact"),
        ("I forgot my password how do I reset it?", "semantic"),
        ("What is quantum computing?", "miss"),
        ("How to make chocolate cake?", "miss"),
    ]

    t0 = time.perf_counter()
    for i in range(queries):
        q_text, expected = test_queries[i % len(test_queries)]
        cache.get(q_text)
    total_time = (time.perf_counter() - t0) * 1000.0

    st = cache.stats()
    click.echo(f"Benchmark completed in {total_time:.2f} ms")
    click.echo(f"Queries / Sec:       {queries / (total_time / 1000.0):.1f} qps")
    click.echo(f"Hit Rate:            {st['hit_rate_pct']}%")
    click.echo(f"  - Exact Hits:      {st['exact_hits']}")
    click.echo(f"  - Semantic Hits:   {st['semantic_hits']}")
    click.echo(f"  - Misses:          {st['misses']}")
    click.echo(f"L1 Mean Latency:     {st['latency_ms']['l1_exact']['mean']:.3f} ms")
    click.echo(f"L2 Mean Latency:     {st['latency_ms']['l2_semantic']['mean']:.3f} ms")
    click.echo(f"Tokens Saved:        {st['tokens_saved']['total_tokens']:,}")
    click.echo(f"Cost Saved (Gemini): ${st['estimated_cost_saved_usd']:.5f}")
    click.echo("=" * 65)


if __name__ == "__main__":
    main()
