"""Tests for CacheMind-AI CLI commands."""

from click.testing import CliRunner
from cachemind.cli import main


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "CacheMind-AI" in result.output


def test_cli_stats_empty():
    runner = CliRunner()
    result = runner.invoke(main, ["stats"])
    assert result.exit_code == 0
    assert "CACHEMIND-AI REAL-TIME OBSERVABILITY STATS" in result.output
    assert "Active Entries:" in result.output


def test_cli_set_and_query_flow(tmp_path):
    db_file = str(tmp_path / "cli_test.db")
    runner = CliRunner()

    # 1. Set entry
    set_res = runner.invoke(main, [
        "set",
        "How do I cancel my plan?",
        "Go to Account settings to cancel.",
        "--sqlite",
        "--db-path",
        db_file,
    ])
    assert set_res.exit_code == 0
    assert "Successfully cached entry" in set_res.output

    # 2. Inspect entry
    insp_res = runner.invoke(main, ["inspect", "--sqlite", "--db-path", db_file])
    assert insp_res.exit_code == 0
    assert "Total Active Entries: 1" in insp_res.output
    assert "How do I cancel my plan?" in insp_res.output

    # 3. Query exact match
    query_exact = runner.invoke(main, [
        "query",
        "How do I cancel my plan?",
        "--sqlite",
        "--db-path",
        db_file,
    ])
    assert query_exact.exit_code == 0
    assert "[HIT - EXACT]" in query_exact.output

    # 4. Query semantic paraphrase
    query_semantic = runner.invoke(main, [
        "query",
        "Can I cancel my plan?",
        "--sqlite",
        "--db-path",
        db_file,
        "--threshold",
        "0.65",
    ])
    assert query_semantic.exit_code == 0
    assert "[HIT - SEMANTIC]" in query_semantic.output

    # 5. Clear entries
    clear_res = runner.invoke(main, ["clear", "--sqlite", "--db-path", db_file])
    assert clear_res.exit_code == 0
    assert "Cleared cache entries" in clear_res.output


def test_cli_benchmark():
    runner = CliRunner()
    result = runner.invoke(main, ["benchmark", "--queries", "8"])
    assert result.exit_code == 0
    assert "RUNNING CACHEMIND BENCHMARK" in result.output
    assert "Hit Rate:" in result.output


def test_cli_dashboard(tmp_path):
    out_html = str(tmp_path / "test_dash.html")
    runner = CliRunner()
    result = runner.invoke(main, ["dashboard", "--output", out_html])
    assert result.exit_code == 0
    assert "Executive dashboard generated successfully" in result.output
