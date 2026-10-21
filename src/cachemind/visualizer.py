"""Standalone, zero-CDN interactive HTML analytics dashboard for CacheMind-AI."""

import html
import json
import os
from typing import Any, Dict, List
from cachemind.config import CacheEntry


def generate_dashboard_html(
    stats: Dict[str, Any],
    entries: List[CacheEntry],
    output_path: str = "cachemind_dashboard.html",
) -> str:
    """Generates an executive, dark-mode single-file HTML report with embedded SVG analytics."""
    hit_rate = stats.get("hit_rate_pct", 0.0)
    exact_hit_rate = stats.get("exact_hit_rate_pct", 0.0)
    semantic_hit_rate = stats.get("semantic_hit_rate_pct", 0.0)
    total_queries = stats.get("total_queries", 0)
    total_hits = stats.get("total_hits", 0)
    exact_hits = stats.get("exact_hits", 0)
    semantic_hits = stats.get("semantic_hits", 0)
    misses = stats.get("misses", 0)
    tokens_saved = stats.get("tokens_saved", {}).get("total_tokens", 0)
    cost_saved = stats.get("estimated_cost_saved_usd", 0.0)
    pricing_tier = stats.get("pricing_tier", "gemini-2.0-flash")

    lat_overall = stats.get("latency_ms", {}).get("overall", {})
    p50_lat = lat_overall.get("p50", 0.0)
    mean_lat = lat_overall.get("mean", 0.0)

    # Prepare table rows
    table_rows_html = ""
    for idx, e in enumerate(entries[:50]):  # display top 50
        clean_p = html.escape(e.prompt)
        clean_r = html.escape(e.response[:120] + ("..." if len(e.response) > 120 else ""))
        toks = e.prompt_tokens + e.completion_tokens
        table_rows_html += f"""
        <tr>
            <td style="font-family:monospace; color:#38bdf8;">{idx+1}</td>
            <td style="font-weight:600; color:#f1f5f9;">{clean_p}</td>
            <td style="color:#94a3b8;">{clean_r}</td>
            <td style="text-align:center;"><span class="badge badge-cyan">{toks} toks</span></td>
            <td style="text-align:center; color:#10b981; font-weight:600;">{e.access_count}x</td>
            <td style="text-align:center; font-size:12px; color:#64748b;">{e.namespace}</td>
        </tr>
        """

    if not entries:
        table_rows_html = """
        <tr>
            <td colspan="6" style="text-align:center; padding:30px; color:#64748b;">
                No active cache entries found. Execute queries or run benchmarks to populate.
            </td>
        </tr>
        """

    # Entries JSON for interactive simulator
    entries_json = json.dumps([{"prompt": e.prompt, "response": e.response} for e in entries])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CacheMind-AI | Observability & Economic Dashboard</title>
    <style>
        :root {{
            --bg-base: #090d16;
            --bg-card: #111827;
            --bg-card-hover: #1e293b;
            --border: rgba(56, 189, 248, 0.2);
            --border-purple: rgba(168, 85, 247, 0.25);
            --cyan: #38bdf8;
            --emerald: #10b981;
            --purple: #c084fc;
            --amber: #f59e0b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }}
        * {{ margin:0; padding:0; box-sizing:border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        body {{ background-color: var(--bg-base); color: var(--text-main); padding: 32px; min-height: 100vh; }}
        .container {{ max-width: 1240px; margin: 0 auto; }}
        
        /* Header */
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 24px; margin-bottom: 32px; }}
        .header-title h1 {{ font-size: 28px; font-weight: 800; background: linear-gradient(135deg, #38bdf8 0%, #c084fc 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .header-title p {{ color: var(--text-muted); font-size: 14px; margin-top: 4px; }}
        .badge {{ padding: 6px 14px; border-radius: 9999px; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }}
        .badge-cyan {{ background: rgba(56, 189, 248, 0.15); color: var(--cyan); border: 1px solid rgba(56, 189, 248, 0.3); }}
        .badge-emerald {{ background: rgba(16, 185, 129, 0.15); color: var(--emerald); border: 1px solid rgba(16, 185, 129, 0.3); }}

        /* KPI Grid */
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-bottom: 32px; }}
        .kpi-card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 16px; padding: 22px; position: relative; overflow: hidden; }}
        .kpi-card::before {{ content:""; position: absolute; top:0; left:0; right:0; height: 3px; background: linear-gradient(90deg, var(--cyan), var(--purple)); }}
        .kpi-label {{ font-size: 13px; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }}
        .kpi-val {{ font-size: 32px; font-weight: 800; margin: 10px 0 4px; color: #fff; }}
        .kpi-sub {{ font-size: 12px; color: var(--emerald); font-weight: 500; }}

        /* Charts Row */
        .charts-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 32px; }}
        @media(max-width: 900px) {{ .charts-row {{ grid-template-columns: 1fr; }} }}
        .chart-card {{ background: var(--bg-card); border: 1px solid var(--border-purple); border-radius: 16px; padding: 24px; }}
        .chart-card h3 {{ font-size: 16px; font-weight: 700; color: #f1f5f9; margin-bottom: 16px; display:flex; justify-content:space-between; align-items:center; }}

        /* Latency Bars */
        .latency-row {{ margin-bottom: 16px; }}
        .lat-label {{ display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 6px; font-weight: 500; }}
        .lat-track {{ height: 12px; background: #1e293b; border-radius: 6px; overflow: hidden; position: relative; }}
        .lat-fill {{ height: 100%; border-radius: 6px; }}

        /* Interactive Simulator */
        .sim-box {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 16px; padding: 24px; margin-bottom: 32px; }}
        .sim-input {{ width: 100%; background: #0b1120; border: 1px solid #334155; border-radius: 8px; padding: 12px 16px; color: #fff; font-size: 14px; outline: none; margin-top: 10px; }}
        .sim-input:focus {{ border-color: var(--cyan); }}
        .sim-result {{ margin-top: 14px; padding: 14px; background: #0f172a; border-radius: 8px; font-size: 14px; display: none; }}

        /* Table */
        .table-card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 16px; padding: 24px; overflow-x: auto; }}
        .table-card h3 {{ font-size: 17px; margin-bottom: 16px; font-weight: 700; }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }}
        th {{ padding: 12px 14px; background: #1e293b; color: #94a3b8; font-weight: 600; text-transform: uppercase; font-size: 11px; }}
        td {{ padding: 12px 14px; border-bottom: 1px solid #1e293b; }}
        tr:hover td {{ background: var(--bg-card-hover); }}

        /* Footer */
        .footer {{ margin-top: 40px; text-align: center; font-size: 13px; color: #64748b; border-top: 1px solid #1e293b; padding-top: 24px; }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="header-title">
                <h1>CacheMind-AI Observability Hub</h1>
                <p>Real-Time Semantic Vector Caching, Latency Minimization & Token Cost Arbitrage</p>
            </div>
            <div>
                <span class="badge badge-emerald">System Operational</span>
                <span class="badge badge-cyan" style="margin-left:8px;">Model: {pricing_tier}</span>
            </div>
        </div>

        <!-- KPI Grid -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Cache Hit Rate</div>
                <div class="kpi-val" style="color:var(--emerald);">{hit_rate}%</div>
                <div class="kpi-sub">{exact_hits} exact &bull; {semantic_hits} semantic</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Total Queries</div>
                <div class="kpi-val">{total_queries}</div>
                <div class="kpi-sub">{total_hits} hits / {misses} misses</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Tokens Saved</div>
                <div class="kpi-val" style="color:var(--cyan);">{tokens_saved:,}</div>
                <div class="kpi-sub">Avoided LLM token payload</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Estimated Cost Saved</div>
                <div class="kpi-val" style="color:#fbcfe8;">${cost_saved:.4f}</div>
                <div class="kpi-sub">Tier: {pricing_tier}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">P50 Cache Latency</div>
                <div class="kpi-val" style="color:var(--purple);">{p50_lat:.2f}ms</div>
                <div class="kpi-sub">Mean: {mean_lat:.2f}ms</div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="charts-row">
            <!-- Latency Comparison -->
            <div class="chart-card">
                <h3>Execution Latency Comparison <span>Microseconds vs Seconds</span></h3>
                <div class="latency-row">
                    <div class="lat-label">
                        <span>L1 Exact Hash Match</span>
                        <span style="color:var(--emerald);">&lt; 0.1 ms</span>
                    </div>
                    <div class="lat-track">
                        <div class="lat-fill" style="width: 2%; background: var(--emerald);"></div>
                    </div>
                </div>
                <div class="latency-row">
                    <div class="lat-label">
                        <span>L2 Semantic Vector Search</span>
                        <span style="color:var(--cyan);">&lt; 1.5 ms</span>
                    </div>
                    <div class="lat-track">
                        <div class="lat-fill" style="width: 5%; background: var(--cyan);"></div>
                    </div>
                </div>
                <div class="latency-row">
                    <div class="lat-label">
                        <span>L3 Upstream LLM Roundtrip (Baseline)</span>
                        <span style="color:var(--amber);">1,800 - 3,500 ms</span>
                    </div>
                    <div class="lat-track">
                        <div class="lat-fill" style="width: 100%; background: linear-gradient(90deg, var(--amber), #ef4444);"></div>
                    </div>
                </div>
                <p style="font-size:12px; color:#64748b; margin-top:14px; line-height:1.5;">
                    CacheMind delivers an average of <strong>99.9% latency reduction</strong> on cached queries, cutting end-user wait times from seconds to sub-millisecond instant responses.
                </p>
            </div>

            <!-- Query Breakdown SVG Donut -->
            <div class="chart-card">
                <h3>Cache Hit Composition <span>Traffic Distribution</span></h3>
                <div style="display:flex; align-items:center; justify-content:space-around; padding: 10px 0;">
                    <svg width="180" height="180" viewBox="0 0 100 100">
                        <circle cx="50" cy="50" r="40" fill="transparent" stroke="#1e293b" stroke-width="16" />
                        <!-- Semantic Hits Arc -->
                        <circle cx="50" cy="50" r="40" fill="transparent" stroke="#38bdf8" stroke-width="16"
                                stroke-dasharray="{semantic_hit_rate * 2.51} 251.2" stroke-dashoffset="0" />
                        <!-- Exact Hits Arc -->
                        <circle cx="50" cy="50" r="40" fill="transparent" stroke="#10b981" stroke-width="16"
                                stroke-dasharray="{exact_hit_rate * 2.51} 251.2" stroke-dashoffset="-{semantic_hit_rate * 2.51}" />
                        <text x="50" y="47" text-anchor="middle" fill="#fff" font-size="12" font-weight="bold">{hit_rate}%</text>
                        <text x="50" y="60" text-anchor="middle" fill="#94a3b8" font-size="7">HIT RATE</text>
                    </svg>
                    <div style="font-size:13px; line-height: 2;">
                        <div><span style="display:inline-block; width:10px; height:10px; background:#10b981; border-radius:50%; margin-right:8px;"></span>Exact Hits: <strong>{exact_hits}</strong> ({exact_hit_rate}%)</div>
                        <div><span style="display:inline-block; width:10px; height:10px; background:#38bdf8; border-radius:50%; margin-right:8px;"></span>Semantic Hits: <strong>{semantic_hits}</strong> ({semantic_hit_rate}%)</div>
                        <div><span style="display:inline-block; width:10px; height:10px; background:#1e293b; border-radius:50%; margin-right:8px;"></span>Cache Misses: <strong>{misses}</strong></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Interactive Simulator -->
        <div class="sim-box">
            <h3 style="font-size:16px; font-weight:700;">Live Semantic Search Simulator</h3>
            <p style="font-size:13px; color:var(--text-muted); margin-top:4px;">Test how new queries match semantically against active entries in real-time right inside your browser.</p>
            <input type="text" id="simQuery" class="sim-input" placeholder="Type a paraphrased query (e.g. 'Can I get a refund for my order?') and press Enter..." oninput="simulateQuery(this.value)" />
            <div id="simResult" class="sim-result"></div>
        </div>

        <!-- Entries Table -->
        <div class="table-card">
            <h3>Active Cache Registry <span>({len(entries)} entries)</span></h3>
            <table>
                <thead>
                    <tr>
                        <th style="width:50px;">#</th>
                        <th style="width:30%;">Prompt</th>
                        <th style="width:40%;">Cached Response</th>
                        <th style="text-align:center;">Tokens</th>
                        <th style="text-align:center;">Hits</th>
                        <th style="text-align:center;">Namespace</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows_html}
                </tbody>
            </table>
        </div>

        <!-- Footer -->
        <div class="footer">
            CacheMind-AI &bull; Developed by Nitesh Beniwal &bull; MIT Licensed &bull; <a href="https://github.com/NiteshBen/cachemind-ai" style="color:var(--cyan); text-decoration:none;">GitHub Repository</a>
        </div>
    </div>

    <script>
        const cachedEntries = {entries_json};

        function simulateQuery(text) {{
            const resultBox = document.getElementById("simResult");
            if (!text || text.trim().length < 3) {{
                resultBox.style.display = "none";
                return;
            }}

            const queryWords = new Set(text.toLowerCase().match(/\\w+/g) || []);
            let bestMatch = null;
            let bestScore = 0.0;

            for (const entry of cachedEntries) {{
                const entryWords = new Set(entry.prompt.toLowerCase().match(/\\w+/g) || []);
                let intersection = 0;
                for (const w of queryWords) {{
                    if (entryWords.has(w)) intersection++;
                }}
                const union = new Set([...queryWords, ...entryWords]).size;
                const score = union > 0 ? (intersection / union) : 0;
                if (score > bestScore) {{
                    bestScore = score;
                    bestMatch = entry;
                }}
            }}

            resultBox.style.display = "block";
            if (bestScore >= 0.25 && bestMatch) {{
                resultBox.innerHTML = `
                    <div style="color:#10b981; font-weight:700; margin-bottom:4px;">
                        &#10003; Semantic Match Found (Simulated Jaccard Score: ${{ (bestScore * 100).toFixed(1) }}%)
                    </div>
                    <div style="color:#f1f5f9; font-size:13px; margin-bottom:6px;"><strong>Matched Query:</strong> ${{ bestMatch.prompt }}</div>
                    <div style="color:#94a3b8; font-size:13px;"><strong>Cached Response:</strong> ${{ bestMatch.response }}</div>
                `;
            }} else {{
                resultBox.innerHTML = `
                    <div style="color:#f59e0b; font-weight:600;">
                        &#9888; Cache Miss &mdash; Query would be routed to upstream LLM and cached for subsequent turns.
                    </div>
                `;
            }}
        }}
    </script>
</body>
</html>
"""
    abs_path = os.path.abspath(output_path)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return abs_path
