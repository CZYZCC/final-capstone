"""
analyze_results.py
==================
Comprehensive multi-model analysis of SmartQG experiment results.

Usage:
    python analyze_results.py                        # reads experiment_data/scored_*.jsonl
    python analyze_results.py --data-dir ./my_data   # custom directory
    python analyze_results.py --output report.html   # custom output name

Produces:
    - analysis_report.html   (interactive charts — open in browser)
    - analysis_summary.txt   (plain-text table for copy-paste into report)
"""

import os
import sys
import json
import glob
import argparse
import statistics
from collections import defaultdict
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

MODEL_ORDER = [
    "no_retrieval",
    "vector_rag",
    "hybrid_rag",
    "unpruned_graph_rag",
    "naive_graph_rag",
    "graph_rag",           # SmartQG — always last
]

MODEL_LABELS = {
    "no_retrieval":       "No Retrieval",
    "vector_rag":         "Vector RAG",
    "hybrid_rag":         "Hybrid RRF",
    "unpruned_graph_rag": "Unpruned GraphRAG",
    "naive_graph_rag":    "Naive GraphRAG",
    "graph_rag":          "SmartQG (Full)",
}

MODEL_COLORS = {
    "no_retrieval":       "#888780",
    "vector_rag":         "#B4B2A9",
    "hybrid_rag":         "#D3D1C7",
    "unpruned_graph_rag": "#85B7EB",
    "naive_graph_rag":    "#378ADD",
    "graph_rag":          "#185FA5",   # darkest — SmartQG
}

DIMS = [
    ("relevance",            "Relevance",             0.05),
    ("correctness",          "Correctness",           0.20),
    ("diagnostic_power",     "Diagnostic Power",      0.20),
    ("multi_hop_dependency", "Multi-Hop Dependency",  0.15),
    ("edge_case_triggering", "Edge-Case Triggering",  0.20),
    ("graph_relational_depth","Graph-Relational Depth",0.10),
    ("diversity",            "Diversity",              0.10),
]

FORMATS = ["mcq_single", "mcq_multi", "true_false", "fill_blank", "open_answer"]
FORMAT_LABELS = {
    "mcq_single":  "MCQ Single",
    "mcq_multi":   "MCQ Multi",
    "true_false":  "True/False",
    "fill_blank":  "Fill-in-Blank",
    "open_answer": "Open Answer",
}

TASK_TYPES = ["computational", "conceptual"]


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_all(data_dir: str) -> dict:
    """
    Returns dict: model_name → list of records
    Each record: {overall, dims{}, format, task_type, topic, generation_time, method}
    """
    data = {}
    pattern = os.path.join(data_dir, "scored_*.jsonl")
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"[!] No scored_*.jsonl files found in '{data_dir}'")
        print("    Make sure you run the experiment first.")
        sys.exit(1)

    for path in files:
        model_name = os.path.basename(path).replace("scored_", "").replace(".jsonl", "")
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    raw   = json.loads(line)
                    meta  = raw.get("metadata", {})
                    score = raw.get("score",    {})
                    rec = {
                        "overall":       float(score.get("overall", 0)),
                        "format":        meta.get("question_format", "mcq_single"),
                        "task_type":     meta.get("expected_type",   "computational"),
                        "topic":         meta.get("topic",           "unknown"),
                        "gen_time":      float(meta.get("generation_time", 0)),
                        "method":        meta.get("method", "unknown"),
                        "dims": {
                            dk: float(score.get(dk, 0))
                            for dk, _, _ in DIMS
                        },
                    }
                    records.append(rec)
                except Exception:
                    continue
        data[model_name] = records
        print(f"  Loaded {len(records):4d} records  ←  {os.path.basename(path)}")

    return data


# ──────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ──────────────────────────────────────────────────────────────────────────────

def mean(vals):
    return statistics.mean(vals) if vals else 0.0

def stdev(vals):
    return statistics.stdev(vals) if len(vals) > 1 else 0.0

def cohen_d(a, b):
    """Effect size between two score lists."""
    if not a or not b:
        return 0.0
    pooled = ((len(a)-1)*stdev(a)**2 + (len(b)-1)*stdev(b)**2) / (len(a)+len(b)-2)
    pooled = pooled**0.5
    return (mean(a) - mean(b)) / pooled if pooled else 0.0

def filter_records(records, fmt=None, task=None):
    out = records
    if fmt:
        out = [r for r in out if r["format"] == fmt]
    if task:
        out = [r for r in out if r["task_type"] == task]
    return out

def get_overall(records, **filters):
    recs = filter_records(records, **filters)
    return [r["overall"] for r in recs]

def get_dim(records, dim_key, **filters):
    recs = filter_records(records, **filters)
    return [r["dims"][dim_key] for r in recs]


# ──────────────────────────────────────────────────────────────────────────────
# Plain-text summary
# ──────────────────────────────────────────────────────────────────────────────

def build_text_summary(data: dict, models: list) -> str:
    lines = []
    W = 26

    def row(label, vals, fmt=".4f"):
        return f"  {label:<{W}}" + "".join(f"{v:>12{fmt}}" for v in vals)

    def sep(char="-"):
        return "  " + char * (W + 12 * len(models))

    header = f"  {'Model':<{W}}" + "".join(f"{MODEL_LABELS.get(m,m):>12}" for m in models)

    # ── Table 1: Overall ──────────────────────────────────────────────────
    lines += ["", "=" * 80, "TABLE 1 — OVERALL SCORE (mean ± std)", "=" * 80, header, sep()]
    for m in models:
        vals = get_overall(data[m]) if m in data else []
        lines.append(f"  {MODEL_LABELS.get(m,m):<{W}}"
                     f"{mean(vals):>10.4f}  ±{stdev(vals):.3f}  (n={len(vals)})")

    # ── Table 2: 7 Dimensions ────────────────────────────────────────────
    lines += ["", "=" * 80, "TABLE 2 — 7 DIMENSIONS (mean overall score)", "=" * 80, header, sep()]
    for dk, dlabel, wt in DIMS:
        vals = [mean(get_dim(data[m], dk)) if m in data else 0 for m in models]
        lines.append(row(f"{dlabel} ({wt:.0%})", vals))

    # ── Table 3: By Format ───────────────────────────────────────────────
    lines += ["", "=" * 80, "TABLE 3 — BY QUESTION FORMAT", "=" * 80, header, sep()]
    for fmt in FORMATS:
        vals = [mean(get_overall(data[m], fmt=fmt)) if m in data else 0 for m in models]
        lines.append(row(FORMAT_LABELS[fmt], vals))

    # ── Table 4: Computational vs Conceptual ────────────────────────────
    lines += ["", "=" * 80, "TABLE 4 — COMPUTATIONAL vs CONCEPTUAL", "=" * 80, header, sep()]
    for task in TASK_TYPES:
        vals = [mean(get_overall(data[m], task=task)) if m in data else 0 for m in models]
        lines.append(row(task.capitalize(), vals))

    # ── Table 5: Format × Task Type ─────────────────────────────────────
    lines += ["", "=" * 80, "TABLE 5 — FORMAT × TASK TYPE (SmartQG vs best baseline)", "=" * 80]
    smart = "graph_rag"
    best_baseline = max(
        [m for m in models if m != smart and m in data],
        key=lambda m: mean(get_overall(data[m])),
        default=None
    )
    if smart in data and best_baseline:
        lines.append(f"  {'Combo':<35}  {'SmartQG':>9}  {'Best Baseline':>13}  {'Delta':>7}  {'Winner':>8}")
        lines.append("  " + "-" * 75)
        for fmt in FORMATS:
            for task in TASK_TYPES:
                sg  = mean(get_overall(data[smart],        fmt=fmt, task=task))
                bl  = mean(get_overall(data[best_baseline], fmt=fmt, task=task))
                delta = sg - bl
                winner = "SmartQG ✓" if delta > 0 else "Baseline"
                label = f"{FORMAT_LABELS[fmt]} / {task[:5]}"
                lines.append(f"  {label:<35}  {sg:>9.4f}  {bl:>13.4f}  {delta:>+7.4f}  {winner:>8}")

    # ── Table 6: Generation time ─────────────────────────────────────────
    lines += ["", "=" * 80, "TABLE 6 — GENERATION TIME (sec/question)", "=" * 80, header, sep()]
    for label, task in [("All formats", None), ("Computational", "computational"), ("Conceptual", "conceptual")]:
        vals = []
        for m in models:
            if m not in data:
                vals.append(0)
                continue
            recs = filter_records(data[m], task=task)
            vals.append(mean([r["gen_time"] for r in recs]))
        lines.append(row(label, vals))

    # ── Table 7: Effect size (Cohen's d vs SmartQG) ──────────────────────
    lines += ["", "=" * 80, "TABLE 7 — EFFECT SIZE vs SmartQG (Cohen's d)", "=" * 80]
    if smart in data:
        sg_scores = get_overall(data[smart])
        lines.append(f"  {'Model':<28}  {'Cohen d':>9}  {'Interpretation':>16}")
        lines.append("  " + "-" * 58)
        for m in [x for x in models if x != smart]:
            if m not in data:
                continue
            d = cohen_d(sg_scores, get_overall(data[m]))
            interp = "small" if abs(d)<0.5 else ("medium" if abs(d)<0.8 else "large")
            lines.append(f"  {MODEL_LABELS.get(m,m):<28}  {d:>+9.4f}  {interp:>16}")

    lines += ["", "=" * 80,
              f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
              "=" * 80, ""]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# HTML report with Chart.js
# ──────────────────────────────────────────────────────────────────────────────

def build_html(data: dict, models: list) -> str:

    colors = [MODEL_COLORS.get(m, "#888") for m in models]
    labels = [MODEL_LABELS.get(m, m)      for m in models]

    def js_arr(vals, dp=4):
        return "[" + ", ".join(f"{v:.{dp}f}" for v in vals) + "]"

    def overall_means():
        return [mean(get_overall(data[m])) if m in data else 0 for m in models]

    def dim_means(dk):
        return [mean(get_dim(data[m], dk)) if m in data else 0 for m in models]

    def fmt_means(fmt):
        return [mean(get_overall(data[m], fmt=fmt)) if m in data else 0 for m in models]

    def task_means(task):
        return [mean(get_overall(data[m], task=task)) if m in data else 0 for m in models]

    # Radar chart: 7 dimensions for SmartQG vs best baseline
    smart        = "graph_rag"
    baselines    = [m for m in models if m != smart and m in data]
    best_baseline = max(baselines, key=lambda m: mean(get_overall(data[m])), default=None)

    radar_sg = [mean(get_dim(data[smart], dk)) if smart in data else 0 for dk, _, _ in DIMS]
    radar_bl = [mean(get_dim(data[best_baseline], dk)) if best_baseline and best_baseline in data else 0 for dk, _, _ in DIMS]
    radar_labels = [dlabel for _, dlabel, _ in DIMS]

    # Per-topic SmartQG overall (for heatmap substitute — bar chart)
    topic_data = defaultdict(list)
    if smart in data:
        for r in data[smart]:
            topic_data[r["topic"]].append(r["overall"])
    topic_means_sorted = sorted(
        [(t, mean(v)) for t, v in topic_data.items()],
        key=lambda x: -x[1]
    )
    topic_names  = [t for t, _ in topic_means_sorted]
    topic_scores = [v for _, v in topic_means_sorted]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SmartQG Experiment Analysis</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: #f5f5f4; color: #2c2c2a; line-height: 1.6; }}
.page {{ max-width: 1200px; margin: 0 auto; padding: 2rem 1.5rem; }}
h1 {{ font-size: 1.6rem; font-weight: 600; margin-bottom: 0.25rem; }}
.subtitle {{ font-size: 0.9rem; color: #888; margin-bottom: 2rem; }}
h2 {{ font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem;
      padding-bottom: 0.5rem; border-bottom: 1.5px solid #e0ddd6; }}
.grid {{ display: grid; gap: 1.5rem; }}
.grid-2 {{ grid-template-columns: 1fr 1fr; }}
.grid-3 {{ grid-template-columns: 1fr 1fr 1fr; }}
.card {{ background: #fff; border-radius: 12px; padding: 1.25rem 1.5rem;
          border: 0.5px solid #e0ddd6; }}
.chart-wrap {{ position: relative; }}
.stat-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 1.5rem; }}
.stat {{ background: #f1efe8; border-radius: 8px; padding: 10px 14px; }}
.stat-label {{ font-size: 11px; color: #888; margin-bottom: 2px; }}
.stat-value {{ font-size: 20px; font-weight: 600; }}
.stat-sub {{ font-size: 11px; color: #aaa; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
th {{ background: #f1efe8; padding: 8px 10px; text-align: right;
      font-weight: 600; font-size: 0.78rem; color: #555; }}
th:first-child {{ text-align: left; }}
td {{ padding: 7px 10px; text-align: right; border-bottom: 0.5px solid #eee; }}
td:first-child {{ text-align: left; font-weight: 500; }}
tr.best td {{ background: #eaf3de; }}
tr.smartqg td {{ background: #e6f1fb; font-weight: 600; }}
.badge {{ display: inline-block; font-size: 10px; padding: 1px 7px;
           border-radius: 10px; margin-left: 4px; }}
.badge-win {{ background: #eaf3de; color: #3b6d11; }}
.badge-lose {{ background: #fcebeb; color: #a32d2d; }}
.section {{ margin-bottom: 2rem; }}
@media (max-width: 760px) {{
  .grid-2, .grid-3 {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<div class="page">

<h1>SmartQG Experiment — Full Analysis</h1>
<p class="subtitle">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp;
Models: {len(models)} &nbsp;|&nbsp;
Total questions: {sum(len(data.get(m,[])) for m in models):,}</p>

<!-- ── Summary stats ── -->
<div class="stat-grid">
"""
    # Top-level stats
    if smart in data:
        sg_mean = mean(get_overall(data[smart]))
        sg_fb   = mean(get_overall(data[smart], fmt="fill_blank"))
        sg_dp   = mean(get_dim(data[smart], "diagnostic_power"))
        html += f"""
  <div class="stat"><div class="stat-label">SmartQG Overall</div>
    <div class="stat-value" style="color:#185FA5">{sg_mean:.4f}</div>
    <div class="stat-sub">out of 5.0</div></div>
  <div class="stat"><div class="stat-label">Fill-in-Blank Score</div>
    <div class="stat-value" style="color:#185FA5">{sg_fb:.4f}</div>
    <div class="stat-sub">hardest format</div></div>
  <div class="stat"><div class="stat-label">Diagnostic Power</div>
    <div class="stat-value" style="color:#185FA5">{sg_dp:.4f}</div>
    <div class="stat-sub">highest dim weight</div></div>
"""
    html += "</div>\n\n"

    # ── Chart 1: Overall comparison (horizontal bar) ─────────────────────
    html += """
<div class="section card">
<h2>1 · Overall Score — All Models</h2>
<div class="chart-wrap" style="height:260px">
<canvas id="c1" role="img" aria-label="Horizontal bar chart of overall scores for all models"></canvas>
</div></div>

<div class="section grid grid-2">
"""

    # ── Chart 2: 7-dimension radar ────────────────────────────────────────
    html += """
<div class="card">
<h2>2 · 7-Dimension Radar</h2>
<div class="chart-wrap" style="height:300px">
<canvas id="c2" role="img" aria-label="Radar chart comparing SmartQG vs best baseline across 7 dimensions"></canvas>
</div></div>
"""

    # ── Chart 3: Format resilience (grouped bar) ──────────────────────────
    html += """
<div class="card">
<h2>3 · Format Resilience</h2>
<div class="chart-wrap" style="height:300px">
<canvas id="c3" role="img" aria-label="Grouped bar chart of scores by question format"></canvas>
</div></div>

</div><!-- end grid-2 -->

<div class="section grid grid-2">
"""

    # ── Chart 4: Computational vs Conceptual ──────────────────────────────
    html += """
<div class="card">
<h2>4 · Computational vs Conceptual</h2>
<div class="chart-wrap" style="height:280px">
<canvas id="c4" role="img" aria-label="Grouped bar chart of computational vs conceptual scores"></canvas>
</div></div>
"""

    # ── Chart 5: Dimensions bar for each model ────────────────────────────
    html += """
<div class="card">
<h2>5 · Dimension Scores by Model</h2>
<div class="chart-wrap" style="height:280px">
<canvas id="c5" role="img" aria-label="Grouped bar chart of all 7 dimensions for each model"></canvas>
</div></div>

</div><!-- end grid-2 -->

<div class="section grid grid-2">
"""

    # ── Chart 6: Topic bar (SmartQG only) ────────────────────────────────
    html += f"""
<div class="card">
<h2>6 · SmartQG Score by Topic</h2>
<div class="chart-wrap" style="height:320px">
<canvas id="c6" role="img" aria-label="Horizontal bar chart of SmartQG scores per topic"></canvas>
</div></div>
"""

    # ── Chart 7: Generation time ──────────────────────────────────────────
    html += """
<div class="card">
<h2>7 · Generation Time (s/question)</h2>
<div class="chart-wrap" style="height:320px">
<canvas id="c7" role="img" aria-label="Bar chart of generation time per model"></canvas>
</div></div>

</div><!-- end grid-2 -->
"""

    # ── Table: Detailed numbers ───────────────────────────────────────────
    html += """
<div class="section card">
<h2>8 · Full Numeric Table</h2>
<table>
<thead><tr>
<th>Model</th>
<th>N</th>
<th>Overall</th>
"""
    for _, dlabel, _ in DIMS:
        html += f"<th>{dlabel}</th>"
    for fmt in FORMATS:
        html += f"<th>{FORMAT_LABELS[fmt]}</th>"
    html += "<th>Comp.</th><th>Conc.</th><th>Time(s)</th></tr></thead><tbody>\n"

    sg_overall = mean(get_overall(data.get(smart, [])))
    for m in models:
        if m not in data:
            continue
        recs    = data[m]
        ov      = mean(get_overall(recs))
        row_cls = "smartqg" if m == smart else ""
        html += f'<tr class="{row_cls}"><td>{MODEL_LABELS.get(m,m)}</td>'
        html += f'<td>{len(recs)}</td>'
        diff = ov - sg_overall
        badge = ""
        if m != smart:
            cls = "badge-lose" if diff < 0 else "badge-win"
            badge = f'<span class="badge {cls}">{diff:+.3f}</span>'
        html += f'<td>{ov:.4f}{badge}</td>'
        for dk, _, _ in DIMS:
            html += f'<td>{mean(get_dim(recs, dk)):.3f}</td>'
        for fmt in FORMATS:
            html += f'<td>{mean(get_overall(recs, fmt=fmt)):.3f}</td>'
        html += f'<td>{mean(get_overall(recs, task="computational")):.3f}</td>'
        html += f'<td>{mean(get_overall(recs, task="conceptual")):.3f}</td>'
        html += f'<td>{mean([r["gen_time"] for r in recs]):.1f}</td>'
        html += '</tr>\n'

    html += "</tbody></table></div>\n"

    # ── JavaScript ────────────────────────────────────────────────────────
    # Prepare all data arrays
    ov_vals  = overall_means()
    fmt_datasets = []
    for m, color, label in zip(models, colors, labels):
        fmt_datasets.append({
            "label": label,
            "data":  [mean(get_overall(data[m], fmt=f)) if m in data else 0 for f in FORMATS],
            "backgroundColor": color,
        })

    comp_datasets = []
    for m, color, label in zip(models, colors, labels):
        comp_datasets.append({
            "label": label,
            "data": [mean(get_overall(data[m], task=t)) if m in data else 0 for t in TASK_TYPES],
            "backgroundColor": color,
        })

    dim_datasets = []
    for m, color, label in zip(models, colors, labels):
        dim_datasets.append({
            "label": label,
            "data": [mean(get_dim(data[m], dk)) if m in data else 0 for dk, _, _ in DIMS],
            "backgroundColor": color,
        })

    gen_times = [mean([r["gen_time"] for r in data[m]]) if m in data else 0 for m in models]

    bl_label = MODEL_LABELS.get(best_baseline, best_baseline) if best_baseline else "Baseline"

    html += f"""
<script>
const modelLabels  = {json.dumps(labels)};
const modelColors  = {json.dumps(colors)};
const formatLabels = {json.dumps([FORMAT_LABELS[f] for f in FORMATS])};
const dimLabels    = {json.dumps(radar_labels)};

// Chart 1 — Overall horizontal bar
new Chart(document.getElementById('c1'), {{
  type: 'bar',
  data: {{
    labels: modelLabels,
    datasets: [{{
      label: 'Overall Score',
      data: {js_arr(ov_vals)},
      backgroundColor: modelColors,
      borderRadius: 4,
    }}]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }},
      tooltip: {{ callbacks: {{ label: ctx => ' ' + ctx.parsed.x.toFixed(4) }} }} }},
    scales: {{
      x: {{ min: 2.5, max: 5.0,
            title: {{ display: true, text: 'Mean Overall Score (1–5)' }} }},
      y: {{ ticks: {{ font: {{ size: 12 }} }} }}
    }}
  }}
}});

// Chart 2 — Radar
new Chart(document.getElementById('c2'), {{
  type: 'radar',
  data: {{
    labels: dimLabels,
    datasets: [
      {{
        label: 'SmartQG',
        data: {js_arr(radar_sg)},
        borderColor: '#185FA5', backgroundColor: 'rgba(24,95,165,0.15)',
        pointBackgroundColor: '#185FA5', borderWidth: 2,
      }},
      {{
        label: '{bl_label}',
        data: {js_arr(radar_bl)},
        borderColor: '#888780', backgroundColor: 'rgba(136,135,128,0.1)',
        pointBackgroundColor: '#888780', borderWidth: 1.5, borderDash: [4,3],
      }}
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    scales: {{ r: {{ min: 1, max: 5,
      ticks: {{ stepSize: 1, font: {{ size: 10 }} }},
      pointLabels: {{ font: {{ size: 11 }} }} }} }},
    plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 12, font: {{ size: 11 }} }} }} }}
  }}
}});

// Chart 3 — Format grouped bar
new Chart(document.getElementById('c3'), {{
  type: 'bar',
  data: {{
    labels: formatLabels,
    datasets: {json.dumps(fmt_datasets)}
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 10, font: {{ size: 10 }} }} }} }},
    scales: {{ y: {{ min: 1.5, max: 5.0 }},
               x: {{ ticks: {{ font: {{ size: 11 }} }} }} }}
  }}
}});

// Chart 4 — Comp vs Conceptual
new Chart(document.getElementById('c4'), {{
  type: 'bar',
  data: {{
    labels: ['Computational', 'Conceptual'],
    datasets: {json.dumps(comp_datasets)}
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 10, font: {{ size: 10 }} }} }} }},
    scales: {{ y: {{ min: 2.5, max: 5.0 }} }}
  }}
}});

// Chart 5 — All dimensions grouped bar
new Chart(document.getElementById('c5'), {{
  type: 'bar',
  data: {{
    labels: dimLabels,
    datasets: {json.dumps(dim_datasets)}
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 10, font: {{ size: 10 }} }} }} }},
    scales: {{ y: {{ min: 1.0, max: 5.0 }},
               x: {{ ticks: {{ font: {{ size: 9 }}, maxRotation: 30 }} }} }}
  }}
}});

// Chart 6 — Topic bar (SmartQG)
new Chart(document.getElementById('c6'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(topic_names)},
    datasets: [{{
      label: 'SmartQG Overall',
      data: {js_arr(topic_scores)},
      backgroundColor: {json.dumps(["#185FA5" if s >= 3.75 else "#854F0B" if s >= 3.5 else "#A32D2D" for s in topic_scores])},
      borderRadius: 3,
    }}]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ min: 2.5, max: 5.0 }},
      y: {{ ticks: {{ font: {{ size: 10 }} }} }}
    }}
  }}
}});

// Chart 7 — Generation time
new Chart(document.getElementById('c7'), {{
  type: 'bar',
  data: {{
    labels: modelLabels,
    datasets: [{{
      label: 'Avg time (s)',
      data: {js_arr(gen_times, 1)},
      backgroundColor: modelColors,
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      y: {{ title: {{ display: true, text: 'seconds / question' }} }},
      x: {{ ticks: {{ font: {{ size: 11 }}, maxRotation: 20 }} }}
    }}
  }}
}});
</script>
</div></body></html>"""

    return html


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SmartQG Results Analyser")
    parser.add_argument("--data-dir", default="experiment_data",
                        help="Directory containing scored_*.jsonl files (default: experiment_data)")
    parser.add_argument("--output",   default="analysis_report.html",
                        help="Output HTML filename (default: analysis_report.html)")
    args = parser.parse_args()

    print(f"\n[*] Loading scored JSONL files from '{args.data_dir}' …\n")
    data = load_all(args.data_dir)

    # Determine which models actually have data, maintain canonical order
    models = [m for m in MODEL_ORDER if m in data]
    # Add any model found in data but not in MODEL_ORDER (edge case)
    for m in data:
        if m not in models:
            models.append(m)

    print(f"\n[*] Models found: {models}")
    print(f"[*] Building analysis …\n")

    # Plain-text summary
    txt_path = args.output.replace(".html", ".txt").replace(".HTML", ".txt")
    if not txt_path.endswith(".txt"):
        txt_path += ".txt"
    txt = build_text_summary(data, models)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)   # also print to terminal
    print(f"[*] Plain-text summary → {txt_path}")

    # HTML report
    html = build_html(data, models)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[*] HTML report        → {args.output}")
    print(f"\n[*] Done. Open {args.output} in your browser to see the charts.\n")


if __name__ == "__main__":
    main()
