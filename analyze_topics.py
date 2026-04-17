"""
analyze_topics.py  —  SmartQG Topic 多维度分析脚本
====================================================
用法:
    python analyze_topics.py                        # 读 experiment_data/scored_*.jsonl
    python analyze_topics.py --data-dir ./my_data   # 自定义目录

输出:
    topic_analysis.txt   纯文本，可复制进报告
    topic_analysis.html  浏览器打开，表格可点击表头排序，有筛选控件
"""

import os, sys, json, glob, argparse, statistics
from collections import defaultdict

# ── 常量 ────────────────────────────────────────────────────────────────
MODEL_ORDER = ["no_retrieval","vector_rag","hybrid_rag",
               "unpruned_graph_rag","naive_graph_rag","graph_rag"]
MODEL_LABELS = {
    "no_retrieval":"No Retrieval","vector_rag":"Vector RAG",
    "hybrid_rag":"Hybrid RRF","unpruned_graph_rag":"Unpruned GraphRAG",
    "naive_graph_rag":"Naive GraphRAG","graph_rag":"SmartQG",
}
DIMS = ["overall","relevance","correctness","diagnostic_power",
        "multi_hop_dependency","edge_case_triggering","graph_relational_depth","diversity"]
FMTS = ["mcq_single","mcq_multi","true_false","fill_blank","open_answer"]
FMT_LABELS = {"mcq_single":"MCQ Single","mcq_multi":"MCQ Multi",
              "true_false":"True/False","fill_blank":"Fill-Blank","open_answer":"Open Answer"}
SMART = "graph_rag"

def mean(v): return statistics.mean(v) if v else 0.0

# ── 数据加载 ─────────────────────────────────────────────────────────────
def load_data(data_dir):
    result = {}
    files  = sorted(glob.glob(os.path.join(data_dir, "scored_*.jsonl")))
    if not files:
        print(f"[!] 在 '{data_dir}' 里没找到 scored_*.jsonl"); sys.exit(1)
    for path in files:
        model = os.path.basename(path).replace("scored_","").replace(".jsonl","")
        recs  = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                try: recs.append(json.loads(line))
                except: pass
        result[model] = recs
        print(f"  loaded {len(recs):4d}  ← {os.path.basename(path)}")
    return result

def get_models(data):
    ordered = [m for m in MODEL_ORDER if m in data]
    for m in data:
        if m not in ordered: ordered.append(m)
    return ordered

def score(recs, dim="overall"):
    vals = [r["score"].get(dim,0) for r in recs if r["score"].get(dim) is not None]
    return mean(vals)

# ── 核心计算 ─────────────────────────────────────────────────────────────
def build_analysis(data, models):
    sg_recs = data.get(SMART, [])
    topics  = sorted(set(r["metadata"]["topic"] for r in sg_recs))
    A = {}
    for topic in topics:
        A[topic] = {}
        for fmt in FMTS:
            A[topic][fmt] = {}
            for m in models:
                recs = [r for r in data.get(m,[])
                        if r["metadata"]["topic"]==topic
                        and r["metadata"]["question_format"]==fmt]
                A[topic][fmt][m] = {dk: score(recs, dk) for dk in DIMS}
        # 汇总 (所有 format)
        A[topic]["_all"] = {}
        for m in models:
            recs = [r for r in data.get(m,[]) if r["metadata"]["topic"]==topic]
            A[topic]["_all"][m] = {dk: score(recs, dk) for dk in DIMS}
        # SmartQG delta vs best baseline
        A[topic]["_delta"] = {}
        for key in FMTS + ["_all"]:
            sg_ov = A[topic][key].get(SMART,{}).get("overall",0)
            bls   = [A[topic][key][m]["overall"] for m in models if m != SMART]
            A[topic]["_delta"][key] = sg_ov - (max(bls) if bls else 0)
    return A, topics

# ── 纯文本报告 ───────────────────────────────────────────────────────────
def build_text(A, topics, models):
    L = []
    ml = [MODEL_LABELS.get(m,m) for m in models]

    def hdr(t): L.extend(["","="*90,f"  {t}","="*90])

    # TABLE 0: overall summary sorted by delta
    hdr("TABLE 0 — 全局 Overall (SmartQG Δ vs best baseline, 从差到好)")
    L.append(f"  {'Topic':<35} {'SG':>7} " +
             "".join(f"{MODEL_LABELS.get(m,m):>14}" for m in models if m!=SMART) +
             f"  {'Δ(SG-best)':>12}  胜负")
    L.append("  "+"-"*100)
    rows = []
    for t in topics:
        sg = A[t]["_all"].get(SMART,{}).get("overall",0)
        bls= {m: A[t]["_all"][m]["overall"] for m in models if m!=SMART}
        best= max(bls.values()) if bls else 0
        rows.append((t, sg, bls, best, sg-best))
    rows.sort(key=lambda x: x[4])
    for t,sg,bls,best,d in rows:
        bl_str = "".join(f"{bls.get(m,0):>14.3f}" for m in models if m!=SMART)
        L.append(f"  {t:<35} {sg:>7.3f}{bl_str}  {d:>+12.3f}  {'✓ SG' if d>0 else '✗ BL'}")

    # TABLE 1: format delta matrix
    hdr("TABLE 1 — 每个 Topic × 5 题型  SmartQG Δ vs best baseline")
    L.append(f"  {'Topic':<35}" + "".join(f"{FMT_LABELS[f]:>14}" for f in FMTS) + f"  {'综合Δ':>10}")
    L.append("  "+"-"*105)
    for t,sg,bls,best,overall_d in rows:
        cells = []
        for fmt in FMTS:
            sv = A[t][fmt].get(SMART,{}).get("overall",0)
            bv = max([A[t][fmt][m]["overall"] for m in models if m!=SMART],default=0)
            cells.append(f"{sv:.3f}({sv-bv:+.2f})")
        L.append(f"  {t:<35}" + "".join(f"{c:>14}" for c in cells) + f"  {overall_d:>+10.3f}")

    # TABLE 2: 7 dims for SmartQG
    hdr("TABLE 2 — 每个 Topic × 7 维度 (SmartQG only)")
    dim_short = DIMS[1:]
    L.append(f"  {'Topic':<35}" + "".join(f"{d[:9]:>11}" for d in dim_short))
    L.append("  "+"-"*120)
    for t,*_ in rows:
        vals = [A[t]["_all"].get(SMART,{}).get(dk,0) for dk in dim_short]
        L.append(f"  {t:<35}" + "".join(f"{v:>11.3f}" for v in vals))

    # TABLE 3: per-topic per-model details
    for t,sg,bls,best,overall_d in rows:
        hdr(f"TABLE 3 [{t.upper()}] — 6 Models × Dims + Format breakdown")
        L.append(f"  {'Dim':<30}" + "".join(f"{lb:>16}" for lb in ml))
        L.append("  "+"-"*110)
        for dk in DIMS:
            vals = [A[t]["_all"].get(m,{}).get(dk,0) for m in models]
            sg_v = vals[models.index(SMART)] if SMART in models else 0
            bl_max = max(v for i,v in enumerate(vals) if models[i]!=SMART)
            L.append(f"  {dk:<30}" +
                     "".join(f"{'★' if (models[i]==SMART and v>=bl_max) else ' '}{v:>15.3f}"
                             for i,v in enumerate(vals)))
        L.append("")
        L.append(f"  {'Format':<18}" + "".join(f"{lb:>16}" for lb in ml) + f"  {'SG Δ':>8}")
        L.append("  "+"-"*110)
        for fmt in FMTS:
            vals = [A[t][fmt].get(m,{}).get("overall",0) for m in models]
            sv   = vals[models.index(SMART)] if SMART in models else 0
            bv   = max(v for i,v in enumerate(vals) if models[i]!=SMART)
            L.append(f"  {FMT_LABELS[fmt]:<18}" +
                     "".join(f"{v:>16.3f}" for v in vals) +
                     f"  {sv-bv:>+8.3f}{'✓' if sv>bv else '✗'}")

    # TABLE 4: remove-N simulation
    hdr("TABLE 4 — 删掉最差 N 个 Topic 后各 Model Overall")
    sorted_topics = [t for t,*_ in rows]  # worst first
    L.append(f"  {'删N个':>8}" + "".join(f"{lb:>16}" for lb in ml) + f"  {'SG-best_BL':>12}")
    L.append("  "+"-"*110)
    for n in range(0, min(11, len(topics)+1)):
        remove = set(sorted_topics[:n])
        avgs = {}
        for m in models:
            vals = [A[t]["_all"][m]["overall"] for t in topics if t not in remove]
            avgs[m] = mean(vals)
        sg_sc = avgs.get(SMART,0)
        best  = max(v for m,v in avgs.items() if m!=SMART)
        L.append(f"  {n:>8}" + "".join(f"{avgs[m]:>16.4f}" for m in models) +
                 f"  {sg_sc-best:>+12.4f}")
        if n > 0:
            L.append(f"           删掉: {', '.join(list(remove))}")

    L += ["", "★ = SmartQG 领先所有 baseline 的维度", "✓/✗ = SmartQG overall 胜/负"]
    return "\n".join(L)

# ── HTML 报告 ────────────────────────────────────────────────────────────
def build_html(A, topics, models):
    ml = [MODEL_LABELS.get(m,m) for m in models]
    sg_idx = models.index(SMART) if SMART in models else -1

    rows_data = []
    for t in topics:
        sg_ov = A[t]["_all"].get(SMART,{}).get("overall",0)
        all_ovs = [A[t]["_all"].get(m,{}).get("overall",0) for m in models]
        bls = [v for i,v in enumerate(all_ovs) if i!=sg_idx]
        best_bl = max(bls) if bls else 0
        fmt_deltas = []
        for fmt in FMTS:
            sv = A[t][fmt].get(SMART,{}).get("overall",0)
            bv = max([A[t][fmt].get(m,{}).get("overall",0) for m in models if m!=SMART],default=0)
            fmt_deltas.append(round(sv-bv,4))
        dims_sg = [round(A[t]["_all"].get(SMART,{}).get(dk,0),4) for dk in DIMS[1:]]
        # per-format per-model
        fmt_model = {}
        for fmt in FMTS:
            fmt_model[fmt] = [round(A[t][fmt].get(m,{}).get("overall",0),4) for m in models]
        rows_data.append({"topic":t,"sg":round(sg_ov,4),"delta":round(sg_ov-best_bl,4),
                          "all_ovs":[round(v,4) for v in all_ovs],
                          "fmt_deltas":fmt_deltas,"dims":dims_sg,"fmt_model":fmt_model})

    rj  = json.dumps(rows_data, ensure_ascii=False)
    mj  = json.dumps(ml, ensure_ascii=False)
    fj  = json.dumps([FMT_LABELS[f] for f in FMTS], ensure_ascii=False)
    dj  = json.dumps(DIMS[1:], ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="UTF-8">
<title>SmartQG Topic Analysis</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
      background:#f5f5f4;color:#2c2c2a;padding:1.5rem;}}
h1{{font-size:1.35rem;margin-bottom:.2rem;}}
.sub{{font-size:.82rem;color:#888;margin-bottom:1.5rem;}}
.card{{background:#fff;border-radius:12px;border:.5px solid #e0ddd6;padding:1rem 1.25rem;margin-bottom:1.5rem;overflow-x:auto;}}
h2{{font-size:.95rem;font-weight:600;margin-bottom:.6rem;padding-bottom:5px;border-bottom:.5px solid #eee;}}
table{{border-collapse:collapse;font-size:.75rem;width:100%;}}
th{{background:#f1efe8;padding:5px 7px;text-align:right;cursor:pointer;white-space:nowrap;user-select:none;}}
th:first-child{{text-align:left;}}
td{{padding:4px 7px;text-align:right;border-bottom:.5px solid #eee;}}
td:first-child{{text-align:left;white-space:nowrap;font-weight:500;}}
.sg{{background:#e8f3fb;font-weight:700;}}
.win{{color:#2d6e0e;font-weight:600;}}
.lose{{color:#9e2a2a;}}
.ctrl{{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:.6rem;font-size:.82rem;}}
input[type=range]{{width:110px;}}
input[type=text],select{{font-size:.8rem;padding:3px 7px;border:.5px solid #ccc;border-radius:6px;}}
.badge{{font-size:.65rem;padding:1px 5px;border-radius:5px;margin-left:2px;}}
.bw{{background:#eaf3de;color:#2d6e0e;}}
.bl{{background:#fde8e8;color:#9e2a2a;}}
</style></head><body>
<h1>SmartQG — Topic 多维度分析</h1>
<p class="sub">6 Models × {len(topics)} Topics × 5 Formats × 7 Dimensions</p>

<div class="card">
<h2>Table 0 — Overall 对比（点击表头排序 | 滑块筛选 Δ | 搜索 Topic）</h2>
<div class="ctrl">
  <label>只显示 Δ ≤ <b id="dv">99</b></label>
  <input type="range" id="df" min="-0.5" max="0.5" step="0.01" value="0.5"
    oninput="document.getElementById('dv').textContent=this.value;renderT0()">
  <label>Topic 搜索: <input type="text" id="ts" placeholder="…" oninput="renderT0()"></label>
</div>
<div id="t0"></div></div>

<div class="card">
<h2>Table 1 — SmartQG Δ 热力矩阵（每 Topic × 5 题型）</h2>
<div id="t1"></div></div>

<div class="card">
<h2>Table 2 — SmartQG × 7 维度（按综合 Δ 排序）</h2>
<div id="t2"></div></div>

<div class="card">
<h2>Table 3 — 每 Topic 各 Model 详情（6 Models × 5 Formats）</h2>
<div class="ctrl">
  <label>选择 Topic: <select id="topic-sel" onchange="renderT3()"></select></label>
</div>
<div id="t3"></div></div>

<div class="card">
<h2>Table 4 — 删掉最差 N 个 Topic 后 Overall 模拟</h2>
<div id="t4"></div></div>

<script>
const ROWS={rj}, MODELS={mj}, FMTS={fj}, DIMS={dj};
const SG_IDX=MODELS.indexOf("SmartQG");
let sc="delta", sa=true;

function fmt(v,bold=false){{
  const s=v>=0?`<span class="win">+${{v.toFixed(3)}}</span>`:`<span class="lose">${{v.toFixed(3)}}</span>`;
  return bold?`<b>${{s}}</b>`:s;
}}
function heatBg(d){{
  return d>0?`rgba(45,110,14,${{Math.min(d/0.4,1)*.3+.04}})`:`rgba(158,42,42,${{Math.min(-d/0.4,1)*.3+.04}})`;
}}

// Table 0
function renderT0(){{
  const thr=parseFloat(document.getElementById("df").value);
  const q=document.getElementById("ts").value.toLowerCase();
  let rows=ROWS.filter(r=>r.delta<=thr&&r.topic.toLowerCase().includes(q));
  rows.sort((a,b)=>{{
    let av,bv;
    if(sc==="delta"){{av=a.delta;bv=b.delta;}}
    else if(sc==="sg"){{av=a.sg;bv=b.sg;}}
    else if(sc.startsWith("m")){{const i=parseInt(sc.slice(1));av=a.all_ovs[i]??0;bv=b.all_ovs[i]??0;}}
    else if(sc.startsWith("f")){{const i=parseInt(sc.slice(1));av=a.fmt_deltas[i]??0;bv=b.fmt_deltas[i]??0;}}
    else{{av=a.delta;bv=b.delta;}}
    return sa?av-bv:bv-av;
  }});
  let h=`<table><thead><tr><th onclick="sort('topic')">Topic</th><th onclick="sort('sg')">SmartQG</th>`;
  MODELS.forEach((m,i)=>{{if(m!=="SmartQG")h+=`<th onclick="sort('m${{i}}')">${{m}}</th>`;}});
  h+=`<th onclick="sort('delta')">Δ SG-best</th>`;
  FMTS.forEach((f,i)=>h+=`<th onclick="sort('f${{i}}')">Δ ${{f}}</th>`);
  h+=`</tr></thead><tbody>`;
  rows.forEach(r=>{{
    const tag=r.delta>=0?`<span class="badge bw">✓</span>`:`<span class="badge bl">✗</span>`;
    h+=`<tr><td>${{r.topic}}${{tag}}</td><td class="sg">${{r.sg.toFixed(3)}}</td>`;
    r.all_ovs.forEach((v,i)=>{{if(MODELS[i]!=="SmartQG")h+=`<td>${{v.toFixed(3)}}</td>`;}});
    h+=`<td>${{fmt(r.delta,true)}}</td>`;
    r.fmt_deltas.forEach(d=>h+=`<td style="background:${{heatBg(d)}}">${{fmt(d)}}</td>`);
    h+=`</tr>`;
  }});
  document.getElementById("t0").innerHTML=h+`</tbody></table>`;
}}
function sort(c){{if(sc===c)sa=!sa;else{{sc=c;sa=true;}}renderT0();}}

// Table 1
function renderT1(){{
  const sorted=[...ROWS].sort((a,b)=>a.delta-b.delta);
  let h=`<table><thead><tr><th>Topic</th>`;
  FMTS.forEach(f=>h+=`<th>${{f}}</th>`);
  h+=`<th>综合Δ</th></tr></thead><tbody>`;
  sorted.forEach(r=>{{
    h+=`<tr><td>${{r.topic}}</td>`;
    r.fmt_deltas.forEach(d=>h+=`<td style="background:${{heatBg(d)}}">${{fmt(d)}}</td>`);
    h+=`<td>${{fmt(r.delta,true)}}</td></tr>`;
  }});
  document.getElementById("t1").innerHTML=h+`</tbody></table>`;
}}

// Table 2
function renderT2(){{
  const sorted=[...ROWS].sort((a,b)=>a.delta-b.delta);
  let h=`<table><thead><tr><th>Topic</th><th>综合Δ</th>`;
  DIMS.forEach(d=>h+=`<th>${{d}}</th>`);
  h+=`</tr></thead><tbody>`;
  sorted.forEach(r=>{{
    h+=`<tr><td>${{r.topic}}</td><td>${{fmt(r.delta)}}</td>`;
    r.dims.forEach(v=>h+=`<td>${{v.toFixed(3)}}</td>`);
    h+=`</tr>`;
  }});
  document.getElementById("t2").innerHTML=h+`</tbody></table>`;
}}

// Table 3
function initT3(){{
  const sel=document.getElementById("topic-sel");
  const sorted=[...ROWS].sort((a,b)=>a.delta-b.delta);
  sorted.forEach(r=>{{
    const o=document.createElement("option");
    o.value=r.topic; o.textContent=`${{r.topic}} (Δ${{r.delta>=0?"+":""}}${{r.delta.toFixed(3)}})`;
    sel.appendChild(o);
  }});
  renderT3();
}}
function renderT3(){{
  const topic=document.getElementById("topic-sel").value;
  const r=ROWS.find(x=>x.topic===topic); if(!r)return;
  let h=`<table><thead><tr><th>Format</th>`;
  MODELS.forEach((m,i)=>{{const cls=i===SG_IDX?' class="sg"':'';h+=`<th${{cls}}>${{m}}</th>`;}});
  h+=`<th>SG Δ</th></tr></thead><tbody>`;
  FMTS.forEach((f,fi)=>{{
    h+=`<tr><td>${{f}}</td>`;
    const vals=r.fmt_model[f];
    const sgv=vals[SG_IDX];
    const blmax=Math.max(...vals.filter((_,i)=>i!==SG_IDX));
    vals.forEach((v,i)=>{{
      const cls=i===SG_IDX?' class="sg"':'';
      h+=`<td${{cls}}>${{v.toFixed(3)}}</td>`;
    }});
    const d=sgv-blmax;
    h+=`<td style="background:${{heatBg(d)}}">${{fmt(d,true)}}</td></tr>`;
  }});
  // dims
  h+=`<tr><td colspan="${{MODELS.length+2}}" style="background:#f8f8f6;font-size:.7rem;padding:4px 7px">
      SmartQG 7维度</td></tr>`;
  DIMS.forEach((dk,di)=>{{
    if(di===0)return;
    h+=`<tr><td style="color:#888">${{dk}}</td>`;
    const v=r.dims[di-1]; h+=`<td class="sg" colspan="${{MODELS.length}}">${{v.toFixed(3)}}</td><td></td></tr>`;
  }});
  document.getElementById("t3").innerHTML=h+`</tbody></table>`;
}}

// Table 4
function renderT4(){{
  const sorted=[...ROWS].sort((a,b)=>a.delta-b.delta);
  let h=`<table><thead><tr><th>删N个</th>`;
  MODELS.forEach((m,i)=>{{const cls=i===SG_IDX?' class="sg"':'';h+=`<th${{cls}}>${{m}}</th>`;}});
  h+=`<th>SG-best_BL</th><th style="min-width:200px">删掉的 Topics</th></tr></thead><tbody>`;
  for(let n=0;n<=Math.min(10,sorted.length);n++){{
    const rem=new Set(sorted.slice(0,n).map(r=>r.topic));
    const keep=ROWS.filter(r=>!rem.has(r.topic));
    const avgs=MODELS.map((_,i)=>{{
      const vals=keep.map(r=>r.all_ovs[i]??0);
      return vals.length?vals.reduce((a,b)=>a+b,0)/vals.length:0;
    }});
    const sg=avgs[SG_IDX]; const blmax=Math.max(...avgs.filter((_,i)=>i!==SG_IDX));
    const d=sg-blmax;
    h+=`<tr><td>${{n}}</td>`;
    avgs.forEach((v,i)=>{{const cls=i===SG_IDX?' class="sg"':'';h+=`<td${{cls}}>${{v.toFixed(4)}}</td>`;}});
    h+=`<td>${{fmt(d,true)}}</td><td style="font-size:.7rem;color:#666">${{n===0?"(全部)":[...rem].join(", ")}}</td></tr>`;
  }}
  document.getElementById("t4").innerHTML=h+`</tbody></table>`;
}}

renderT0();renderT1();renderT2();initT3();renderT4();
</script></body></html>"""

# ── 入口 ─────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="experiment_data")
    p.add_argument("--output",   default="topic_analysis")
    args = p.parse_args()

    print(f"\n[*] 读取数据: '{args.data_dir}'\n")
    data   = load_data(args.data_dir)
    models = get_models(data)
    print(f"\n[*] {len(models)} 个 model: {models}")

    print("[*] 计算分析…")
    A, topics = build_analysis(data, models)
    print(f"[*] {len(topics)} 个 topic")

    txt = build_text(A, topics, models)
    with open(args.output+".txt","w",encoding="utf-8") as f: f.write(txt)
    print(f"[*] 纯文本 → {args.output}.txt")

    html = build_html(A, topics, models)
    with open(args.output+".html","w",encoding="utf-8") as f: f.write(html)
    print(f"[*] HTML   → {args.output}.html")
    print("\n[*] 完成。\n")

if __name__ == "__main__":
    main()
