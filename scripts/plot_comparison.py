#!/usr/bin/env python3
"""Cross-run comparison plots: how #nodes and offered load affect each metric.

Reads ``aggregate.json`` (written by plot_metrics.py) and ``config.json``
(written by pipeline.py) from every run directory under ``--runs-dir`` and emits
a self-contained ``comparison.html`` dashboard, styled like ``dashboard.html``.

Per metric (PDR / latency / CPU util / parent switch / DIO sent per minute) the page offers two
views -- fix #nodes (x = bit/s per node) or fix bit/s per node (x = #nodes) --
each a box-and-whisker candle per objective function. The load axis is the
config's PPM restated as bits per second per node: ppm * packet_size * 8 / 60.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plotly_utils import plotly_src

# statistics kept per metric in aggregate.json (written by plot_metrics.py)
AGGREGATIONS = ["min", "q1", "median", "avg", "q3", "max"]


def bits_per_second(ppm: object, packet_size: object) -> float | int | None:
    """Offered load per node in bit/s: packets/min * bytes/packet * 8 / 60 s."""
    if ppm is None or packet_size is None:
        return None
    bps = round(ppm * packet_size * 8 / 60, 2)
    return int(bps) if bps == int(bps) else bps


def collect_runs(runs_dir: Path) -> list[dict]:
    """Gather (config.json, aggregate.json) pairs from every run directory."""
    runs: list[dict] = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        cfg_path = run_dir / "config.json"
        agg_path = run_dir / "aggregate.json"
        if not (cfg_path.exists() and agg_path.exists()):
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
            agg = json.loads(agg_path.read_text())
        except json.JSONDecodeError as exc:
            print(f"  Skipping {run_dir.name}: {exc}")
            continue
        if cfg.get("num_of_nodes") is None or cfg.get("ppm") is None:
            print(f"  Skipping {run_dir.name}: missing num_of_nodes / ppm")
            continue
        try:
            etx = round(1 / (cfg["success_tx"] * cfg["success_rx"]), 2)
        except (KeyError, TypeError, ZeroDivisionError):
            etx = None
        runs.append(
            {
                "run_id": run_dir.name,
                "num_of_nodes": cfg.get("num_of_nodes"),
                "ppm": cfg.get("ppm"),
                "bps": bits_per_second(cfg.get("ppm"), cfg.get("packet_size")),
                "seed": cfg.get("seed"),
                "rpl_of": cfg.get("rpl_of"),
                "topo_type": cfg.get("topo_type"),
                "platform": cfg.get("platform"),
                "packet_size": cfg.get("packet_size"),
                "buffer_size": cfg.get("buffer_size"),
                "duration": cfg.get("duration"),
                "interference_range": cfg.get("interference_range"),
                "etx": etx,
                "agg": {k: agg.get(k, {}) for k in AGGREGATIONS},
            }
        )
    return runs


# Config keys not swept by simulate.sh (those are num_of_nodes / ppm / rpl_of /
# seed, already surfaced by the page's own controls) but still worth knowing
# at a glance, mirroring dashboard.html's "Run config" strip.
FIXED_CONFIG_KEYS = [
    "packet_size",
    "buffer_size",
    "duration",
    "interference_range",
    "topo_type",
    "platform",
    "etx",
]


def compute_fixed_config(runs: list[dict]) -> list[tuple[str, object]]:
    """Values of FIXED_CONFIG_KEYS, or "mixed" where runs disagree."""
    items = []
    for key in FIXED_CONFIG_KEYS:
        values = {r[key] for r in runs if r.get(key) is not None}
        if not values:
            continue
        items.append((key, values.pop() if len(values) == 1 else "mixed"))
    return items


def _render_strip(title: str, items: list[tuple[str, object]], bg: str = "#f6f8fa") -> str:
    """Render one labelled key/value strip, styled like dashboard.html's."""
    if not items:
        return ""
    body = "".join(
        f'<span style="margin-right:18px;white-space:nowrap;"><b>{k}</b>: {v}</span>'
        for k, v in items
    )
    return (
        '<div style="font-family:Arial;font-size:13px;color:#2a3f5f;'
        f"padding:10px 16px;background:{bg};border-bottom:1px solid #e0e0e0;"
        'display:flex;flex-wrap:wrap;align-items:center;">'
        f'<b style="margin-right:18px;">{title}</b>'
        f"{body}</div>"
    )


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="__PLOTLY_SRC__"></script>
<style>
  body{font-family:Arial,Helvetica,sans-serif;margin:0;background:#fff;color:#2a3f5f;}
  header{padding:12px 16px;background:#f6f8fa;border-bottom:1px solid #e0e0e0;}
  header h1{font-size:16px;margin:0 0 4px;}
  header .meta{font-size:12px;color:#5a6b8c;}
  .controls{display:flex;flex-wrap:wrap;gap:14px 26px;align-items:center;
    padding:12px 16px;background:#eef2f7;border-bottom:1px solid #e0e0e0;font-size:13px;}
  .controls fieldset{border:none;margin:0;padding:0;display:flex;gap:10px;align-items:center;}
  .controls legend{font-weight:bold;padding:0;margin-right:6px;}
  .controls label{display:inline-flex;gap:4px;align-items:center;white-space:nowrap;}
  select{font-size:13px;padding:2px 4px;}
  .note{font-size:11px;color:#5a6b8c;padding:6px 16px;background:#f6f8fa;
    border-bottom:1px solid #e0e0e0;}
  #pickLink{font-weight:bold;color:#4363d8;text-decoration:none;}
  #pickLink:hover{text-decoration:underline;}
  #pickMissing{color:#b00;}
  .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;padding:8px;}
  .chart{height:460px;border:1px solid #ececec;}
  @media (max-width:900px){.grid{grid-template-columns:1fr;}}
  .empty{padding:40px 16px;font-size:14px;color:#b00;}
  #dashboardArea{padding:8px;height:85vh;}
  #dashboardFrame{width:100%;height:100%;border:1px solid #ececec;}
  #dashboardMissing{padding:40px 16px;font-size:14px;color:#b00;}
</style>
</head>
<body>
<header>
  <h1>Metric comparison across runs</h1>
  <div class="meta" id="meta"></div>
</header>
__FIXED_CONFIG_HTML__
<div class="controls">
  <fieldset>
    <legend>View</legend>
    <label><input type="radio" name="mode" value="fix_nodes" checked> Fix #nodes</label>
    <label><input type="radio" name="mode" value="fix_bps"> Fix bit/s per node</label>
    <label><input type="radio" name="mode" value="dashboard"> Run dashboard</label>
  </fieldset>
  <fieldset id="fixedWrap">
    <legend id="fixedLabel">Value</legend>
    <select id="fixed"></select>
  </fieldset>
</div>
<div class="controls">
  <fieldset>
    <legend id="pickLegend">Pick a run</legend>
    <label>#nodes <select id="pickNodes"></select></label>
    <label>bit/s per node <select id="pickBps"></select></label>
    <label>OF <select id="pickOf"></select></label>
    <label>seed <select id="pickSeed"></select></label>
  </fieldset>
  <a id="pickLink" href="#" target="_blank" rel="noopener" hidden>Open in new tab &rarr;</a>
  <span id="pickMissing" hidden>No matching run</span>
</div>
<div class="note">A box-and-whisker candle per objective function (one per x
  value) &ndash; box = Q1&ndash;Q3, line = median, whiskers = min/max, dashed =
  mean. "Run dashboard" embeds the picked run's own dashboard.html below,
  in-page.</div>
<div id="charts" class="grid"></div>
<div id="dashboardArea" hidden>
  <iframe id="dashboardFrame" title="Run dashboard"></iframe>
  <div id="dashboardMissing" hidden>No matching run -- adjust the picker above.</div>
</div>
<div id="empty" class="empty" hidden>No run directories with both config.json and
  aggregate.json were found under the runs directory.</div>
<script>
const RUNS = __RUNS_JSON__;
const METRICS = [
  {key:'pdr',           label:'PDR',           scale:100, unit:'%',  axis:'PDR (%)'},
  {key:'latency',       label:'Latency',       scale:1,   unit:' s', axis:'Latency (s)'},
  {key:'cpu_util',      label:'CPU util',      scale:1,   unit:'%',  axis:'CPU util (%)'},
  {key:'parent_switch', label:'Parent switch', scale:1,   unit:'',   axis:'Parent switches (per node)'},
  {key:'dio_per_min',   label:'DIO sent',      scale:1,   unit:'/min', axis:'DIO sent per minute (per node)'},
];
// Cycled by index rather than keyed by name, so any number of distinct
// rpl_of values present in RUNS (not just of0/mhrof) gets its own color.
const OF_PALETTE = [
  {color:'#4363d8', fill:'rgba(67,99,216,0.30)'},
  {color:'#e6194b', fill:'rgba(230,25,75,0.28)'},
  {color:'#3cb44b', fill:'rgba(60,180,75,0.28)'},
  {color:'#f58231', fill:'rgba(245,130,49,0.28)'},
  {color:'#911eb4', fill:'rgba(145,30,180,0.28)'},
  {color:'#46f0f0', fill:'rgba(70,240,240,0.28)'},
];

const $ = s => document.querySelector(s);
const uniqNums = a => [...new Set(a)].filter(v => v != null).sort((x, y) => x - y);
const uniqStrs = a => [...new Set(a)].filter(v => v != null).sort();
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
const RPL_OFS = uniqStrs(RUNS.map(r => r.rpl_of));
const ofStyle = of_ => OF_PALETTE[RPL_OFS.indexOf(of_) % OF_PALETTE.length];

function init(){
  if(!RUNS.length){ $('#charts').hidden = true; $('#empty').hidden = false; return; }
  $('#meta').textContent =
    RUNS.length + ' runs · #nodes: ' + uniqNums(RUNS.map(r => r.num_of_nodes)).join(', ')
    + ' · bit/s per node: ' + uniqNums(RUNS.map(r => r.bps)).join(', ')
    + ' · seeds: ' + uniqNums(RUNS.map(r => r.seed)).join(', ');

  METRICS.forEach((m, i) => {
    const d = document.createElement('div');
    d.className = 'chart';
    d.id = 'chart' + i;
    $('#charts').appendChild(d);
  });

  document.querySelectorAll('input[name=mode]').forEach(el => el.addEventListener('change', render));
  $('#fixed').addEventListener('change', render);
  initRunPicker();
  render();
}

// dropdowns to pick one run's (#nodes, bit/s per node, OF, seed) and link to its dashboard.html.
// Each select is rebuilt from only the runs still matching the selects "above" it, so
// every reachable combination corresponds to a real run -- no dead-end picks possible.
const PICK_CHAIN = [
  {id: '#pickNodes', key: 'num_of_nodes', uniq: uniqNums},
  {id: '#pickBps', key: 'bps', uniq: uniqNums},
  {id: '#pickOf', key: 'rpl_of', uniq: uniqStrs},
  {id: '#pickSeed', key: 'seed', uniq: uniqNums},
];

function fillSelect(id, vals){
  const el = $(id);
  const prev = el.value;
  el.innerHTML = vals.map(v => '<option value="' + v + '">' + v + '</option>').join('');
  if(vals.map(String).includes(prev)) el.value = prev;
}

// rebuild every select from `from` onward, each scoped by the (now-fixed) selects before it
function cascadePicker(from){
  let scoped = RUNS;
  for(let i = 0; i < PICK_CHAIN.length; i++){
    const f = PICK_CHAIN[i];
    if(i >= from) fillSelect(f.id, f.uniq(scoped.map(r => r[f.key])));
    scoped = scoped.filter(r => String(r[f.key]) === $(f.id).value);
  }
}

function initRunPicker(){
  cascadePicker(0);
  PICK_CHAIN.forEach((f, i) => $(f.id).addEventListener('change', () => { cascadePicker(i + 1); updatePick(); }));
  updatePick();
}

function currentPickMatch(){
  return RUNS.find(r => PICK_CHAIN.every(f => String(r[f.key]) === $(f.id).value));
}

function updatePick(){
  const match = currentPickMatch();
  const link = $('#pickLink'), missing = $('#pickMissing');
  link.hidden = !match;
  missing.hidden = !!match;
  if(match) link.href = match.run_id + '/dashboard.html';
  updateDashboardFrame(match);
}

// loads the picked run's own dashboard.html into the in-page iframe (only
// while the "Run dashboard" view is active) instead of opening a new tab.
function updateDashboardFrame(match){
  if(mode() !== 'dashboard') return;
  match = match || currentPickMatch();
  const frame = $('#dashboardFrame'), missing = $('#dashboardMissing');
  if(match){
    const src = match.run_id + '/dashboard.html';
    if(frame.getAttribute('src') !== src) frame.src = src;
    frame.hidden = false;
    missing.hidden = true;
  } else {
    frame.hidden = true;
    missing.hidden = false;
  }
}

function mode(){ return document.querySelector('input[name=mode]:checked').value; }

function syncFixed(){
  const m = mode();
  const wrap = $('#fixedWrap');
  if(m === 'dashboard'){ wrap.style.display = 'none'; return; }
  wrap.style.display = '';
  const key = (m === 'fix_nodes') ? 'num_of_nodes' : 'bps';
  $('#fixedLabel').textContent = (m === 'fix_nodes') ? '# nodes' : 'bit/s per node';
  const vals = uniqNums(RUNS.map(r => r[key]));
  const prev = $('#fixed').value;
  $('#fixed').innerHTML = vals.map(v => '<option value="' + v + '">' + v + '</option>').join('');
  if(vals.map(String).includes(prev)) $('#fixed').value = prev;
}

function valueOf(r, agg, key, scale){
  const raw = (r.agg && r.agg[agg]) ? r.agg[agg][key] : null;
  return (raw == null || Number.isNaN(raw)) ? null : raw * scale;
}

// free-axis values (bit/s per node, or node counts) present for the current fixed view
function fixedXs(m, fixedVal){
  const freeKey = (m === 'fix_nodes') ? 'bps' : 'num_of_nodes';
  const inScope = r => (m === 'fix_nodes') ? r.num_of_nodes === fixedVal
                                           : r.bps === fixedVal;
  return uniqNums(RUNS.filter(inScope).map(r => r[freeKey]));
}

// one box-and-whisker candle per objective function, grouped per-x-value.
// Box = Q1..Q3, line = median, whiskers = min/max, dashed = mean. Runs that
// share the same (fixed value, free value, OF) are averaged stat-by-stat.
function tracesCandle(metric, m, fixedVal){
  const freeKey = (m === 'fix_nodes') ? 'bps' : 'num_of_nodes';
  const inScope = r => (m === 'fix_nodes') ? r.num_of_nodes === fixedVal
                                           : r.bps === fixedVal;
  const xs = fixedXs(m, fixedVal);
  const out = [];
  for(const of_ of RPL_OFS){
    const xArr = [], lo = [], q1 = [], med = [], q3 = [], hi = [], mn = [];
    for(const xv of xs){
      const rows = RUNS.filter(r => inScope(r) && r.rpl_of === of_ && r[freeKey] === xv);
      if(!rows.length) continue;
      const stat = k => {
        const v = rows.map(r => valueOf(r, k, metric.key, metric.scale)).filter(x => x != null);
        return v.length ? mean(v) : null;
      };
      const s = [stat('min'), stat('q1'), stat('median'), stat('q3'), stat('max'), stat('avg')];
      if(s.some(v => v == null)) continue;
      xArr.push(String(xv));
      lo.push(s[0]); q1.push(s[1]); med.push(s[2]); q3.push(s[3]); hi.push(s[4]); mn.push(s[5]);
    }
    if(!xArr.length) continue;
    const style = ofStyle(of_);
    out.push({
      type:'box', name:of_, x:xArr,
      lowerfence:lo, q1:q1, median:med, q3:q3, upperfence:hi, mean:mn,
      boxmean:true, whiskerwidth:0.5,
      marker:{color:style.color},
      line:{color:style.color, width:1.5},
      fillcolor:style.fill,
    });
  }
  return out;
}

function layout(metric, m, fixedVal, chartHeight){
  const base = {
    height:chartHeight,
    margin:{l:80, r:16, t:56, b:70},
    font:{size:16},
    legend:{orientation:'h', y:-0.15, yanchor:'top', font:{size:15}},
    paper_bgcolor:'white', plot_bgcolor:'#E5ECF6',
    title:{text:'', font:{size:20}},
  };
  const xtitle = (m === 'fix_nodes') ? 'bits per second per node' : 'number of nodes';
  const fixtxt = (m === 'fix_nodes')
    ? ('number of nodes = ' + fixedVal) : ('bit/s per node = ' + fixedVal);
  const cats = fixedXs(m, fixedVal).map(String);
  base.title.text = metric.label + ' vs ' + xtitle + '  (' + fixtxt + ')';
  base.boxmode = 'group';
  base.xaxis = {
    title:{text:xtitle}, type:'category',
    categoryorder:'array', categoryarray:cats,
    tickmode:'array', tickvals:cats,
    showgrid:true, gridcolor:'rgba(128,128,128,0.3)',
  };
  base.yaxis = {
    title:{text:metric.axis}, gridcolor:'rgba(128,128,128,0.3)',
    rangemode:'tozero', nticks:12,
  };
  return base;
}

function render(){
  syncFixed();
  const m = mode();
  $('#charts').style.display = (m === 'dashboard') ? 'none' : '';
  $('#dashboardArea').hidden = (m !== 'dashboard');
  if(m === 'dashboard'){ updateDashboardFrame(); return; }
  const chartHeight = 736;
  const fixedVal = Number($('#fixed').value);
  METRICS.forEach((metric, i) => {
    const div = document.getElementById('chart' + i);
    div.style.height = chartHeight + 'px';
    Plotly.react(div, tracesCandle(metric, m, fixedVal), layout(metric, m, fixedVal, chartHeight),
      {responsive:true, displaylogo:false});
  });
}

init();
</script>
</body>
</html>
"""


def build_html(runs: list[dict], plotly_js_src: str) -> str:
    fixed_config_html = _render_strip("Fixed config", compute_fixed_config(runs))
    return (
        HTML_TEMPLATE.replace("__TITLE__", "Metric comparison across runs")
        .replace("__PLOTLY_SRC__", plotly_js_src)
        .replace("__FIXED_CONFIG_HTML__", fixed_config_html)
        .replace("__RUNS_JSON__", json.dumps(runs).replace("</", "<\\/"))
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--runs-dir", default="runs", type=Path, help="directory of run subdirs")
    ap.add_argument(
        "--output",
        default=None,
        type=Path,
        help="output HTML path (default: <runs-dir>/comparison.html)",
    )
    args = ap.parse_args()

    if not args.runs_dir.is_dir():
        ap.error(f"runs dir not found: {args.runs_dir}")
    out_path = args.output or (args.runs_dir / "comparison.html")

    runs = collect_runs(args.runs_dir)
    print(f"Collected {len(runs)} run(s) with aggregate.json from {args.runs_dir}/")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_html(runs, plotly_src(out_path.parent, args.runs_dir)))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
