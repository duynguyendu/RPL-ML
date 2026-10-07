#!/usr/bin/env python3
"""Cross-run comparison plots: how #nodes and offered load affect each metric.

Reads ``aggregate.json`` (written by plot_metrics.py) and ``config.json``
(written by pipeline.py) from every run directory under ``--runs-dir`` and emits
a self-contained ``comparison.html`` dashboard, styled like ``dashboard.html``.

Per metric (PDR / latency / CPU usage / parent switch / DIO sent per minute) the page offers two
views -- fix #nodes (x = bps per node) or fix bps per node (x = #nodes) --
each a box-and-whisker candle per objective function. The load axis is the
config's ``bps`` (bps per node); runs from before ``bps`` existed
fall back to their PPM restated as ppm * packet_size * 8 / 60.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plotly_utils import plotly_src

# statistics kept per metric in aggregate.json (written by plot_metrics.py)
AGGREGATIONS = ["min", "q1", "median", "avg", "q3", "max"]


def bits_per_second(cfg: dict) -> float | int | None:
    """Offered load per node in bit/s: the config's bps, or for older runs
    packets/min * bytes/packet * 8 / 60 s."""
    if cfg.get("bps") is not None:
        return cfg["bps"]
    ppm, packet_size = cfg.get("ppm"), cfg.get("packet_size")
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
        bps = bits_per_second(cfg)
        if cfg.get("num_of_nodes") is None or bps is None:
            print(f"  Skipping {run_dir.name}: missing num_of_nodes / bps")
            continue
        try:
            etx = round(1 / (cfg["success_tx"] * cfg["success_rx"]), 2)
        except (KeyError, TypeError, ZeroDivisionError):
            etx = None
        runs.append(
            {
                "run_id": run_dir.name,
                "num_of_nodes": cfg.get("num_of_nodes"),
                "bps": bps,
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
                "predict_us": agg.get("total", {}).get("predict_us"),
                "predict_count": agg.get("total", {}).get("predict_count"),
            }
        )
    return runs


# Config keys not swept by simulate.sh (those are num_of_nodes / bps / rpl_of /
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


# display order for known OFs (mirrors OF_ORDER in the page script)
OF_ORDER = ["mhrof", "of0", "mlof_lgbm", "mlof_dtree"]


def compute_predict_time(runs: list[dict]) -> list[tuple[str, object]]:
    """Average MLOF predict_pdr() run time per rpl_of, over all its runs' calls."""
    items = []
    ofs = {r["rpl_of"] for r in runs if r.get("rpl_of") is not None}
    for of in sorted(ofs, key=lambda o: (OF_ORDER.index(o) if o in OF_ORDER else len(OF_ORDER), o)):
        of_runs = [r for r in runs if r.get("rpl_of") == of and r.get("predict_count")]
        if not of_runs:
            continue
        us = sum(r["predict_us"] for r in of_runs)
        count = sum(r["predict_count"] for r in of_runs)
        label = "MRHOF" if of == "mhrof" else of.upper()
        items.append((label, f"{us / count:.1f} us ({len(of_runs)} runs)"))
    return items


def _render_strip(title: str, items: list[tuple[str, object]], bg: str = "#fdfdfe") -> str:
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
  header{padding:12px 16px;background:#fdfdfe;border-bottom:1px solid #e0e0e0;}
  header h1{font-size:16px;margin:0 0 4px;}
  header .meta{font-size:12px;color:#5a6b8c;}
  .controls{display:flex;flex-wrap:wrap;gap:14px 26px;align-items:center;
    padding:12px 16px;background:#fbfcfd;border-bottom:1px solid #e0e0e0;font-size:13px;}
  .controls fieldset{border:none;margin:0;padding:0;display:flex;gap:10px;align-items:center;}
  .controls legend{font-weight:bold;padding:0;margin-right:6px;}
  .controls label{display:inline-flex;gap:4px;align-items:center;white-space:nowrap;}
  select{font-size:13px;padding:2px 4px;}
  .note{font-size:11px;color:#5a6b8c;padding:6px 16px;background:#fdfdfe;
    border-bottom:1px solid #e0e0e0;}
  #pickLink{font-weight:bold;color:#4363d8;text-decoration:none;}
  #pickLink:hover{text-decoration:underline;}
  #pickMissing{color:#b00;}
  .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;padding:8px;}
  .card{border:1px solid #ececec;background:#fff;}
  .chart-title{font-size:16px;font-weight:bold;margin:0;padding:10px 14px 0;color:#2a3f5f;}
  .chart{height:460px;}
  .card.wide{grid-column:1 / -1;}
  @media (max-width:1300px){.grid{grid-template-columns:1fr;}}
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
    <label><input type="radio" name="mode" value="fix_bps"> Fix bps per node</label>
    <label><input type="radio" name="mode" value="facet_nodes"> Mean &plusmn; 95% CI by #nodes</label>
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
    <label>bps per node <select id="pickBps"></select></label>
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
  {key:'pdr',           file:'pdr',           label:'PDR',           scale:100, unit:'%',  title:'PDR',                        axis:'PDR (%)'},
  {key:'parent_switch', file:'parent_switch', label:'Parent switch', scale:1,   unit:'',   title:'Parent switches per node',   axis:'Parent switches per node'},
  {key:'latency',       file:'latency',       label:'Latency',       scale:1,   unit:' s', title:'End-to-end latency',         axis:'End-to-end latency (s)'},
  {key:'cpu_util',      file:'cpu',           label:'CPU usage',     scale:1,   unit:'%',  title:'CPU usage',                  axis:'CPU usage (%)'},
  {key:'dio_per_min',   file:'dio_sent',      label:'DIO sent',      scale:1,   unit:'/min', title:'DIO transmission rate per node', axis:'DIO messages per node per minute'},
  {key:'hop_count',     file:'hop_count',     label:'Hop count',     scale:1,   unit:'',   title:'Average hop count',          axis:'Average hop count'},
];
// Cycled by index rather than keyed by name, so any number of distinct
// rpl_of values present in RUNS (not just of0/mhrof) gets its own color.
const OF_PALETTE = [
  {color:'#4363d8', fill:'rgba(67,99,216,0.45)'},
  {color:'#e6194b', fill:'rgba(230,25,75,0.45)'},
  {color:'#3cb44b', fill:'rgba(60,180,75,0.45)'},
  {color:'#f58231', fill:'rgba(245,130,49,0.45)'},
  {color:'#911eb4', fill:'rgba(145,30,180,0.45)'},
  {color:'#46f0f0', fill:'rgba(70,240,240,0.45)'},
];

const $ = s => document.querySelector(s);
const uniqNums = a => [...new Set(a)].filter(v => v != null).sort((x, y) => x - y);
const uniqStrs = a => [...new Set(a)].filter(v => v != null).sort();
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
// display order for known OFs; any other rpl_of follows, alphabetically
const OF_ORDER = ['mhrof', 'of0', 'mlof_lgbm', 'mlof_dtree'];
const ofRank = of_ => { const i = OF_ORDER.indexOf(of_); return i < 0 ? OF_ORDER.length : i; };
const uniqOfs = a => uniqStrs(a).sort((x, y) => ofRank(x) - ofRank(y));
const RPL_OFS = uniqOfs(RUNS.map(r => r.rpl_of));
// legend / picker name for an rpl_of value, e.g. mhrof -> MRHOF, mlof_dtree -> MLOF_DTREE
const ofLabel = of_ => (of_ === 'mhrof') ? 'MRHOF' : String(of_).toUpperCase();
const ofStyle = of_ => OF_PALETTE[RPL_OFS.indexOf(of_) % OF_PALETTE.length];

function init(){
  if(!RUNS.length){ $('#charts').hidden = true; $('#empty').hidden = false; return; }
  $('#meta').textContent =
    RUNS.length + ' runs · #nodes: ' + uniqNums(RUNS.map(r => r.num_of_nodes)).join(', ')
    + ' · bps per node: ' + uniqNums(RUNS.map(r => r.bps)).join(', ')
    + ' · seeds: ' + uniqNums(RUNS.map(r => r.seed)).join(', ');

  METRICS.forEach((m, i) => {
    const card = document.createElement('div');
    card.className = 'card';
    const h = document.createElement('h2');
    h.className = 'chart-title';
    h.id = 'chartTitle' + i;
    const d = document.createElement('div');
    d.className = 'chart';
    d.id = 'chart' + i;
    card.append(h, d);
    $('#charts').appendChild(card);
  });

  document.querySelectorAll('input[name=mode]').forEach(el => el.addEventListener('change', render));
  $('#fixed').addEventListener('change', render);
  initRunPicker();
  render();
}

// dropdowns to pick one run's (#nodes, bps per node, OF, seed) and link to its dashboard.html.
// Each select is rebuilt from only the runs still matching the selects "above" it, so
// every reachable combination corresponds to a real run -- no dead-end picks possible.
const PICK_CHAIN = [
  {id: '#pickNodes', key: 'num_of_nodes', uniq: uniqNums},
  {id: '#pickBps', key: 'bps', uniq: uniqNums},
  {id: '#pickOf', key: 'rpl_of', uniq: uniqOfs, label: ofLabel},
  {id: '#pickSeed', key: 'seed', uniq: uniqNums},
];

function fillSelect(id, vals, label = v => v){
  const el = $(id);
  const prev = el.value;
  el.innerHTML = vals.map(v => '<option value="' + v + '">' + label(v) + '</option>').join('');
  if(vals.map(String).includes(prev)) el.value = prev;
}

// rebuild every select from `from` onward, each scoped by the (now-fixed) selects before it
function cascadePicker(from){
  let scoped = RUNS;
  for(let i = 0; i < PICK_CHAIN.length; i++){
    const f = PICK_CHAIN[i];
    if(i >= from) fillSelect(f.id, f.uniq(scoped.map(r => r[f.key])), f.label);
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
  if(m === 'dashboard' || m === 'facet_nodes'){ wrap.style.display = 'none'; return; }
  wrap.style.display = '';
  const key = (m === 'fix_nodes') ? 'num_of_nodes' : 'bps';
  $('#fixedLabel').textContent = (m === 'fix_nodes') ? '# nodes' : 'bps per node';
  const vals = uniqNums(RUNS.map(r => r[key]));
  const prev = $('#fixed').value;
  $('#fixed').innerHTML = vals.map(v => '<option value="' + v + '">' + v + '</option>').join('');
  if(vals.map(String).includes(prev)) $('#fixed').value = prev;
}

function valueOf(r, agg, key, scale){
  const raw = (r.agg && r.agg[agg]) ? r.agg[agg][key] : null;
  return (raw == null || Number.isNaN(raw)) ? null : raw * scale;
}

// free-axis values (bps per node, or node counts) present for the current fixed view
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
      type:'box', name:ofLabel(of_), x:xArr,
      lowerfence:lo, q1:q1, median:med, q3:q3, upperfence:hi, mean:mn,
      boxmean:true, whiskerwidth:0.5,
      marker:{color:style.color},
      line:{color:style.color, width:1.5},
      fillcolor:style.fill,
    });
  }
  return out;
}

// PNG export sized for the thesis: the page is 155 mm (6.1 in) wide and
// figures go in at \textwidth, so an EXPORT_W px wide chart prints 18 px text
// at ~9 pt; the scale renders that width at 300 ppi
const EXPORT_W = 880;
const EXPORT_SCALE = 6.1 * 300 / EXPORT_W;

// per-OF marker shape and line dash, so overlapping points stay distinguishable
// without colour (and in greyscale print)
const OF_SYMBOL = ['circle', 'square', 'diamond', 'triangle-up', 'cross', 'star'];
const OF_DASH = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot'];

// pull a hex colour ~20% of the way towards its grey to soften the saturation
function mute(hex, k = 0.2){
  const c = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
  const g = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
  return 'rgb(' + c.map(v => Math.round(v + (g - v) * k)).join(',') + ')';
}

// two-sided 95% Student-t critical value for df degrees of freedom: table for
// df 1-30, Cornish-Fisher expansion beyond (within 0.001 of the exact value)
const T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042];
function tCrit95(df){
  if(df < 1) return NaN;
  if(df <= T95.length) return T95[df - 1];
  const z = 1.959964;
  return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df * df);
}

// mean and 95% CI half-width of the per-run average of `metric` over `rows`
function meanCI(rows, metric){
  const v = rows.map(r => valueOf(r, 'avg', metric.key, metric.scale)).filter(x => x != null);
  if(!v.length) return null;
  const mu = mean(v);
  if(v.length < 2) return {mean:mu, ci:0, n:v.length};
  const sd = Math.sqrt(v.reduce((s, x) => s + (x - mu) ** 2, 0) / (v.length - 1));
  return {mean:mu, ci:tCrit95(v.length - 1) * sd / Math.sqrt(v.length), n:v.length};
}

// one chart per metric with a panel per node count: x = bps, one line per OF
// through the mean of the per-run averages, error bars = 95% CI over runs
function renderFacets(){
  const nodes = uniqNums(RUNS.map(r => r.num_of_nodes));
  const bpsAll = uniqNums(RUNS.map(r => r.bps));
  const gap = 0.015, w = (1 - gap * (nodes.length - 1)) / nodes.length;
  const step = bpsAll.length > 1 ? Math.min(...bpsAll.slice(1).map((b, i) => b - bpsAll[i])) : 16;
  // drawn at the export size so the page shows exactly what the PNG will be
  const chartHeight = 520;
  METRICS.forEach((metric, i) => {
    const traces = [];
    const lay = {
      width:EXPORT_W, height:chartHeight, autosize:false,
      margin:{l:80, r:10, t:80, b:70},
      font:{size:18},
      legend:{orientation:'h', x:0.5, xanchor:'center', y:1.09, yanchor:'bottom', font:{size:16}},
      paper_bgcolor:'white', plot_bgcolor:'#FBFCFE',
      annotations:[],
    };
    nodes.forEach((n, k) => {
      const ax = k ? String(k + 1) : '';
      const lo = k * (w + gap);
      lay['xaxis' + ax] = {
        domain:[lo, lo + w], anchor:'y' + ax,
        tickmode:'array', tickvals:bpsAll, ticktext:bpsAll.map(String), tickfont:{size:16},
        range:[bpsAll[0] - step * 0.5, bpsAll[bpsAll.length - 1] + step * 0.5],
        showgrid:true, gridcolor:'rgba(128,128,128,0.3)', zeroline:false,
      };
      lay['yaxis' + ax] = {
        anchor:'x' + ax, tickfont:{size:16}, gridcolor:'rgba(128,128,128,0.3)',
        rangemode:'tozero', nticks:12,
      };
      if(k){ lay['yaxis' + ax].matches = 'y'; lay['yaxis' + ax].showticklabels = false; }
      lay.annotations.push({
        text:'<b>N = ' + n + '</b>', xref:'paper', yref:'paper', showarrow:false,
        x:lo + w / 2, xanchor:'center', y:1.0, yanchor:'bottom', font:{size:17},
      });
      for(const of_ of RPL_OFS){
        const pts = bpsAll.map(b => [b, meanCI(RUNS.filter(r =>
          r.num_of_nodes === n && r.bps === b && r.rpl_of === of_), metric)])
          .filter(([, c]) => c);
        if(!pts.length) continue;
        const color = mute(ofStyle(of_).color);
        const j = RPL_OFS.indexOf(of_);
        traces.push({
          type:'scatter', mode:'lines+markers', xaxis:'x' + ax, yaxis:'y' + ax,
          name:ofLabel(of_), legendgroup:of_, showlegend:k === 0,
          x:pts.map(([b]) => b), y:pts.map(([, c]) => c.mean),
          customdata:pts.map(([b, c]) => [b, c.ci, c.n]),
          error_y:{type:'data', array:pts.map(([, c]) => c.ci), thickness:1, width:3, color:color},
          line:{color:color, width:1.25, dash:OF_DASH[j % OF_DASH.length]},
          marker:{color:color, size:8, symbol:OF_SYMBOL[j % OF_SYMBOL.length],
                  line:{color:'white', width:1}},
          hovertemplate:ofLabel(of_) + ', N = ' + n + ', %{customdata[0]} bps<br>' +
            'mean: %{y:.3f} &plusmn; %{customdata[1]:.3f} (%{customdata[2]} runs)<extra></extra>',
        });
      }
    });
    lay.xaxis.title = {text:'Traffic rate per node (bps)', font:{size:18, weight:'bold'}};
    // centre the shared x title under the middle panel
    if(nodes.length === 3){ lay.xaxis2.title = lay.xaxis.title; delete lay.xaxis.title; }
    lay.yaxis.title = {text:metric.axis, font:{size:18, weight:'bold'}};

    const div = document.getElementById('chart' + i);
    div.style.height = chartHeight + 'px';
    div.style.width = EXPORT_W + 'px';
    div.style.margin = '0 auto';
    document.getElementById('chartTitle' + i).textContent =
      metric.title + ' versus traffic rate per node (mean ± 95% CI over runs)';
    Plotly.react(div, traces, lay,
      {responsive:false, displaylogo:false,
       toImageButtonOptions:{format:'png', filename:metric.file + '_all_nodes',
                             width:EXPORT_W, height:chartHeight, scale:EXPORT_SCALE}});
  });
}

function layout(metric, m, fixedVal, chartHeight){
  const base = {
    height:chartHeight,
    margin:{l:90, r:16, t:56, b:80},
    font:{size:18},
    // horizontal legend centred just above the plot area
    legend:{orientation:'h', x:0.5, xanchor:'center', y:1.02, yanchor:'bottom', font:{size:14}},
    paper_bgcolor:'white', plot_bgcolor:'#FBFCFE',
  };
  const xtitle = (m === 'fix_nodes') ? 'Traffic rate per node (bps)' : 'Number of nodes';
  const xname = (m === 'fix_nodes') ? 'traffic rate per node' : 'number of nodes';
  const fixtxt = (m === 'fix_nodes')
    ? ('N = ' + fixedVal + ' nodes') : (fixedVal + ' bps per node');
  const cats = fixedXs(m, fixedVal).map(String);
  base.chartTitle = metric.title + ' versus ' + xname + ' (' + fixtxt + ')';
  base.boxmode = 'group';
  base.xaxis = {
    title:{text:xtitle, font:{size:19, weight:'bold'}}, type:'category',
    tickfont:{size:17},
    categoryorder:'array', categoryarray:cats,
    tickmode:'array', tickvals:cats,
    showgrid:true, gridcolor:'rgba(128,128,128,0.3)',
  };
  base.yaxis = {
    title:{text:metric.axis, font:{size:19, weight:'bold'}}, tickfont:{size:17},
    gridcolor:'rgba(128,128,128,0.3)',
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
  document.querySelectorAll('.card').forEach(c => c.classList.toggle('wide', m === 'facet_nodes'));
  if(m === 'facet_nodes'){ renderFacets(); return; }
  const chartHeight = 736;
  const fixedVal = Number($('#fixed').value);
  METRICS.forEach((metric, i) => {
    const div = document.getElementById('chart' + i);
    div.style.height = chartHeight + 'px';
    div.style.width = '';
    div.style.margin = '';
    const lay = layout(metric, m, fixedVal, chartHeight);
    document.getElementById('chartTitle' + i).textContent = lay.chartTitle;
    delete lay.chartTitle;
    // "Download plot as PNG" file name: <metric>_<#nodes>, or <metric>_<bps>bps
    // in the fixed-bps view so the two views never share a name
    const filename = metric.file + '_' + fixedVal + (m === 'fix_nodes' ? '' : 'bps');
    Plotly.react(div, tracesCandle(metric, m, fixedVal), lay,
      {responsive:true, displaylogo:false,
       toImageButtonOptions:{format:'png', filename:filename,
                             width:EXPORT_W, height:chartHeight, scale:EXPORT_SCALE}});
  });
}

init();
</script>
</body>
</html>
"""


def build_html(runs: list[dict], plotly_js_src: str) -> str:
    fixed_config_html = _render_strip(
        "Fixed config", compute_fixed_config(runs)
    ) + _render_strip("Avg predict_pdr() time", compute_predict_time(runs))
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
