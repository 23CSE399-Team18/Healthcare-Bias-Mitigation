"""Individual before/after bias dashboards in the style of the project's reference pages
(D:/plots): navy header, sidebar (plot type, variable, dataset info, legend), distribution and
box-plot grids, correlation heatmaps, Q-Q plot, one bias-specific panel and the full method
comparison table. Self-contained HTML + inline SVG; no external scripts; no patient rows embedded
(only binned counts, quantiles and summary statistics).
"""
from __future__ import annotations

import html
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from common import clean_json

BIAS_FILES = [("missing", "01_missing_data_bias_dashboard.html", "Missing Data Bias"),
              ("imbalance", "02_class_imbalance_bias_dashboard.html", "Class Imbalance Bias"),
              ("outlier", "03_outlier_bias_dashboard.html", "Outlier Bias"),
              ("scale", "04_feature_scale_bias_dashboard.html", "Feature Scale Bias"),
              ("demographic", "05_demographic_bias_dashboard.html", "Demographic Bias"),
              ("duplicate", "06_duplicate_bias_dashboard.html", "Duplicate Bias"),
              ("noise", "07_data_noise_bias_dashboard.html", "Data Noise Bias"),
              ("domain", "08_encoding_domain_bias_dashboard.html", "Encoding / Domain Bias")]


def _hist(values, edges):
    v = values[np.isfinite(values)]
    return np.histogram(np.clip(v, edges[0], edges[-1]), edges)[0].tolist(), int(len(v))


def _box(values):
    v = np.sort(values[np.isfinite(values)])
    if len(v) == 0:
        return None
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = v[(v >= lo_f) & (v <= hi_f)]
    out = v[(v < lo_f) | (v > hi_f)]
    shown = np.r_[out[:20], out[-20:]] if len(out) > 40 else out
    return {"q1": float(q1), "median": float(med), "q3": float(q3), "whisk_lo": float(inside.min()),
            "whisk_hi": float(inside.max()), "outliers": np.unique(np.round(shown, 4)).tolist(), "n_outliers": int(len(out)),
            "n": int(len(v)), "mean": float(v.mean())}


def _qq(values, k=50):
    v = values[np.isfinite(values)]
    if len(v) < 3:
        return None
    p = (np.arange(1, k + 1) - 0.5) / k
    return {"theoretical": norm.ppf(p).round(4).tolist(), "sample": np.quantile(v, p).round(5).tolist()}


def variable_payload(before, after, separate_axes=False, bins=30):
    """before/after: 1-D arrays (may contain NaN). Shares are per series so row counts can differ."""
    b, a = np.asarray(before, float), np.asarray(after, float)
    fin = np.r_[b[np.isfinite(b)], a[np.isfinite(a)]]
    out = {"box_before": _box(b), "box_after": _box(a), "qq_before": _qq(b), "qq_after": _qq(a),
           "missing_before": float(np.isnan(b).mean()), "missing_after": float(np.isnan(a).mean()),
           "separate_axes": separate_axes}
    if len(fin) == 0:
        return out
    if separate_axes:
        for key, v in (("before", b), ("after", a)):
            f = v[np.isfinite(v)]
            e = np.linspace(f.min(), f.max() if f.max() > f.min() else f.min() + 1, bins + 1)
            out[f"edges_{key}"] = e.round(5).tolist()
            out[f"hist_{key}"], out[f"n_{key}"] = _hist(v, e)
    else:
        lo, hi = np.percentile(fin, [0, 100])
        uniq = np.unique(fin)
        if len(uniq) <= bins:                       # discrete: one bin per value
            step = np.min(np.diff(uniq)) if len(uniq) > 1 else 1.0
            e = np.r_[uniq - step / 2, uniq[-1] + step / 2]
        else:
            e = np.linspace(lo, hi if hi > lo else lo + 1, bins + 1)
        out["edges"] = e.round(5).tolist()
        out["hist_before"], out["n_before"] = _hist(b, e)
        out["hist_after"], out["n_after"] = _hist(a, e)
    return out


def changed_score(before, after):
    b, a = np.asarray(before, float), np.asarray(after, float)
    b, a = b[np.isfinite(b)], a[np.isfinite(a)]
    if len(b) == 0 or len(a) == 0:
        return 0.0
    q = np.linspace(0.01, 0.99, 25)
    sd = np.std(b) or 1.0
    return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean() / sd + abs(len(a) - len(b)) / max(len(b), 1))


def core_payload(Xb, Xa, variables, labels, separate_axes=False, max_grid=12, priority=None):
    """Distribution/box/Q-Q data for every variable, grid = most changed (or priority) variables."""
    per = {v: variable_payload(Xb[v].to_numpy(float), Xa[v].to_numpy(float), separate_axes) for v in variables}
    scores = {v: changed_score(Xb[v], Xa[v]) for v in variables}
    order = list(priority or []) + sorted(variables, key=lambda v: -scores[v])
    grid = list(dict.fromkeys([v for v in order if v in variables]))[:max_grid]
    corr_vars = grid[:10]
    cb = np.nan_to_num(pd.DataFrame(Xb[corr_vars]).corr().to_numpy()).round(3).tolist() if corr_vars else []
    ca = np.nan_to_num(pd.DataFrame(Xa[corr_vars]).corr().to_numpy()).round(3).tolist() if corr_vars else []
    return {"variables": variables, "labels": labels, "per": per, "grid": grid, "corr_vars": corr_vars,
            "corr_before": cb, "corr_after": ca, "changed": scores}


def bin_matrix(flags, variables, n_bins=20):
    """Share of flagged cells per variable (rows) and record group (columns, row order)."""
    if len(flags) == 0:
        return [[0.0] * n_bins for _ in variables]
    groups = np.array_split(np.arange(len(flags)), n_bins)
    return [[float(flags[v].to_numpy()[g].mean()) if len(g) else 0.0 for g in groups] for v in variables]


def table_payload(table):
    t = table.copy()
    out = {"columns": [str(c) for c in t.columns], "rows": []}
    for name, row in t.iterrows():
        vals = []
        for c in t.columns:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                vals.append(None if not math.isfinite(float(v)) else round(float(v), 4))
            elif isinstance(v, (np.bool_, bool)):
                vals.append("yes" if v else "no")
            else:
                vals.append(str(v))
        out["rows"].append({"method": str(name), "values": vals})
    return out


# --------------------------------------------------------------------------- HTML
CSS = r"""
:root{--navy:#0B1E3F;--navy2:#132a52;--bg:#EEF1F7;--card:#FFFFFF;--before:#6FA0DA;--before-line:#3E6FA8;
--after:#7BC17E;--after-line:#4C9B50;--text:#1B2333;--muted:#6B7686;--border:#E1E5EE;--blue-hi:#2563C8;--red-hi:#D0433F;
--shadow:0 1px 3px rgba(15,23,42,.06),0 1px 2px rgba(15,23,42,.08);}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--text)}
header.topbar{background:linear-gradient(180deg,var(--navy),var(--navy2));color:#fff;padding:18px 28px;text-align:center;box-shadow:var(--shadow)}
header.topbar h1{margin:0;font-size:22px;letter-spacing:.2px;font-weight:700}header.topbar .sub{margin-top:5px;font-size:12.5px;color:#B9C6E6}
header.topbar a{color:#cfe0ff}
.layout{display:grid;grid-template-columns:280px 1fr;gap:16px;padding:16px;align-items:start}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;box-shadow:var(--shadow);padding:16px}
.side .card{margin-bottom:14px}.side-label{font-size:12.5px;font-weight:700;color:var(--blue-hi);margin:0 0 8px 0}
select.control{width:100%;padding:8px 10px;border:1px solid var(--border);border-radius:6px;background:#fff;font-size:13px;color:var(--text)}
.varlist{max-height:230px;overflow-y:auto;border:1px solid var(--border);border-radius:6px;margin-top:8px}
.varlist .opt{padding:7px 10px;font-size:12.5px;cursor:pointer;border-bottom:1px solid #F0F2F7}
.varlist .opt:hover{background:#F3F6FC}.varlist .opt.active{background:var(--blue-hi);color:#fff;font-weight:600}
.info-title,.legend-title{font-size:13px;font-weight:700;margin:0 0 10px 0}
.info-row{display:flex;justify-content:space-between;gap:8px;font-size:12.3px;padding:4px 0;border-bottom:1px dashed #EEF0F5}
.info-row span.k{color:var(--muted)}.info-row b{text-align:right}
.legend-row{display:flex;align-items:center;gap:8px;font-size:12.5px;padding:3px 0}
.swatch{width:13px;height:13px;border-radius:3px;display:inline-block;flex:none}
.panel-title{font-size:14px;font-weight:700;color:var(--blue-hi);margin:0 0 12px 0;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px}
.legend-inline{display:flex;gap:14px;font-size:11.5px;color:var(--muted);align-items:center;font-weight:400}
.legend-inline .lg{display:flex;align-items:center;gap:5px}
.grid6{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
@media (max-width:1400px){.grid6{grid-template-columns:repeat(3,1fr)}}
@media (max-width:1100px){.grid6{grid-template-columns:repeat(2,1fr)}.layout{grid-template-columns:1fr}}
@media (max-width:640px){.grid6{grid-template-columns:1fr}.row2{grid-template-columns:1fr!important}header.topbar{padding:14px 16px}.layout{padding:10px}}
.chart-box{border:1px solid var(--border);border-radius:8px;padding:8px 8px 4px;background:#fbfcfe;cursor:pointer}
.chart-box.focus{outline:2px solid var(--blue-hi)}
.chart-box .ct{font-size:12px;font-weight:700;text-align:center;margin-bottom:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.chart-box .cs{font-size:10.5px;color:var(--muted);text-align:center}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}
.mt{margin-top:16px}svg text{font-family:inherit}
.corr-wrap{display:flex;gap:18px;flex-wrap:wrap;justify-content:center;overflow-x:auto}
.mini-title{text-align:center;font-weight:700;font-size:12.5px;margin-bottom:6px}
table.cmp{border-collapse:collapse;width:100%;font-size:12px}table.cmp th,table.cmp td{border:1px solid var(--border);padding:6px 7px;text-align:right}
table.cmp th{background:#F4F6FB;text-align:center;font-weight:700}table.cmp td:first-child,table.cmp td.txt{text-align:left}
table.cmp tr.sel td{background:#EAF3FF;font-weight:700}table.cmp tr.app td:first-child{box-shadow:inset 4px 0 0 var(--after-line)}
.tbl-wrap{overflow-x:auto}.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11.5px;font-weight:700}
.badge.yes{background:#FDECEC;color:var(--red-hi)}.badge.no{background:#E8F5E9;color:#2E7D32}.badge.na{background:#EEF0F5;color:var(--muted)}
.note{font-size:12px;color:var(--muted);line-height:1.5}.note b{color:var(--text)}
.summary-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}@media (max-width:1100px){.summary-grid{grid-template-columns:repeat(2,1fr)}}
.stat{border:1px solid var(--border);border-radius:8px;padding:10px;background:#fbfcfe}.stat .k{font-size:11.5px;color:var(--muted)}.stat .v{font-size:17px;font-weight:700;margin-top:3px}
#tooltip{position:fixed;pointer-events:none;background:#0B1E3F;color:#fff;font-size:11.5px;padding:5px 8px;border-radius:5px;display:none;z-index:10;max-width:320px}
footer.note{padding:6px 28px 22px;text-align:center}
"""

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title><style>__CSS__</style></head><body>
<div id="tooltip"></div>
<header class="topbar"><h1 id="ttl"></h1><div class="sub" id="sub"></div></header>
<div class="layout">
 <div class="side">
  <div class="card"><p class="side-label">1. Select Plot Type</p>
   <select id="plotType" class="control"><option value="hist">Histogram</option><option value="density">Density Curve</option></select></div>
  <div class="card"><p class="side-label">2. Select Variable (Q-Q plot / focus)</p>
   <select id="varDropdown" class="control"></select><div class="varlist" id="varList"></div></div>
  <div class="card"><p class="info-title">&#9432; Dataset Info</p><div id="info"></div></div>
  <div class="card"><p class="legend-title">&#128273; Legend</p>
   <div class="legend-row"><span class="swatch" style="background:var(--before);border:1.5px solid var(--before-line)"></span><span id="lgB">Before</span></div>
   <div class="legend-row"><span class="swatch" style="background:var(--after);border:1.5px solid var(--after-line)"></span><span id="lgA">After</span></div>
   <p class="note" id="legendNote"></p><p class="note"><a href="index.html">&#8592; All bias dashboards for this dataset</a></p></div>
 </div>
 <div class="main">
  <div class="card"><p class="panel-title">Bias summary</p><div class="summary-grid" id="summary"></div><p class="note" id="decision"></p></div>
  <div class="card mt"><p class="panel-title"><span id="distTitle">Distribution Comparison (Before vs After)</span>
   <span class="legend-inline"><span class="lg"><span class="swatch" style="background:var(--before)"></span>Before</span><span class="lg"><span class="swatch" style="background:var(--after)"></span>After</span></span></p>
   <p class="note" id="distNote"></p><div class="grid6" id="distGrid"></div></div>
  <div class="card mt"><p class="panel-title">Box Plot Comparison (Before vs After)</p><p class="note" id="boxNote"></p><div class="grid6" id="boxGrid"></div></div>
  <div class="row2">
   <div class="card"><p class="panel-title">Q-Q Plot (Normality Check) <span class="legend-inline" id="qqVarName"></span></p><div id="qqChart" style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap"></div></div>
   <div class="card"><p class="panel-title">Correlation Heatmap (Pearson)</p><div class="corr-wrap" id="corrChart"></div></div>
  </div>
  <div class="card mt"><p class="panel-title"><span id="specTitle"></span><span class="legend-inline" id="specChips"></span></p><p class="note" id="specNote"></p><div id="spec"></div></div>
  <div class="card mt"><p class="panel-title">Method Comparison (every method &times; every metric)
   <span class="legend-inline"><span class="lg"><span class="swatch" style="background:#EAF3FF;border:1px solid var(--blue-hi)"></span>Best mean rank</span><span class="lg"><span class="swatch" style="background:var(--after-line)"></span>Applied to the data</span></span></p>
   <div class="tbl-wrap" id="cmp"></div><p class="note" id="protocol"></p><div class="note" id="methodNotes"></div></div>
 </div>
</div>
<footer class="note" id="foot"></footer>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const D = JSON.parse(document.getElementById('payload').textContent);
const NS="http://www.w3.org/2000/svg";
function el(t,a){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e}
function txt(svg,x,y,s,a){const t=el('text',Object.assign({x:x,y:y,'font-size':9,fill:'#6B7686'},a||{}));t.textContent=s;svg.appendChild(t);return t}
const tip=document.getElementById('tooltip');
function showTip(e,s){tip.textContent=s;tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY-10)+'px'}
function hideTip(){tip.style.display='none'}
function hov(node,s){node.addEventListener('mousemove',e=>showTip(e,s));node.addEventListener('mouseleave',hideTip);return node}
function fmt(v,d){if(v===null||v===undefined)return 'n/a';if(typeof v!=='number')return String(v);const a=Math.abs(v);if(a!==0&&(a<0.001||a>=1e5))return v.toExponential(2);return v.toFixed(d===undefined?(a>=100?1:3):d)}
function pct(v){return v===null||v===undefined?'n/a':(100*v).toFixed(2)+'%'}
const C={b:'#6FA0DA',bl:'#3E6FA8',a:'#7BC17E',al:'#4C9B50'};
document.title=D.title;
document.getElementById('ttl').textContent='Interactive EDA Dashboard \u2014 '+D.bias_title+' (Before vs After Preprocessing)';
document.getElementById('sub').textContent=D.subtitle;
document.getElementById('lgB').textContent=D.legend_before;document.getElementById('lgA').textContent=D.legend_after;
document.getElementById('legendNote').textContent=D.units_note;
// ---------- info + summary
document.getElementById('info').innerHTML=D.info.map(r=>`<div class="info-row"><span class="k">${r[0]}</span><b>${r[1]}</b></div>`).join('');
const badge=D.detected===null?'<span class="badge na">not applicable</span>':(D.detected?'<span class="badge yes">DETECTED</span>':'<span class="badge no">not detected</span>');
document.getElementById('summary').innerHTML=[['Bias status',badge],[D.stat_label+' (before)',D.stat_before],[D.stat_label+' (after)',D.stat_after],['Applied to the data',D.applied||'none (before = after)']]
 .map(s=>`<div class="stat"><div class="k">${s[0]}</div><div class="v">${s[1]}</div></div>`).join('');
document.getElementById('decision').innerHTML=D.decision;
document.getElementById('distNote').textContent=D.dist_note;document.getElementById('boxNote').textContent=D.box_note;
document.getElementById('protocol').innerHTML='<b>Benchmark protocol.</b> '+D.protocol;
document.getElementById('methodNotes').innerHTML=D.method_notes.map(n=>`<div>&bull; <b>${n[0]}</b>: ${n[1]}</div>`).join('');
document.getElementById('foot').textContent=D.footer;
// ---------- variable controls
const P=D.core.per, labels=D.core.labels;let focus=D.core.grid[0]||D.core.variables[0];
const dd=document.getElementById('varDropdown'), vl=document.getElementById('varList');
D.core.variables.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=labels[v]||v;dd.appendChild(o);
 const d=document.createElement('div');d.className='opt';d.dataset.v=v;d.textContent=labels[v]||v;d.onclick=()=>setFocus(v);vl.appendChild(d)});
dd.onchange=()=>setFocus(dd.value);
function setFocus(v){focus=v;dd.value=v;vl.querySelectorAll('.opt').forEach(o=>o.classList.toggle('active',o.dataset.v===v));drawQQ();
 if(!D.core.grid.includes(v)){D.core.grid=[v].concat(D.core.grid.slice(0,D.core.grid.length-1))}drawGrids()}
// ---------- histogram / density
function drawDist(svg,p){svg.innerHTML='';const W=270,H=170,L=34,B=22,T=8,R=6;svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
 const series=[];const sep=p.separate_axes;
 for(const k of ['before','after']){const h=p['hist_'+k];if(!h)continue;const n=p['n_'+k]||1;series.push({k:k,h:h.map(c=>c/n),c:h,n:n,e:sep?p['edges_'+k]:p.edges})}
 if(!series.length){txt(svg,W/2,H/2,'no values',{'text-anchor':'middle'});return}
 const maxY=Math.max(1e-9,...series.flatMap(s=>s.h));const pw=W-L-R,ph=H-T-B;
 for(let i=0;i<=4;i++){const y=H-B-i/4*ph;svg.appendChild(el('line',{x1:L,y1:y,x2:W-R,y2:y,stroke:'#EDEFF4'}));txt(svg,L-4,y+3,(100*maxY*i/4).toFixed(maxY<0.05?1:0)+'%',{'text-anchor':'end','font-size':8})}
 svg.appendChild(el('line',{x1:L,y1:T,x2:L,y2:H-B,stroke:'#C8CEDE'}));svg.appendChild(el('line',{x1:L,y1:H-B,x2:W-R,y2:H-B,stroke:'#C8CEDE'}));
 const mode=document.getElementById('plotType').value;
 series.forEach((s,si)=>{const nb=s.h.length,bw=pw/nb,col=s.k==='before'?C.b:C.a,line=s.k==='before'?C.bl:C.al;
  if(mode==='hist'){for(let i=0;i<nb;i++){const hgt=s.h[i]/maxY*ph;const inner=si===1;const w=inner?bw*0.55:Math.max(bw-1,1);const x=L+i*bw+(inner?(bw-w)/2:0.5);
    hov(svg.appendChild(el('rect',{x:x,y:H-B-hgt,width:w,height:Math.max(hgt,0),fill:col,'fill-opacity':inner?0.9:0.75,stroke:line,'stroke-width':0.8})),
     `${s.k==='before'?'Before':'After'}: ${fmt(s.e[i])} to ${fmt(s.e[i+1])} \u2192 ${s.c[i].toLocaleString()} rows (${(100*s.h[i]).toFixed(2)}%)`)}}
  else{let d='';const sm=s.h.map((v,i)=>((s.h[i-1]??v)+2*v+(s.h[i+1]??v))/4);sm.forEach((v,i)=>{const x=L+(i+0.5)*bw,y=H-B-v/maxY*ph;d+=(i?'L':'M')+x+','+y});
   svg.appendChild(el('path',{d:d,fill:'none',stroke:line,'stroke-width':2}))}});
 const e0=series[0].e;txt(svg,L,H-8,fmt(e0[0]),{'font-size':8});txt(svg,W-R,H-8,fmt(e0[e0.length-1]),{'text-anchor':'end','font-size':8});
 if(sep)txt(svg,(L+W)/2,H-8,'each series on its own range',{'text-anchor':'middle','font-size':7.5})}
function drawBox(svg,p){svg.innerHTML='';const W=270,H=150,L=44,B=18,T=10,R=8;svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
 const bs=[['Before',p.box_before,C.b,C.bl],['After',p.box_after,C.a,C.al]].filter(x=>x[1]);if(!bs.length)return;
 const sep=p.separate_axes;
 bs.forEach((b,i)=>{const s=b[1];const vals=[s.whisk_lo,s.whisk_hi,...s.outliers];const lo=sep?Math.min(...vals):Math.min(...bs.flatMap(x=>[x[1].whisk_lo,...x[1].outliers]));
  const hi=sep?Math.max(...vals):Math.max(...bs.flatMap(x=>[x[1].whisk_hi,...x[1].outliers]));const rg=(hi-lo)||1;const y=v=>H-B-(v-lo)/rg*(H-B-T);
  const cx=L+(i+0.5)*(W-L-R)/2;const bw=46;
  if(i===0||sep){[lo,(lo+hi)/2,hi].forEach(v=>txt(svg,sep?cx-bw/2-4:L-4,y(v)+3,fmt(v),{'text-anchor':'end','font-size':8}))}
  svg.appendChild(el('line',{x1:cx,y1:y(s.whisk_lo),x2:cx,y2:y(s.q1),stroke:b[3]}));svg.appendChild(el('line',{x1:cx,y1:y(s.q3),x2:cx,y2:y(s.whisk_hi),stroke:b[3]}));
  [s.whisk_lo,s.whisk_hi].forEach(v=>svg.appendChild(el('line',{x1:cx-10,y1:y(v),x2:cx+10,y2:y(v),stroke:b[3]})));
  hov(svg.appendChild(el('rect',{x:cx-bw/2,y:y(s.q3),width:bw,height:Math.max(y(s.q1)-y(s.q3),1),fill:b[2],'fill-opacity':0.8,stroke:b[3]})),
   `${b[0]}: n=${s.n.toLocaleString()} median ${fmt(s.median)} IQR ${fmt(s.q1)}\u2013${fmt(s.q3)} whiskers ${fmt(s.whisk_lo)}\u2013${fmt(s.whisk_hi)}; ${s.n_outliers.toLocaleString()} points beyond 1.5 IQR`);
  svg.appendChild(el('line',{x1:cx-bw/2,y1:y(s.median),x2:cx+bw/2,y2:y(s.median),stroke:'#1B2333','stroke-width':1.6}));
  s.outliers.forEach(v=>svg.appendChild(el('circle',{cx:cx,cy:y(v),r:2,fill:'none',stroke:b[3]})));
  txt(svg,cx,H-5,b[0]+' ('+s.n_outliers.toLocaleString()+' out)',{'text-anchor':'middle','font-size':8.5,fill:'#1B2333'})})}
function drawGrids(){const dg=document.getElementById('distGrid'),bg=document.getElementById('boxGrid');dg.innerHTML='';bg.innerHTML='';
 D.core.grid.forEach(v=>{const p=P[v];for(const [g,f] of [[dg,drawDist],[bg,drawBox]]){const box=document.createElement('div');box.className='chart-box'+(v===focus?' focus':'');
  box.innerHTML=`<div class="ct" title="${labels[v]||v}">${labels[v]||v}</div><div class="cs">missing: ${pct(p.missing_before)} \u2192 ${pct(p.missing_after)}</div>`;
  const s=el('svg',{width:'100%'});box.appendChild(s);box.onclick=()=>setFocus(v);g.appendChild(box);f(s,p)}})}
function drawQQ(){const c=document.getElementById('qqChart');c.innerHTML='';document.getElementById('qqVarName').textContent=labels[focus]||focus;
 for(const [k,col,line,name] of [['qq_before',C.b,C.bl,'Before'],['qq_after',C.a,C.al,'After']]){const q=P[focus][k];const wrap=document.createElement('div');
  wrap.innerHTML=`<div class="mini-title">${name}</div>`;const s=el('svg',{width:260,height:230,viewBox:'0 0 260 230'});wrap.appendChild(s);c.appendChild(wrap);
  if(!q){txt(s,130,115,'no values',{'text-anchor':'middle'});continue}
  const L=40,B=26,T=8,R=8,W=260,H=230;const tx=q.theoretical,sy=q.sample;const xl=Math.min(...tx),xh=Math.max(...tx),yl=Math.min(...sy),yh=Math.max(...sy)||yl+1;
  const X=v=>L+(v-xl)/(xh-xl)*(W-L-R),Y=v=>H-B-(v-yl)/((yh-yl)||1)*(H-B-T);
  s.appendChild(el('rect',{x:L,y:T,width:W-L-R,height:H-B-T,fill:'#FBFCFE',stroke:'#E1E5EE'}));
  const i1=Math.floor(tx.length*0.25),i3=Math.floor(tx.length*0.75);const slope=(sy[i3]-sy[i1])/((tx[i3]-tx[i1])||1),inter=sy[i1]-slope*tx[i1];
  s.appendChild(el('line',{x1:X(xl),y1:Y(inter+slope*xl),x2:X(xh),y2:Y(inter+slope*xh),stroke:'#D0433F','stroke-dasharray':'4 3'}));
  tx.forEach((t,i)=>hov(s.appendChild(el('circle',{cx:X(t),cy:Y(sy[i]),r:3,fill:col,stroke:line})),`${name}: normal quantile ${fmt(t,2)} \u2192 sample ${fmt(sy[i])}`));
  txt(s,(L+W)/2,H-6,'Theoretical normal quantile',{'text-anchor':'middle'});txt(s,L-4,Y(yl)+3,fmt(yl),{'text-anchor':'end','font-size':8});txt(s,L-4,Y(yh)+3,fmt(yh),{'text-anchor':'end','font-size':8});
  const yt=txt(s,10,(T+H-B)/2,'Sample quantile',{'text-anchor':'middle'});yt.setAttribute('transform',`rotate(-90 10 ${(T+H-B)/2})`)}}
function corrColor(v){const a=Math.min(1,Math.abs(v));return v>=0?`rgba(37,99,200,${0.08+0.85*a})`:`rgba(208,67,63,${0.08+0.85*a})`}
function drawCorr(){const c=document.getElementById('corrChart');c.innerHTML='';const vs=D.core.corr_vars;if(!vs.length){c.textContent='No numeric variables.';return}
 for(const [m,name] of [[D.core.corr_before,'Before'],[D.core.corr_after,'After']]){const n=vs.length,cs=Math.max(18,Math.min(30,260/n)),L=96,T=6;
  const s=el('svg',{width:L+n*cs+4,height:T+n*cs+92});const wrap=document.createElement('div');wrap.innerHTML=`<div class="mini-title">${name}</div>`;wrap.appendChild(s);c.appendChild(wrap);
  vs.forEach((v,i)=>{txt(s,L-4,T+i*cs+cs/2+3,(labels[v]||v).slice(0,16),{'text-anchor':'end','font-size':8.5});
   const t=txt(s,L+i*cs+cs/2,T+n*cs+6,(labels[v]||v).slice(0,16),{'font-size':8.5,'text-anchor':'end'});t.setAttribute('transform',`rotate(-55 ${L+i*cs+cs/2} ${T+n*cs+6})`);
   vs.forEach((w,j)=>{const val=m[i][j];hov(s.appendChild(el('rect',{x:L+j*cs,y:T+i*cs,width:cs-1,height:cs-1,fill:corrColor(val)})),`${name}: r(${labels[v]||v}, ${labels[w]||w}) = ${val.toFixed(3)}`);
    if(cs>=24)txt(s,L+j*cs+cs/2,T+i*cs+cs/2+3,val.toFixed(2),{'text-anchor':'middle','font-size':7.5,fill:Math.abs(val)>0.55?'#fff':'#1B2333'})})})}}
// ---------- bias-specific panels
function heat(container,title,rows,mat,max,label,cols,axis){const n=mat[0]?mat[0].length:0,cs=cols?Math.max(34,Math.min(60,420/Math.max(n,1))):Math.max(8,Math.min(16,420/Math.max(n,1))),rh=15,L=150;
 const s=el('svg',{width:L+n*cs+10,height:rows.length*rh+(cols?40:30)});const wrap=document.createElement('div');wrap.style.flex='1';wrap.innerHTML=`<div class="mini-title">${title}</div>`;wrap.appendChild(s);container.appendChild(wrap);
 rows.forEach((r,i)=>{txt(s,L-6,i*rh+11,(labels[r]||r).slice(0,22),{'text-anchor':'end','font-size':9,fill:'#1B2333'});
  mat[i].forEach((v,j)=>hov(s.appendChild(el('rect',{x:L+j*cs,y:i*rh,width:cs-1.5,height:rh-2,rx:2,fill:v>0?`rgba(37,99,200,${0.15+0.85*Math.min(1,v/(max||1))})`:'#FFFFFF',stroke:'#D7DEEC','stroke-width':0.6})),
   `${title}: ${labels[r]||r}, ${cols?cols[j]:'record group '+(j+1)}: ${label} ${cols?v.toFixed(3):(100*v).toFixed(2)+'%'}`))});
 if(cols){cols.forEach((c2,j)=>txt(s,L+j*cs+cs/2,rows.length*rh+12,String(c2).slice(0,9),{'text-anchor':'middle','font-size':8}))}
 txt(s,L,rows.length*rh+(cols?26:14),axis||'record groups (row order) \u2192',{'font-size':8.5})}
function bars(container,title,cats,series,opt){opt=opt||{};const W=opt.w||420,H=opt.h||220,L=48,B=56,T=14,R=8;const s=el('svg',{width:'100%',viewBox:`0 0 ${W} ${H}`});
 const wrap=document.createElement('div');wrap.style.flex='1';wrap.style.minWidth='300px';wrap.innerHTML=`<div class="mini-title">${title}</div>`;wrap.appendChild(s);container.appendChild(wrap);
 const all=series.flatMap(x=>x.v.filter(v=>v!==null));const mx=Math.max(1e-9,...all.map(Math.abs))*(opt.log?1:1.1);const pw=W-L-R,ph=H-T-B;
 const yv=v=>opt.log?Math.log10(Math.max(v,1e-6))-Math.log10(Math.max(Math.min(...all.filter(x=>x>0)),1e-6)):v;const ym=opt.log?yv(mx)||1:mx;
 for(let i=0;i<=4;i++){const y=H-B-i/4*ph;s.appendChild(el('line',{x1:L,y1:y,x2:W-R,y2:y,stroke:'#EDEFF4'}));
  txt(s,L-4,y+3,opt.log?fmt(Math.pow(10,Math.log10(Math.max(Math.min(...all.filter(x=>x>0)),1e-6))+ym*i/4)):(opt.pct?(100*mx*i/4).toFixed(1)+'%':fmt(mx*i/4)),{'text-anchor':'end','font-size':8})}
 const gw=pw/cats.length,bw=Math.min(28,gw/(series.length+0.6));
 cats.forEach((c,i)=>{series.forEach((se,k)=>{const v=se.v[i];if(v===null||v===undefined)return;const h=Math.max(0,yv(v)/ym*ph);const x=L+i*gw+(gw-bw*series.length)/2+k*bw;
   hov(s.appendChild(el('rect',{x:x,y:H-B-h,width:bw-2,height:h,fill:se.c,stroke:se.l,'stroke-width':0.8})),`${se.n} \u2014 ${c}: ${opt.pct?(100*v).toFixed(2)+'%':fmt(v)}`)});
  const t=txt(s,L+i*gw+gw/2,H-B+12,String(c).slice(0,18),{'text-anchor':'end','font-size':8.5,fill:'#1B2333'});t.setAttribute('transform',`rotate(-35 ${L+i*gw+gw/2} ${H-B+12})`)})}
function drawSpec(){const S=D.spec,c=document.getElementById('spec');document.getElementById('specTitle').textContent=S.title;document.getElementById('specNote').innerHTML=S.note||'';
 c.style.display='flex';c.style.gap='18px';c.style.flexWrap='wrap';
 const BA=(nb,na)=>[{n:'Before',v:nb,c:C.b,l:C.bl},{n:'After',v:na,c:C.a,l:C.al}];
 if(S.kind==='heat'){const mx=Math.max(1e-9,...S.before.flat(),...S.after.flat());heat(c,'Before',S.rows,S.before,mx,S.label,S.cols,S.axis);heat(c,'After',S.rows,S.after,mx,S.label,S.cols,S.axis);
  document.getElementById('specChips').innerHTML=`<span class="lg"><span class="swatch" style="background:#2563C8"></span>${S.label}</span><span class="lg"><span class="swatch" style="background:#fff;border:1px solid #D7DEEC"></span>none</span>`}
 (S.bars||[]).forEach(b=>bars(c,b.title,b.cats,b.series?b.series.map((s,i)=>({n:s.n,v:s.v,c:[C.b,C.a,'#C9A227','#9C6ADE'][i%4],l:[C.bl,C.al,'#9A7A12','#6B3FB0'][i%4]})):BA(b.before,b.after),b.opt));
 if(S.table){const t=document.createElement('div');t.className='tbl-wrap';t.style.flexBasis='100%';t.innerHTML=S.table;c.appendChild(t)}}
// ---------- method comparison table
function drawTable(){const T=D.table;if(!T){document.getElementById('cmp').textContent='No benchmark for this bias.';return}
 const dirs=D.directions||{};let h='<table class="cmp"><thead><tr><th>Method</th>'+T.columns.map(c=>`<th>${c}${dirs[c]?(dirs[c]==='max'?' \u2191':' \u2193'):''}</th>`).join('')+'</tr></thead><tbody>';
 T.rows.forEach(r=>{const cls=(r.method===D.best?'sel ':'')+(r.method===D.applied_method?'app':'');h+=`<tr class="${cls}"><td>${r.method}${r.method===D.best?' \u2605':''}</td>`+
  r.values.map(v=>typeof v==='number'?`<td>${fmt(v,4)}</td>`:`<td class="txt">${v===null?'n/a':v}</td>`).join('')+'</tr>'});
 document.getElementById('cmp').innerHTML=h+'</tbody></table>'}
document.getElementById('plotType').onchange=drawGrids;
setFocus(focus);drawCorr();drawSpec();drawTable();
</script></body></html>"""


def render(payload, path):
    text = TEMPLATE.replace("__TITLE__", html.escape(payload["title"])).replace("__CSS__", CSS)
    data = json.dumps(clean_json(payload), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.replace("__PAYLOAD__", data), encoding="utf-8")


INDEX = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title><style>__CSS__ .tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:14px}
.tile{display:block;text-decoration:none;color:var(--text)}.tile:hover .card{outline:2px solid var(--blue-hi)}.tile h3{margin:0 0 6px 0;font-size:15px;color:var(--blue-hi)}
.wrap{padding:16px;max-width:1400px;margin:0 auto}</style></head><body>
<header class="topbar"><h1>__H1__</h1><div class="sub">__SUB__</div></header><div class="wrap">
<div class="card"><p class="panel-title">Pipeline order and outcome</p>__SUMMARY__</div>
<div class="tiles mt">__TILES__</div><p class="note mt">__NOTE__</p></div></body></html>"""


def render_index(path, title, h1, sub, summary_html, tiles, note):
    tiles_html = "".join(
        f'<a class="tile" href="{html.escape(t["file"])}"><div class="card"><h3>{html.escape(t["title"])}</h3>'
        f'<div class="note">{t["status"]}</div><div class="note" style="margin-top:6px">{html.escape(t["line"])}</div></div></a>'
        for t in tiles)
    text = (INDEX.replace("__TITLE__", html.escape(title)).replace("__CSS__", CSS).replace("__H1__", html.escape(h1))
            .replace("__SUB__", html.escape(sub)).replace("__SUMMARY__", summary_html).replace("__TILES__", tiles_html)
            .replace("__NOTE__", note))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
