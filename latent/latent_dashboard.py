"""Four individual latent-metric dashboards (CKA, Macro-F1, Silhouette, NMI/ARI) + index.

Same visual style as the preprocessing bias dashboards. Before = per-dataset PCA (no shared
space); After = the method chosen in the sidebar (default: the selected method).
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latent_alignment import ROOT, clean_json  # noqa: E402

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


PAGES = [("cka", "01_cka_alignment_dashboard.html", "Cross-Dataset Alignment (CKA)", "CKA (alignment)",
          "Higher = latent features of the five datasets are aligned in a common space."),
         ("f1", "02_macro_f1_preservation_dashboard.html", "Clinical Information Preservation (Macro-F1)", "Macro-F1 (preservation)",
          "Higher = the latent features keep the information needed to classify each dataset's outcome."),
         ("sil", "03_silhouette_structure_dashboard.html", "Latent Structure (Silhouette Score)", "Silhouette (structure)",
          "Higher = clinical classes are more clearly separated in the latent space."),
         ("nmi", "04_nmi_ari_domain_invariance_dashboard.html", "Domain Invariance (NMI and ARI)", "NMI (domain)",
          "Lower = less dataset-specific information remains in the common latent space.")]
DATASETS = ["heart", "diabetes", "hypertension", "respiratory", "hospital"]   # hospital = cross-hospital ICU cohort (eICU + MIMIC)

TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title><style>__CSS__</style></head><body><div id="tooltip"></div>
<header class="topbar"><h1 id="ttl"></h1><div class="sub" id="sub"></div></header>
<div class="layout"><div class="side">
 <div class="card"><p class="side-label">1. Select Comparison Method (After)</p><select id="method" class="control"></select></div>
 <div class="card"><p class="side-label">2. Select Dataset (per-dataset charts)</p><select id="ds" class="control"></select></div>
 <div class="card"><p class="info-title">&#9432; Dataset Info</p><div id="info"></div></div>
 <div class="card"><p class="legend-title">&#128273; Legend</p>
  <div class="legend-row"><span class="swatch" style="background:var(--before);border:1.5px solid var(--before-line)"></span>Before: per-dataset PCA (no shared space)</div>
  <div class="legend-row"><span class="swatch" style="background:var(--after);border:1.5px solid var(--after-line)"></span><span id="lgA"></span></div>
  <div class="legend-row" id="lgU" style="display:none"><span class="swatch" style="background:#B8BFCC;border:1.5px solid #7A8394"></span>Uncompressed preprocessed features</div>
  <p class="note"><a href="index.html">&#8592; All latent dashboards</a></p></div>
</div><div class="main">
 <div class="card"><p class="panel-title">Metric summary</p><div class="summary-grid" id="summary"></div><p class="note" id="meaning"></p></div>
 <div class="card mt"><p class="panel-title" id="mainTitle"></p><div id="main" style="display:flex;gap:18px;flex-wrap:wrap"></div><p class="note" id="mainNote"></p></div>
 <div class="card mt"><p class="panel-title" id="secondTitle"></p><div id="second" style="display:flex;gap:18px;flex-wrap:wrap"></div></div>
 <div class="card mt"><p class="panel-title">Method Comparison (all latent methods &times; all metrics)
  <span class="legend-inline"><span class="lg"><span class="swatch" style="background:#EAF3FF;border:1px solid var(--blue-hi)"></span>Selected (best mean rank)</span></span></p>
  <div class="tbl-wrap" id="cmp"></div><div class="note" id="notes"></div></div>
</div></div><footer class="note" id="foot"></footer>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const D=JSON.parse(document.getElementById('payload').textContent);const NS="http://www.w3.org/2000/svg";
function el(t,a){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e}
function txt(s,x,y,v,a){const t=el('text',Object.assign({x:x,y:y,'font-size':9,fill:'#6B7686'},a||{}));t.textContent=v;s.appendChild(t);return t}
const tip=document.getElementById('tooltip');function hov(n,s){n.addEventListener('mousemove',e=>{tip.textContent=s;tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY-10)+'px'});n.addEventListener('mouseleave',()=>tip.style.display='none');return n}
function f3(v){return v===null||v===undefined||Number.isNaN(v)?'n/a':v.toFixed(3)}
const C={b:'#6FA0DA',bl:'#3E6FA8',a:'#7BC17E',al:'#4C9B50',u:'#B8BFCC',ul:'#7A8394'};const DSC=['#2563C8','#D0433F','#C9A227','#2E9E6B','#7B4FC9'];
const BEFORE='PCA (no alignment)';document.title=D.title;document.getElementById('ttl').textContent='Interactive Latent Dashboard — '+D.page_title;
document.getElementById('sub').textContent=D.subtitle;document.getElementById('meaning').innerHTML=D.meaning;document.getElementById('foot').textContent=D.footer;
document.getElementById('info').innerHTML=D.info.map(r=>`<div class="info-row"><span class="k">${r[0]}</span><b>${r[1]}</b></div>`).join('');
const ms=document.getElementById('method');D.methods.filter(m=>m!==BEFORE).forEach(m=>{const o=document.createElement('option');o.value=m;o.textContent=m+(m===D.selected?' (selected)':'');ms.appendChild(o)});ms.value=D.selected;
const dsel=document.getElementById('ds');D.datasets.forEach(d=>{const o=document.createElement('option');o.value=d;o.textContent=d;dsel.appendChild(o)});
function bars(c,title,cats,series,opt){opt=opt||{};const W=opt.w||460,H=230,L=46,B=50,T=12,R=8;const s=el('svg',{width:'100%',viewBox:`0 0 ${W} ${H}`});const w=document.createElement('div');w.style.flex='1';w.style.minWidth='300px';w.innerHTML=`<div class="mini-title">${title}</div>`;w.appendChild(s);c.appendChild(w);
 const all=series.flatMap(x=>x.v).filter(v=>v!==null&&!Number.isNaN(v));const lo=Math.min(0,...all),hi=Math.max(1e-9,...all)*1.1;const pw=W-L-R,ph=H-T-B;const y=v=>H-B-(v-lo)/(hi-lo)*ph;
 for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4;s.appendChild(el('line',{x1:L,y1:y(v),x2:W-R,y2:y(v),stroke:'#EDEFF4'}));txt(s,L-4,y(v)+3,v.toFixed(2),{'text-anchor':'end','font-size':8})}
 s.appendChild(el('line',{x1:L,y1:y(0),x2:W-R,y2:y(0),stroke:'#C8CEDE'}));const gw=pw/cats.length,bw=Math.min(30,gw/(series.length+0.6));
 cats.forEach((ct,i)=>{series.forEach((se,k)=>{const v=se.v[i];if(v===null||Number.isNaN(v))return;const x=L+i*gw+(gw-bw*series.length)/2+k*bw;const y0=y(Math.max(0,v)),h=Math.abs(y(v)-y(0));
  hov(s.appendChild(el('rect',{x:x,y:v>=0?y(v):y(0),width:bw-2,height:Math.max(h,0.5),fill:se.c,stroke:se.l})),`${se.n} — ${ct}: ${f3(v)}`)});
  const t=txt(s,L+i*gw+gw/2,H-B+12,String(ct).slice(0,22),{'text-anchor':'end','font-size':8.5,fill:'#1B2333'});t.setAttribute('transform',`rotate(-30 ${L+i*gw+gw/2} ${H-B+12})`)})}
function matrix(c,title,m,rows,cols,fmt,maxv){const cs=56,L=90,T=8;const s=el('svg',{width:L+cols.length*cs+6,height:T+rows.length*cs+40});const w=document.createElement('div');w.innerHTML=`<div class="mini-title">${title}</div>`;w.appendChild(s);c.appendChild(w);
 rows.forEach((r,i)=>{txt(s,L-6,T+i*cs+cs/2+3,r,{'text-anchor':'end','font-size':9.5,fill:'#1B2333'});cols.forEach((cl,j)=>{const v=m[i][j];const a=v===null?0:Math.min(1,Math.abs(v)/(maxv||1));
  hov(s.appendChild(el('rect',{x:L+j*cs,y:T+i*cs,width:cs-2,height:cs-2,rx:3,fill:v===null?'#F2F4F8':`rgba(37,99,200,${0.1+0.85*a})`})),`${title}: ${r} / ${cl}: ${fmt(v)}`);
  txt(s,L+j*cs+cs/2,T+i*cs+cs/2+3,fmt(v),{'text-anchor':'middle','font-size':9.5,fill:a>0.55?'#fff':'#1B2333'})})});
 cols.forEach((cl,j)=>txt(s,L+j*cs+cs/2,T+rows.length*cs+14,cl,{'text-anchor':'middle','font-size':9,fill:'#1B2333'}))}
function scatter(c,title,sc,colorBy,labelsOf){const W=330,H=280,L=10,T=10;const s=el('svg',{width:W,height:H});const w=document.createElement('div');w.innerHTML=`<div class="mini-title">${title}</div>`;w.appendChild(s);c.appendChild(w);
 const xs=sc.x,ys=sc.y;const xl=Math.min(...xs),xh=Math.max(...xs),yl=Math.min(...ys),yh=Math.max(...ys);s.appendChild(el('rect',{x:L,y:T,width:W-2*L,height:H-2*T,fill:'#FBFCFE',stroke:'#E1E5EE'}));
 xs.forEach((x,i)=>{const g=colorBy[i];hov(s.appendChild(el('circle',{cx:L+(x-xl)/((xh-xl)||1)*(W-2*L),cy:H-T-(ys[i]-yl)/((yh-yl)||1)*(H-2*T),r:2.4,fill:DSC[g%DSC.length],'fill-opacity':0.55})),labelsOf(i))})}
function legendDs(c,names){const d=document.createElement('div');d.className='legend-inline';d.style.flexBasis='100%';d.innerHTML=names.map((n,i)=>`<span class="lg"><span class="swatch" style="background:${DSC[i]}"></span>${n}</span>`).join('');c.appendChild(d)}
function draw(){const m=ms.value,ds=dsel.value,R=D.results,B=R[BEFORE],A=R[m];document.getElementById('lgA').textContent='After: '+m;
 const key=D.metric;const stats=D.page==='nmi'?[['NMI before',f3(B['NMI (domain)'])],['NMI after',f3(A['NMI (domain)'])],['ARI before',f3(B['ARI (domain)'])],['ARI after',f3(A['ARI (domain)'])]]
  :[[D.metric+' before',f3(B[key])],[D.metric+' after',f3(A[key])],['Selected method',D.selected],['Change',f3(A[key]-B[key])]];
 document.getElementById('summary').innerHTML=stats.map(s=>`<div class="stat"><div class="k">${s[0]}</div><div class="v">${s[1]}</div></div>`).join('');
 const main=document.getElementById('main'),second=document.getElementById('second');main.innerHTML='';second.innerHTML='';
 const meth=D.methods;const names=D.datasets;
 if(D.page==='cka'){document.getElementById('mainTitle').textContent='Pairwise centroid CKA between datasets (validation)';
  matrix(main,'Before',B.cka_matrix,names,names,v=>v===null?'n/a':v.toFixed(2),1);matrix(main,'After',A.cka_matrix,names,names,v=>v===null?'n/a':v.toFixed(2),1);
  document.getElementById('mainNote').textContent='Each cell compares the mean latent vector of every shared age × sex × BMI-category stratum (≥ 20 rows) in two datasets with linear CKA (diagonal = 1); pairs with the cross-hospital ICU cohort use age × sex strata because it has no BMI. Shared strata per pair: '+names.map((a,i)=>names.slice(i+1).map((b,j)=>`${a}-${b}: ${A.cka_support[i][i+1+j]}`).join(', ')).filter(x=>x).join(', ')+'. Different people are compared through demographic strata, not as paired patients.';
  document.getElementById('secondTitle').textContent='Mean cross-dataset CKA by method';bars(second,'CKA (higher is better)',meth,[{n:'CKA',v:meth.map(x=>R[x]['CKA (alignment)']),c:C.a,l:C.al}])}
 if(D.page==='f1'){document.getElementById('lgU').style.display='flex';document.getElementById('mainTitle').textContent='Validation Macro-F1 of a logistic probe, per dataset';
  bars(main,'Macro-F1 (higher is better)',names,[{n:'Uncompressed',v:names.map(n=>D.uncompressed[n]),c:C.u,l:C.ul},{n:'Before (PCA)',v:names.map(n=>B.per_dataset[n].macro_f1),c:C.b,l:C.bl},{n:'After ('+m+')',v:names.map(n=>A.per_dataset[n].macro_f1),c:C.a,l:C.al}],{w:560});
  bars(main,'Probe AUROC',names,[{n:'Before (PCA)',v:names.map(n=>B.per_dataset[n].auroc),c:C.b,l:C.bl},{n:'After',v:names.map(n=>A.per_dataset[n].auroc),c:C.a,l:C.al}]);
  document.getElementById('mainNote').textContent='Class-balanced logistic regression fitted on training latents (real rows), threshold 0.5, scored on validation. Uncompressed = the same probe on the preprocessed features. hospital = mean over its four recorded conditions (heart, diabetes, hypertension, respiratory).';
  document.getElementById('secondTitle').textContent='Mean Macro-F1 by method';bars(second,'Macro-F1',meth,[{n:'Macro-F1',v:meth.map(x=>R[x]['Macro-F1 (preservation)']),c:C.a,l:C.al}])}
 if(D.page==='sil'){document.getElementById('mainTitle').textContent='Silhouette of clinical classes per dataset (validation)';
  bars(main,'Silhouette (higher is better)',names,[{n:'Before (PCA)',v:names.map(n=>B.per_dataset[n].silhouette),c:C.b,l:C.bl},{n:'After',v:names.map(n=>A.per_dataset[n].silhouette),c:C.a,l:C.al}],{w:520});
  const i=names.indexOf(ds);for(const [lab,res] of [['Before',B],['After',A]]){const sc=res.scatter;const keep=sc.dataset.map((g,k)=>g===i?k:-1).filter(k=>k>=0);
   scatter(main,`${lab}: ${ds} (2-D PCA of the pooled latent), colour = class`,{x:keep.map(k=>sc.x[k]),y:keep.map(k=>sc.y[k])},keep.map(k=>sc.label[k]?1:0),k=>'class '+sc.label[keep[k]])}
  legendDs(main,['class 0 (negative)','class 1 (positive)']);
  document.getElementById('mainNote').textContent='Silhouette on up to 2,000 validation rows per dataset using the recorded outcome as the cluster label. Scatter: up to 300 validation rows; use the dataset selector.';
  document.getElementById('secondTitle').textContent='Mean silhouette by method';bars(second,'Silhouette',meth,[{n:'Silhouette',v:meth.map(x=>R[x]['Silhouette (structure)']),c:C.a,l:C.al}])}
 if(D.page==='nmi'){document.getElementById('mainTitle').textContent='Pooled validation latents coloured by dataset (2-D PCA projection)';
  for(const [lab,res] of [['Before',B],['After',A]])scatter(main,lab,res.scatter,res.scatter.dataset,k=>names[res.scatter.dataset[k]]);legendDs(main,names);
  matrix(main,'Before: clusters × dataset',B.contingency,names,names.map((_,i)=>'k'+(i+1)),v=>String(v),Math.max(...B.contingency.flat()));
  matrix(main,'After: clusters × dataset',A.contingency,names,names.map((_,i)=>'k'+(i+1)),v=>String(v),Math.max(...A.contingency.flat()));
  document.getElementById('mainNote').textContent='k-means (k = number of datasets = '+names.length+') fitted on 2,000 training latents per dataset, applied to 2,000 validation latents per dataset; NMI/ARI compare clusters with dataset identity (0 = no dataset information).';
  document.getElementById('secondTitle').textContent='NMI and ARI by method (lower is better)';bars(second,'NMI / ARI',meth,[{n:'NMI',v:meth.map(x=>R[x]['NMI (domain)']),c:C.b,l:C.bl},{n:'ARI',v:meth.map(x=>R[x]['ARI (domain)']),c:C.a,l:C.al}])}
 let h='<table class="cmp"><thead><tr><th>Method</th>'+D.table.columns.map(c=>`<th>${c}${D.directions[c]?(D.directions[c]==='max'?' ↑':' ↓'):''}</th>`).join('')+'</tr></thead><tbody>';
 D.table.rows.forEach(r=>{h+=`<tr class="${r.method===D.selected?'sel':''}"><td>${r.method}${r.method===D.selected?' ★':''}</td>`+r.values.map(v=>typeof v==='number'?`<td>${v.toFixed(4)}</td>`:`<td class="txt">${v}</td>`).join('')+'</tr>'});
 document.getElementById('cmp').innerHTML=h+'</tbody></table>';document.getElementById('notes').innerHTML=D.notes.map(n=>`<div>&bull; <b>${n[0]}</b>: ${n[1]}</div>`).join('')}
ms.onchange=draw;dsel.onchange=draw;draw();
</script></body></html>"""


def build(results, table, selected, uncompressed, notes, directions):
    out = ROOT / "latent" / "plots"
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads((ROOT / "latent" / "latent_report.json").read_text())
    info = [[n, f"{r['train']:,} / {r['validation']:,} / {r['test']:,}" + (f" (+ MIMIC {r['external']:,})" if "external" in r else "")]
            for n, r in report["rows"].items()]
    info.insert(0, ["Rows (train / val / test)", ""])
    info.append(["Latent dimension", str(report["dimension"])])
    tbl = {"columns": [str(c) for c in table.columns],
           "rows": [{"method": m, "values": [v if isinstance(v, str) else round(float(v), 4) for v in row]} for m, row in table.iterrows()]}
    tiles = []
    for key, file, title, metric, meaning in PAGES:
        payload = {"title": f"Latent | {title}", "page": key, "page_title": title, "metric": metric,
                   "subtitle": "Heart (BRFSS) · Diabetes (NHANES) · Hypertension (NHIS) · Respiratory (eICU) · Cross-hospital ICU (eICU hospitals + MIMIC) · one common 16-D latent space",
                   "meaning": meaning + " Selection rule: best mean rank over CKA, Macro-F1, Silhouette, NMI and ARI.",
                   "methods": list(results), "selected": selected, "datasets": DATASETS, "info": info,
                   "results": {m: {k: v for k, v in r.items()} for m, r in results.items()}, "uncompressed": uncompressed,
                   "table": tbl, "directions": directions, "notes": [[k, v] for k, v in notes.items()],
                   "footer": "Generated by latent/latent_alignment.py · validation rows only · test rows (and the MIMIC external hospital) are encoded but never scored"}
        text = TEMPLATE.replace("__TITLE__", html.escape(payload["title"])).replace("__CSS__", CSS)
        data = json.dumps(clean_json(payload), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
        (out / file).write_text(text.replace("__PAYLOAD__", data), encoding="utf-8")
        r = results[selected]
        value = (f"NMI {r['NMI (domain)']:.3f} / ARI {r['ARI (domain)']:.3f}" if key == "nmi" else f"{metric}: {r[metric]:.3f}")
        before = results["PCA (no alignment)"]
        bval = (f"NMI {before['NMI (domain)']:.3f} / ARI {before['ARI (domain)']:.3f}" if key == "nmi" else f"{before[metric]:.3f}")
        tiles.append({"file": file, "title": title, "status": f"<b>{selected}</b>: {value}", "line": f"Before (PCA, no shared space): {bval}"})
    rows = "".join(f"<tr><td>{m}</td>" + "".join(f"<td>{float(table.loc[m, c]):.4f}</td>" for c in list(directions) + ["mean_rank"]) + "</tr>"
                   for m in table.index)
    summary = ("<div class='tbl-wrap'><table class='cmp'><thead><tr><th>Method</th>" + "".join(f"<th>{c}</th>" for c in list(directions) + ["mean rank"])
               + "</tr></thead><tbody>" + rows + f"</tbody></table></div><p class='note'>Selected: <b>{selected}</b>. Files: "
               "<code>latent/data/combined_latent_&lt;split&gt;.csv</code> (all five datasets) and "
               "<code>latent/data/&lt;dataset&gt;_latent_&lt;split&gt;.csv</code>.</p>")
    render_index(out / "index.html", "Latent dashboards", "Combined latent feature extraction — individual metric dashboards",
                 "Five datasets mapped into one 16-dimensional latent space", summary, tiles,
                 "Each tile opens one latent metric. Blue = before (per-dataset PCA), green = after (latent method).")
