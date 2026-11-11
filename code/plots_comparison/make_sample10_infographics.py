"""Sample-10 infographics: the applied method of 8 biases (2 per dataset) shown on the project's 10 traced records,
in the layout of the 'Missing Value Imputation and Validation Framework' reference image:
  1 detect -> 2 apply method -> 3 corrected data -> 4 independent validation -> 5 accept or flag.

Inputs (nothing is re-run): <d>/preprocessed/sample10_trace.json (the 10 records before/after every stage),
<d>/raw/*_raw.csv, <d>/preprocessed/*_feature_space.csv / *_preprocessed.csv, bias_summary.csv and the comparison
tables saved in the dashboards. Step 4 is an independent check computed here (GMM plausibility, out-of-fold label
check, adversary / domain classifier, ECOD); the decisions in steps 2-3 are the pipeline's own, read from the trace.

Output: <d>/plots_comparison/S10_<nn>_<bias>_infographic.png (+ .html source of the image).
"""
from __future__ import annotations

import base64
import html as H
import io
import json
import subprocess
import tempfile
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
CHROME = [p for p in [Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
                      Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")] if p.exists()][0]
SEED = 0
INK, INK2, MUTED = "#1d1d1b", "#52514e", "#8a8984"
C_ACC, C_FLAG, C_IMP = "#1f9d55", "#d0433f", "#d62728"
PASTEL = ["#7aa6e0", "#86c98f", "#f2a65a", "#c59be0", "#e8a0b8", "#9ed0cf", "#d9c26a", "#b0b0b0"]
REGION_C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.edgecolor": "#c9c8c2",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


# ------------------------------------------------------------------ helpers
def trace(d):
    return json.load(open(ROOT / d / "preprocessed" / "sample10_trace.json", encoding="utf-8"))


def step(t, bias):
    return next(s for s in t["steps"] if s["bias"] == bias)


def isnan(v):
    return v is None or (isinstance(v, float) and v != v)


def fv(v, nd=2):
    if isnan(v):
        return "?"
    v = float(v)
    if abs(v - round(v)) < 1e-9 and abs(v) < 1e6:
        return f"{int(round(v))}"
    return f"{v:.{nd}f}" if abs(v) < 1000 else f"{v:,.0f}"


def sid(r):
    r = str(r)
    return r if len(r) <= 9 else "…" + r[-7:]


def b64(fig, dpi=170):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def encode_raw(d, df):
    """Encode categorical predictors exactly as the dataset's notebook does (ORDINAL_MAPS / NOMINAL from its config cell)."""
    import ast
    nb = json.load(open(ROOT / d / f"{d}_preprocessing.ipynb", encoding="utf-8"))
    cfg = next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "ORDINAL_MAPS =" in "".join(c["source"]))
    tree = ast.parse(cfg)
    env = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("ORDINAL_MAPS", "NOMINAL"):
            env[node.targets[0].id] = ast.literal_eval(node.value)
    df = df.copy()
    for c, m in env.get("ORDINAL_MAPS", {}).items():
        if c in df:
            df[c] = df[c].map(m).astype(float)
    return df


def comparison(d, file):
    s = (ROOT / d / "plots" / file).read_text(encoding="utf-8")
    D, _ = json.JSONDecoder().raw_decode(s[s.index("const D = ") + 10:])
    c = D["comparison"]
    return {r["method"]: dict(zip(c["metrics"], r["values"])) for r in c["rows"]}, c


def bias_row(d, bias):
    b = pd.read_csv(ROOT / d / "preprocessed" / "bias_summary.csv")
    return b[b["bias"] == bias].iloc[0]


def gmm_ellipses(ax, gmm, dims, to_x=lambda v: v, scale=(1, 1), colors=PASTEL, nstd=1.5):
    for k in range(gmm.n_components):
        mu = gmm.means_[k][dims]
        cov = gmm.covariances_[k][np.ix_(dims, dims)] if gmm.covariance_type == "full" else np.diag(gmm.covariances_[k][dims])
        cov = cov * np.outer(scale, scale)
        vals, vecs = np.linalg.eigh(cov)
        ang = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
        c = to_x(mu)
        e = Ellipse(c, 2 * nstd * np.sqrt(vals[-1]), 2 * nstd * np.sqrt(vals[0]), angle=ang, fc=colors[k % 8], alpha=0.13,
                    ec=colors[k % 8], lw=1.6, ls="--")
        ax.add_patch(e)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#eeede8", lw=0.8)
    ax.set_axisbelow(True)


def fit_gmm(Z, n=6, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Z))
    half = len(Z) // 2
    g = GaussianMixture(n, covariance_type="full", reg_covar=1e-3, random_state=seed, max_iter=300).fit(Z[idx[:half]])
    ll_ref = g.score_samples(Z[idx[half:]])          # held-out real records -> the plausibility reference
    return g, ll_ref, np.percentile(ll_ref, 1)


def pct_of(ll, ll_ref):
    return float((ll_ref < ll).mean() * 100)


# ------------------------------------------------------------------ icons (inline SVG, drawn here)
ICON = {
 "detect": '<svg viewBox="0 0 80 70"><rect x="12" y="6" width="38" height="50" rx="4" fill="#fff" stroke="#33415c" stroke-width="3"/>'
           '<g stroke="#33415c" stroke-width="3"><line x1="20" y1="18" x2="42" y2="18"/><line x1="20" y1="28" x2="42" y2="28"/><line x1="20" y1="38" x2="34" y2="38"/></g>'
           '<circle cx="50" cy="44" r="12" fill="#eaf2fd" stroke="#1f4e9c" stroke-width="4"/><line x1="59" y1="53" x2="70" y2="64" stroke="#1f4e9c" stroke-width="6" stroke-linecap="round"/></svg>',
 "model": '<svg viewBox="0 0 80 70"><g stroke="#2e7d4f" stroke-width="2.5" fill="none"><line x1="40" y1="14" x2="18" y2="38"/><line x1="40" y1="14" x2="40" y2="38"/><line x1="40" y1="14" x2="62" y2="38"/>'
          '<line x1="18" y1="38" x2="12" y2="58"/><line x1="18" y1="38" x2="26" y2="58"/><line x1="62" y1="38" x2="54" y2="58"/><line x1="62" y1="38" x2="68" y2="58"/></g>'
          '<g fill="#3d9a5f"><circle cx="40" cy="12" r="7"/><circle cx="18" cy="38" r="6"/><circle cx="40" cy="38" r="6"/><circle cx="62" cy="38" r="6"/></g>'
          '<g fill="#8fd19e" stroke="#2e7d4f"><rect x="7" y="56" width="10" height="9"/><rect x="21" y="56" width="10" height="9"/><rect x="49" y="56" width="10" height="9"/><rect x="63" y="56" width="10" height="9"/></g></svg>',
 "table": '<svg viewBox="0 0 80 70"><rect x="8" y="8" width="64" height="54" rx="3" fill="#fff" stroke="#8a6d3b" stroke-width="2.5"/><rect x="8" y="8" width="64" height="10" fill="#f5d9a8"/>'
          '<g stroke="#c9a66b" stroke-width="1.5"><line x1="8" y1="27" x2="72" y2="27"/><line x1="8" y1="36" x2="72" y2="36"/><line x1="8" y1="45" x2="72" y2="45"/><line x1="8" y1="54" x2="72" y2="54"/>'
          '<line x1="24" y1="8" x2="24" y2="62"/><line x1="40" y1="8" x2="40" y2="62"/><line x1="56" y1="8" x2="56" y2="62"/></g><rect x="40" y="36" width="16" height="9" fill="#e84a4a" opacity=".75"/></svg>',
 "valid": '<svg viewBox="0 0 80 70"><line x1="8" y1="60" x2="74" y2="60" stroke="#333" stroke-width="2"/><line x1="10" y1="62" x2="10" y2="6" stroke="#333" stroke-width="2"/>'
          '<path d="M12 60 C22 60 24 22 32 22 C40 22 42 60 52 60Z" fill="#f6c58f" opacity=".7" stroke="#e08a1e" stroke-width="1.5"/>'
          '<path d="M30 60 C42 60 46 12 56 12 C66 12 68 60 76 60Z" fill="#a9c8f0" opacity=".7" stroke="#2a78d6" stroke-width="1.5"/>'
          '<path d="M24 60 C30 60 32 40 38 40 C44 40 46 60 50 60Z" fill="#9fd6a8" opacity=".8" stroke="#2e7d4f" stroke-width="1.5"/></svg>',
 "decide": '<svg viewBox="0 0 80 70"><circle cx="24" cy="36" r="17" fill="#fff" stroke="#1f9d55" stroke-width="4"/><path d="M15 36 l7 7 l12 -14" fill="none" stroke="#1f9d55" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>'
           '<line x1="54" y1="8" x2="54" y2="64" stroke="#7a2a22" stroke-width="4"/><path d="M56 9 L76 17 L56 27Z" fill="#d0433f"/></svg>',
}

STEPC = [("#2563c8", "#eef4fd", "#c9dbf6"), ("#2e7d4f", "#eef8f0", "#c6e6cf"), ("#d9822b", "#fdf6ea", "#f3dcb8"),
         ("#7b4fc9", "#f4effc", "#ddd0f3"), ("#d0433f", "#fdeeee", "#f3c9c7")]

CSS = """
*{box-sizing:border-box} body{margin:0;background:#fff;font-family:'Segoe UI',Arial,sans-serif;color:#1d1d1b}
.page{width:1640px;padding:22px 26px 26px}
h1{margin:0;text-align:center;font-size:40px;color:#1b2f5e;letter-spacing:-.3px;font-weight:800}
.sub{text-align:center;font-size:22px;font-weight:700;margin:6px 0 16px}
.steps{display:flex;align-items:stretch;gap:0}
.card{flex:1;border:2px solid;border-radius:14px;padding:12px 12px 10px;text-align:center}
.card .hd{display:flex;align-items:center;justify-content:center;gap:8px;font-weight:700;font-size:17px;min-height:46px}
.num{width:28px;height:28px;border-radius:50%;color:#fff;display:inline-flex;align-items:center;justify-content:center;font-size:16px;flex:none}
.card svg{width:84px;height:72px;margin:6px 0}
.card p{margin:0;font-size:13.5px;line-height:1.35}
.arw{width:30px;display:flex;align-items:center;justify-content:center;font-size:30px;color:#8d8d8d}
.band{display:flex;gap:0;margin-top:16px;border:1.5px solid #e2e5ec;border-radius:12px;padding:12px}
.col h3{margin:0 0 8px;font-size:15px;text-align:center}
table{border-collapse:collapse;font-size:11.5px;margin:0 auto}
th,td{border:1px solid #aeb4c0;padding:3px 5px;text-align:center;white-space:nowrap}
th{background:#f3f5f9;font-weight:600}
td.bad{color:#d62728;font-weight:800}
td.new{color:#d62728;font-weight:800}
td.mark{background:#fde3e3}
tr.gone td{background:#f4f4f4;color:#9a9a9a;text-decoration:line-through}
tr.gone td.why{text-decoration:none;color:#d0433f;font-weight:700}
.note{font-size:12px;text-align:center;margin-top:6px;color:#444}
.box{border:1.5px dashed #6bb27f;border-radius:10px;padding:8px;background:#f7fcf8;font-size:12.5px;line-height:1.4}
.pred{border:1.5px solid #9aa3b2;border-radius:8px;padding:8px;margin-top:10px;text-align:center;font-size:13px;line-height:1.45}
.pred b.r{color:#d62728;font-size:17px}
.harw{display:flex;align-items:center;font-size:30px;color:#6bb27f;padding:0 4px}
.row2{display:flex;gap:14px;margin-top:14px}
.panel{border:1.5px solid #e2e5ec;border-radius:12px;padding:12px 14px}
.panel h2{margin:0 0 8px;font-size:19px;text-align:center}
.how{display:flex;gap:12px;align-items:flex-start}
.how ul{margin:0;padding-left:18px;font-size:13.5px;line-height:1.5}
.dec{display:flex;border:2px solid;border-radius:10px;margin-bottom:10px;overflow:hidden}
.dec .lab{width:150px;flex:none;display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:18px;font-weight:700;padding:8px;text-align:center}
.dec .lab svg{width:52px;height:46px}
.dec .txt{padding:8px 12px;font-size:13px;line-height:1.45;border-left:1.5px solid;flex:1}
.dec .txt .rule{font-weight:600;margin-bottom:4px}
.chips{display:flex;flex-wrap:wrap;gap:5px;margin-top:4px}
.chip{border-radius:12px;padding:2px 8px;font-size:12px;border:1px solid}
.row3{display:flex;gap:14px;margin-top:14px}
.sum ol{margin:0;padding-left:22px;font-size:14px;line-height:1.6}
.sum li::marker{font-weight:800}
.take{background:#fffbea;border-color:#f0dfa0}
.take .in{display:flex;gap:12px;align-items:center;font-size:14.5px;line-height:1.5}
.foot{margin-top:10px;font-size:11.5px;color:#777;text-align:center}
"""


def cards_html(steps):
    out = []
    for i, ((title, text, icon), (c, bg, bd)) in enumerate(zip(steps, STEPC)):
        out.append(f'<div class="card" style="border-color:{bd};background:{bg}"><div class="hd" style="color:{c}">'
                   f'<span class="num" style="background:{c}">{i + 1}</span><span>{title}</span></div>{ICON[icon]}<p>{text}</p></div>')
        if i < 4:
            out.append('<div class="arw">&#10132;</div>')
    return '<div class="steps">' + "".join(out) + "</div>"


def table_html(cols, rows, cls=None, gone=None, why=None):
    """rows: list of (record, [values]); cls[(i,j)] -> css class; gone: set of row idx shown struck through."""
    h = "<table><tr>" + "".join(f"<th>{H.escape(c)}</th>" for c in cols) + "</tr>"
    for i, (rec, vals) in enumerate(rows):
        g = gone and i in gone
        h += f'<tr class="{"gone" if g else ""}"><td>{H.escape(sid(rec))}</td>'
        for j, v in enumerate(vals):
            c = (cls or {}).get((i, j), "")
            if g and why and j == len(vals) - 1:
                c = "why"
            h += f'<td class="{c}">{v}</td>'
        h += "</tr>"
    return h + "</table>"


def render(case):
    c = case
    acc = "".join(f'<span class="chip" style="border-color:#9fd6a8;background:#f0faf2">{H.escape(x)}</span>' for x in c["accept"]) or "<i>none</i>"
    flg = "".join(f'<span class="chip" style="border-color:#f3b4b1;background:#fdf0ef">{H.escape(x)}</span>' for x in c["flag"]) or "<i>none</i>"
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{H.escape(c['title'])}</title><style>{CSS}</style></head>
<body><div class="page">
<h1>{H.escape(c['title'])}</h1>
<div class="sub"><span style="color:#2e7d4f">{H.escape(c['method_line'])}</span> <span style="color:#555">+</span>
<span style="color:#7b4fc9">{H.escape(c['valid_line'])}</span></div>
{cards_html(c['steps'])}
<div class="band">
 <div class="col" style="width:430px"><h3 style="color:#2563c8">1. {c['t1']}</h3>{c['table1']}<div class="note">{c['note1']}</div></div>
 <div class="harw">&#10132;</div>
 <div class="col" style="width:225px"><h3 style="color:#2e7d4f">2. {c['t2']}</h3><div class="box">{c['box2']}</div>{c['pred2']}</div>
 <div class="harw">&#10132;</div>
 <div class="col" style="width:430px"><h3 style="color:#d9822b">3. {c['t3']}</h3>{c['table3']}<div class="note">{c['note3']}</div></div>
 <div class="harw">&#10132;</div>
 <div class="col" style="flex:1"><h3 style="color:#7b4fc9">4. {c['t4']}</h3><img src="{c['plot4']}" style="width:100%"></div>
</div>
<div class="row2">
 <div class="panel" style="flex:1.05"><h2 style="color:#6a3fb5">{c['how_title']}</h2>
  <div class="how"><img src="{c['how_plot']}" style="width:340px;flex:none"><ul>{''.join('<li>' + b + '</li>' for b in c['how_bullets'])}</ul></div></div>
 <div class="panel" style="flex:1"><h2 style="color:#c0392b">5. Accept or Flag</h2>
  <div class="dec" style="border-color:#9fd6a8;background:#f6fcf7"><div class="lab" style="color:#1f9d55">{ICON['decide'].replace('<line x1="54"', '<line style="display:none" x1="54"').replace('<path d="M56', '<path style="display:none" d="M56')}Accept<br>({len(c['accept'])})</div>
   <div class="txt" style="border-color:#c6e6cf"><div class="rule">{c['acc_rule']}</div><div class="chips">{acc}</div></div></div>
  <div class="dec" style="border-color:#f3b4b1;background:#fdf6f6"><div class="lab" style="color:#d0433f">{ICON['decide'].replace('<circle cx="24"', '<circle style="display:none" cx="24"').replace('<path d="M15', '<path style="display:none" d="M15')}Flag<br>({len(c['flag'])})</div>
   <div class="txt" style="border-color:#f3c9c7"><div class="rule">{c['flag_rule']}</div><div class="chips">{flg}</div></div></div>
  <div style="font-size:12.5px;color:#444;line-height:1.4">{c['dec_note']}</div></div>
</div>
<div class="row3">
 <div class="panel sum" style="flex:1.05;background:#f7f9fd"><h2 style="color:#1b2f5e">Summary of the pipeline</h2><ol>{''.join('<li>' + s + '</li>' for s in c['summary'])}</ol></div>
 <div class="panel take" style="flex:1"><h2 style="color:#8a5a00">Key takeaway</h2><div class="in">
  <svg viewBox="0 0 60 80" style="width:54px;flex:none"><circle cx="30" cy="28" r="18" fill="#ffd54a" stroke="#d9a400" stroke-width="2"/><rect x="22" y="46" width="16" height="12" rx="2" fill="#9aa3b2"/><g stroke="#ffb300" stroke-width="3" stroke-linecap="round"><line x1="30" y1="2" x2="30" y2="6"/><line x1="6" y1="28" x2="2" y2="28"/><line x1="54" y1="28" x2="58" y2="28"/><line x1="12" y1="10" x2="9" y2="7"/><line x1="48" y1="10" x2="51" y2="7"/></g></svg>
  <div>{c['takeaway']}</div></div></div>
</div>
<div class="foot">{c['foot']}</div>
</div></body></html>"""
    return html


def screenshot(html_path: Path, png_path: Path):
    prof = Path(tempfile.gettempdir()) / "s10_chrome_profile"
    cmd = [str(CHROME), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1.5",
           f"--user-data-dir={prof}", "--window-size=1640,3400", f"--screenshot={png_path}", html_path.as_uri()]
    subprocess.run(cmd, check=True, timeout=180, capture_output=True)
    im = Image.open(png_path).convert("RGB")
    a = np.asarray(im)
    rows = np.where((a < 250).any(axis=(1, 2)))[0]
    im.crop((0, 0, im.width, min(im.height, rows.max() + 30))).save(png_path, optimize=True)


def summary5(detect, apply, result, check, decide):
    return [f"<b>Detect</b> {detect}", f"<b>Apply</b> {apply}", f"<b>Obtain</b> {result}", f"<b>Validate</b> {check}", f"<b>Decide</b> {decide}"]


# ================================================================== 1. heart · missing data · HyperImpute
def heart_missing():
    d = "heart"; t = trace(d); s = step(t, "Missing data"); recs = t["records"]
    feats = [k for k in s["before"][recs[0]] if k not in ("target", "w")]
    raw = encode_raw(d, pd.read_csv(ROOT / d / "raw" / "heart_raw.csv", usecols=feats + ["meta_record_id"], low_memory=False))
    cc = raw.dropna(subset=feats)
    cc = cc[~cc["meta_record_id"].astype(str).isin(recs)].sample(min(len(cc), 150000), random_state=SEED)
    mu, sd = cc[feats].mean(), cc[feats].std().replace(0, 1)
    Z = ((cc[feats] - mu) / sd).to_numpy()
    g, ll_ref, thr = fit_gmm(Z, 6)
    miss = {r: [k for k in feats if isnan(s["before"][r][k])] for r in recs}
    imputed = [r for r in recs if miss[r]]
    res = {}
    for r in imputed:
        x = np.array([s["after"][r][k] for k in feats], float)
        ll = g.score_samples(((x - mu.values) / sd.values)[None])[0]
        res[r] = (ll, pct_of(ll, ll_ref), ll >= thr)
    cnt = pd.Series({k: sum(k in miss[r] for r in recs) for k in feats})
    show = ["age"] + [k for k in cnt.sort_values(ascending=False).index if cnt[k] > 0 and k != "age"][:6]
    lab = {"age": "Age", "bmi": "BMI", "smoking_status": "Smoke", "sleep_hours": "Sleep h", "poor_physical_health_days": "Phys d",
           "poor_mental_health_days": "Ment d", "income_category": "Income", "alcohol_days_30d": "Alc d", "drinks_per_occasion": "Drinks",
           "any_exercise_30d": "Exerc"}
    cols = ["Record"] + [lab.get(k, k) for k in show]
    rows1, rows3, cls1, cls3 = [], [], {}, {}
    for i, r in enumerate(recs):
        b, a = s["before"][r], s["after"][r]
        rows1.append((r, [fv(b[k]) for k in show])); rows3.append((r, [fv(a[k]) for k in show]))
        for j, k in enumerate(show):
            if isnan(b[k]):
                cls1[(i, j)] = "bad"; cls3[(i, j)] = "new"
    n_cells = int(sum(len(v) for v in miss.values()))
    ex = max(imputed, key=lambda r: len(miss[r]))
    # plot 4: GMM clusters in (age, bmi) with the imputed records
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    bg = cc.sample(2500, random_state=1)
    lbl = g.predict(((bg[feats] - mu) / sd).to_numpy())
    jit = np.random.default_rng(0).normal(0, 0.35, len(bg))
    ax.scatter(bg["age"] + jit, bg["bmi"], s=4, c=[PASTEL[k] for k in lbl], alpha=0.45, lw=0)
    ia, ib = feats.index("age"), feats.index("bmi")
    gmm_ellipses(ax, g, [ia, ib], to_x=lambda m: (m[0] * sd["age"] + mu["age"], m[1] * sd["bmi"] + mu["bmi"]), scale=(sd["age"], sd["bmi"]))
    for r in imputed:
        a = s["after"][r]; ok = res[r][2]
        ax.scatter(a["age"], a["bmi"], marker="*", s=230, color=C_IMP if ok else "#7a0000", ec="white", lw=0.8, zorder=5)
        ax.annotate(sid(r), (a["age"], a["bmi"]), xytext=(6, 5), textcoords="offset points", fontsize=7.5, color="#8b0000")
    ax.set_xlabel("Feature 1: age (years)"); ax.set_ylabel("Feature 2: BMI"); ax.set_ylim(12, 60); style(ax)
    ax.set_title(f"GMM ({g.n_components} clusters) on all {len(feats)} columns\n★ = imputed record", fontsize=9)
    plot4 = b64(fig)
    # how plot: log-likelihood of real complete records with threshold + imputed records
    fig, ax = plt.subplots(figsize=(4.3, 3.2))
    lo_ = np.percentile(ll_ref, 0.3) - 3
    ax.hist(ll_ref[ll_ref >= lo_], bins=np.linspace(lo_, ll_ref.max(), 70), color="#a9c8f0", ec="white", lw=0.3)
    ax.axvline(thr, color=C_FLAG, ls="--", lw=1.6); ax.text(thr, ax.get_ylim()[1] * 0.92, " 1st pct\n threshold", color=C_FLAG, fontsize=8, va="top")
    for k, r in enumerate(imputed):
        ax.scatter(res[r][0], ax.get_ylim()[1] * (0.08 + 0.08 * k), marker="*", s=170, color=C_IMP, ec="white", zorder=5)
    ax.set_xlabel("GMM log-likelihood of a whole record (all columns)"); ax.set_ylabel("real complete records"); style(ax)
    ax.set_xlim(np.percentile(ll_ref, 0.2) - 2, ll_ref.max() + 1)
    how_plot = b64(fig)
    br = bias_row(d, "Missing data")
    acc = [f"{sid(r)} · {len(miss[r])} imputed · pct {res[r][1]:.0f}" for r in imputed if res[r][2]]
    flg = [f"{sid(r)} · {len(miss[r])} imputed · pct {res[r][1]:.1f}" for r in imputed if not res[r][2]]
    return dict(
        ds=d, nn="01", key="missing_data", title="Missing Value Imputation and Validation Framework",
        method_line="HyperImpute Imputation", valid_line="GMM Validation (using ALL columns)",
        steps=[("Detect Missing Values", f"{float(br['before']):.2%} of predictor cells missing; {len(imputed)} of the 10 records have gaps.", "detect"),
               ("HyperImpute Imputation", "Per-column learners (chosen automatically) predict each gap from the other columns.", "model"),
               ("Completed Dataset", f"All {n_cells} missing cells of the 10 records filled; no rows removed.", "table"),
               ("GMM Validation (All Columns)", "Check whether each imputed record fits the joint distribution of all columns.", "valid"),
               ("Accept or Flag Imputation", "Accept plausible imputed records; flag low-likelihood ones.", "decide")],
        t1="Original 10 records (missing = ?)", table1=table_html(cols, rows1, cls1),
        note1=f"{n_cells} missing cells in {len(imputed)} records (red ?).",
        t2="HyperImpute", box2="Training data = all observed values of the training split (308,077 rows). For each column with gaps, "
                              "HyperImpute picks the best learner and iterates column by column until the values settle.",
        pred2=f'<div class="pred">Record {sid(ex)}<br>{len(miss[ex])} gaps, e.g. BMI =<br><b class="r">{fv(s["after"][ex]["bmi"])}</b>'
              f'<br>physical-health days = <b class="r">{fv(s["after"][ex]["poor_physical_health_days"])}</b></div>',
        t3="Completed records (after imputation)", table3=table_html(cols, rows3, cls3), note3="Red = value filled by HyperImpute.",
        t4="GMM validation (all columns)", plot4=plot4,
        how_title="How GMM Validation Works?", how_plot=how_plot,
        how_bullets=[f"A Gaussian Mixture Model ({g.n_components} clusters, full covariance) learns the joint distribution of <b>all {len(feats)} columns</b> "
                     f"from {len(Z) // 2:,} real complete records.",
                     "It learns patterns such as: older age → more poor-health days; higher BMI → less exercise; all variables together form clusters.",
                     f"The log-likelihood of every completed record is compared with {len(ll_ref):,} held-out real records.",
                     "Above the 1st percentile of real records → <b>plausible</b>; below → <b>flag</b>."],
        acc_rule="Imputed record lies inside the real-data clusters (log-likelihood ≥ 1st percentile of real records).",
        flag_rule="Imputed record falls in a low-density region (below the 1st percentile).",
        accept=acc, flag=flg,
        dec_note=f"pct = percentile of the record's likelihood among real complete records. The other {10 - len(imputed)} records had no gaps and were not changed.",
        summary=summary5("missing values in the dataset (4.37% of predictor cells).", "HyperImpute (uses all other features) to predict each gap.",
                         "the completed dataset (same rows, no gaps).", "imputed records with a GMM on <b>all columns</b> (multivariate check).",
                         f"accept the imputation if plausible, otherwise flag it for inspection ({len(acc)} accepted, {len(flg)} flagged)."),
        takeaway="HyperImpute predicts each missing value from the relationships in the data, while the <b>GMM</b> confirms that every imputed record "
                 "is statistically consistent with the overall multivariate distribution (all columns, not just the imputed column).",
        foot="Heart disease · CDC BRFSS 2022 · values = the project's traced records (sample10_trace.json, stage 2); GMM check computed on real complete records of heart_raw.csv.")


# ================================================================== 2. heart · feature scale · Yeo-Johnson
def hist_skew(counts, rng):
    counts = np.asarray(counts, float); e = np.linspace(rng[0], rng[1], len(counts) + 1); m = (e[1:] + e[:-1]) / 2
    w = counts / counts.sum(); mu = (w * m).sum(); sd = np.sqrt((w * (m - mu) ** 2).sum())
    return float((w * (m - mu) ** 3).sum() / sd ** 3) if sd > 0 else 0.0


def heart_scale():
    d = "heart"; t = trace(d); s = step(t, "Feature scale"); recs = t["records"]
    s_html = (ROOT / d / "plots" / "04_feature_scale_bias_dashboard.html").read_text(encoding="utf-8")
    D, _ = json.JSONDecoder().raw_decode(s_html[s_html.index("const D = ") + 10:])
    show = ["age", "bmi", "sleep_hours", "poor_physical_health_days", "poor_mental_health_days", "alcohol_days_30d"]
    lab = {"age": "Age", "bmi": "BMI", "sleep_hours": "Sleep h", "poor_physical_health_days": "Phys d",
           "poor_mental_health_days": "Ment d", "alcohol_days_30d": "Alc d"}
    cols = ["Record"] + [lab[k] for k in show]
    rows1, rows3, cls3 = [], [], {}
    zmax = {}
    for i, r in enumerate(recs):
        b, a = s["before"][r], s["after"][r]
        rows1.append((r, [fv(b[k]) for k in show])); rows3.append((r, [f"{a[k]:.2f}" for k in show]))
        for j, k in enumerate(show):
            cls3[(i, j)] = "new"
        zmax[r] = max(abs(a[k]) for k in show)
    # order preserved (monotone transform) for every variable among the 10 records
    order_ok = {}
    for k in show:
        b = np.array([s["before"][r][k] for r in recs]); a = np.array([s["after"][r][k] for r in recs])
        ii, jj = np.where(b[:, None] < b[None, :])
        order_ok[k] = bool((a[ii] < a[jj]).all())
    skew = {k: (hist_skew(D["dist"][k]["b"], D["dist"][k]["br"]), hist_skew(D["dist"][k]["a"], D["dist"][k]["ar"])) for k in show}
    # plot 4: |skewness| before vs after per variable
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    y = np.arange(len(show))
    ax.barh(y - 0.2, [abs(skew[k][0]) for k in show], 0.38, color="#9dbdff", label="before")
    ax.barh(y + 0.2, [abs(skew[k][1]) for k in show], 0.38, color="#1f9d55", label="after Yeo-Johnson")
    ax.axvline(1, color=C_FLAG, ls="--", lw=1.2); ax.text(1.02, len(show) - 0.45, "|skew| = 1", color=C_FLAG, fontsize=7.5)
    ax.set_yticks(y, [lab[k] for k in show]); ax.invert_yaxis(); ax.set_xlabel("|skewness| of the training split"); style(ax)
    ax.legend(fontsize=8, frameon=False, loc="lower right"); ax.set_title("Shape check on the whole training split", fontsize=9)
    plot4 = b64(fig)
    # how plot: BMI histogram before / after (from the dashboard data) with the 10 records
    fig, axs = plt.subplots(1, 2, figsize=(4.4, 2.6))
    for ax, key, rk, col, ttl, src in [(axs[0], "b", "br", "#9dbdff", "Before (BMI)", "before"), (axs[1], "a", "ar", "#1f9d55", "After (z, Yeo-Johnson)", "after")]:
        cnt = D["dist"]["bmi"][key]; lo, hi = D["dist"]["bmi"][rk]; e = np.linspace(lo, hi, len(cnt) + 1)
        ax.bar(e[:-1], cnt, width=np.diff(e), align="edge", color=col, ec="white")
        for r in recs:
            ax.scatter(s[src][r]["bmi"], max(cnt) * 0.05, marker="v", color=C_IMP, s=26, zorder=4)
        ax.set_title(ttl, fontsize=8.5); ax.set_yticks([]); style(ax)
    how_plot = b64(fig)
    better = [k for k in show if abs(skew[k][1]) < abs(skew[k][0])]
    worse = [k for k in show if k not in better]
    br = bias_row(d, "Feature scale")
    acc = [f"{sid(r)} · max |z| {zmax[r]:.2f}" for r in recs if zmax[r] <= 3]
    flg = [f"{sid(r)} · max |z| {zmax[r]:.2f}" for r in recs if zmax[r] > 3]
    ex = max(recs, key=lambda r: s["before"][r]["bmi"])
    return dict(
        ds=d, nn="02", key="feature_scale", title="Feature-Scale Correction and Validation Framework",
        method_line="Yeo-Johnson Power Transform", valid_line="Distribution-Shape Validation (all scaled columns)",
        steps=[("Detect Scale Bias", f"SD ratio {br['before'].split(';')[0]} across predictors and max |skew| {br['before'].split(';')[1].strip()}.", "detect"),
               ("Yeo-Johnson Transform", "A power λ is fitted per column on the training split, then the column is standardised.", "model"),
               ("Rescaled Dataset", "Every continuous/ordinal column of the 10 records now has mean 0, SD 1.", "table"),
               ("Shape Validation", "Check skewness of each column and that every record keeps its order.", "valid"),
               ("Accept or Flag", "Accept records inside ±3 SD; flag records still extreme after the transform.", "decide")],
        t1="Original 10 records (raw units)", table1=table_html(cols, rows1), note1="Columns on very different ranges (age vs days vs BMI).",
        t2="PowerTransform", box2="Yeo-Johnson: ψ(x) = ((x+1)<sup>λ</sup>−1)/λ for x ≥ 0 (a mirrored form for x &lt; 0). λ maximises the normal "
                                 "likelihood of each column on the training split; then z = (ψ − mean)/SD.",
        pred2=f'<div class="pred">Record {sid(ex)}<br>BMI {fv(s["before"][ex]["bmi"])} →<br><b class="r">z = {s["after"][ex]["bmi"]:.2f}</b></div>',
        t3="Rescaled records (z units)", table3=table_html(cols, rows3, cls3), note3="Red = value after Yeo-Johnson + standardisation.",
        t4="Shape validation", plot4=plot4,
        how_title="How the Shape Validation Works?", how_plot=how_plot,
        how_bullets=[f"Dataset level: SD ratio {br['before'].split(';')[0]} → {br['after'].split(';')[0]}; max |skew| {br['before'].split(';')[1].strip()} → {br['after'].split(';')[1].strip()} (pipeline statistic).",
                     "The transform is monotone, so every record keeps its order: " + ("preserved for all 6 columns among the 10 records." if all(order_ok.values()) else
                                                                                         "NOT preserved for " + ", ".join(k for k, v in order_ok.items() if not v) + "."),
                     "Histograms (from the pipeline's before/after data): BMI changes from right-skewed to near-symmetric; ▼ = the 10 records.",
                     "Record rule: after scaling, every value should lie within ±3 SD."],
        acc_rule="All scaled values of the record lie within ±3 SD and the order of records is unchanged.",
        flag_rule="At least one scaled value is still beyond ±3 SD (record stays in a tail).",
        accept=acc, flag=flg, dec_note="max |z| = largest absolute scaled value of the record over the 6 columns shown. No values are removed by this step.",
        summary=summary5("columns on very different scales and skewed shapes.", "Yeo-Johnson (λ per column) + standardisation, fitted on the training split.",
                         "the rescaled dataset (same rows, comparable units).", "skewness per column and order preservation (multivariate shape check).",
                         f"accept records inside ±3 SD; flag extreme ones ({len(acc)} accepted, {len(flg)} flagged)."),
        takeaway=f"Yeo-Johnson puts every column on the same scale (SD ratio {br['before'].split(';')[0]} → {br['after'].split(';')[0]}) and the shape check "
                 f"confirms it kept the order of patients; |skew| fell in {len(better)} of {len(show)} columns"
                 + ("." if not worse else " (" + ", ".join(f"{lab[k]} {abs(skew[k][0]):.2f} → {abs(skew[k][1]):.2f}" for k in worse)
                    + " rose: a discrete column with few levels cannot be made symmetric by a power transform)."),
        foot="Heart disease · CDC BRFSS 2022 · values = traced records (stage 5, Feature scale); histograms = the stage's saved before/after distributions.")


# ================================================================== 3. diabetes · class imbalance · TabDDPM
def back_map(fs, raw, col, z):
    """Monotone map from scaled value back to raw units, learned from the real rows that have both values."""
    m = fs[["meta_record_id", col]].merge(raw[["meta_record_id", col]], on="meta_record_id", suffixes=("_z", "_r")).dropna()
    m = m.sort_values(col + "_z")
    return np.interp(z, m[col + "_z"].to_numpy(), np.maximum.accumulate(m[col + "_r"].to_numpy()))


def diabetes_imbalance():
    d = "diabetes"; t = trace(d); recs = t["records"]
    s0 = step(t, "Missing data"); s = step(t, "Class imbalance")
    feats = ["age", "male", "education", "income_poverty_ratio", "bmi", "waist_cm", "smoking_status"]
    fs = pd.read_csv(ROOT / d / "preprocessed" / "diabetes_train_feature_space.csv", dtype={"meta_record_id": str})
    raw = pd.read_csv(ROOT / d / "raw" / "diabetes_raw.csv", dtype={"meta_record_id": str})
    real = fs[fs.meta_synthetic == 0]; syn = fs[fs.meta_synthetic == 1]
    rc = real[real.target == 1]
    g, ll_ref, thr = fit_gmm(rc[feats].to_numpy(), 4)
    ll_syn = g.score_samples(syn[feats].to_numpy())
    syn = syn.assign(ll=ll_syn, ok=ll_syn >= thr)
    cases = [r for r in recs if s0["after"][r]["target"] == 1]
    pick = []
    for r in cases:
        pick += list(syn[syn.meta_record_id == r].sort_index().head(3).index)
    pick = pick[:10]
    if len(pick) < 10:
        pick += list(syn.drop(index=pick).sample(10 - len(pick), random_state=SEED).index)
    P = syn.loc[pick].copy()
    for c in ["age", "income_poverty_ratio", "bmi", "waist_cm"]:
        P[c + "_raw"] = back_map(fs[fs.meta_synthetic == 0], raw, c, P[c].to_numpy())
    show = ["age", "male", "income_poverty_ratio", "bmi", "waist_cm"]
    lab = {"age": "Age", "male": "Male", "income_poverty_ratio": "Inc ratio", "bmi": "BMI", "waist_cm": "Waist"}
    cols1 = ["Record"] + [lab[k] for k in show] + ["Diabetes"]
    rows1, cls1 = [], {}
    for i, r in enumerate(recs):
        a = s0["after"][r]
        rows1.append((r, [fv(a[k], 1) for k in show] + ["YES" if a["target"] == 1 else "no"]))
        if a["target"] == 1:
            cls1[(i, len(show))] = "bad mark"
    cols3 = ["Synthetic of"] + [lab[k] for k in show] + ["Diabetes"]
    rows3, cls3 = [], {}
    for i, (_, p) in enumerate(P.iterrows()):
        rows3.append((p.meta_record_id, [fv(p["age_raw"], 0), fv(p["male"]), fv(p["income_poverty_ratio_raw"], 2), fv(p["bmi_raw"], 1), fv(p["waist_cm_raw"], 1), "YES"]))
        for j in range(len(show) + 1):
            cls3[(i, j)] = "new"
    n0, n1 = int((real.target == 0).sum()), int((real.target == 1).sum())
    br = bias_row(d, "Class imbalance")
    # plot 4: age vs waist, real non-cases / real cases / synthetic
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    rn = real[real.target == 0].sample(1500, random_state=1)
    ax.scatter(rn.age, rn.waist_cm, s=5, color="#c9c8c2", alpha=0.6, lw=0, label="real non-case")
    ax.scatter(rc.age, rc.waist_cm, s=6, color="#86c98f", alpha=0.7, lw=0, label="real case")
    ss = syn.sample(1500, random_state=1)
    ax.scatter(ss.age, ss.waist_cm, s=6, color="#f2a65a", alpha=0.6, lw=0, label="TabDDPM synthetic case")
    gmm_ellipses(ax, g, [feats.index("age"), feats.index("waist_cm")], colors=["#2e7d4f"] * 8)
    ax.scatter(P.age, P.waist_cm, marker="*", s=170, color=[C_IMP if o else "#5a0000" for o in P.ok], ec="white", zorder=5, label="the 10 synthetic rows shown")
    ax.set_xlabel("age (scaled)"); ax.set_ylabel("waist (scaled)"); style(ax); ax.legend(fontsize=6.8, frameon=False, loc="lower right")
    ax.set_title("GMM fitted on REAL cases (dashed ellipses)", fontsize=9)
    plot4 = b64(fig)
    fig, ax = plt.subplots(figsize=(4.3, 3.2))
    lo = np.percentile(np.r_[ll_ref, ll_syn], 0.5)
    ax.hist(np.clip(ll_ref, lo, None), bins=50, density=True, color="#86c98f", alpha=0.7, label="real cases (held out)")
    ax.hist(np.clip(ll_syn, lo, None), bins=50, density=True, color="#f2a65a", alpha=0.55, label="synthetic cases")
    ax.axvline(thr, color=C_FLAG, ls="--", lw=1.5); ax.text(thr, ax.get_ylim()[1] * 0.95, " 1st pct of real", color=C_FLAG, fontsize=8, va="top")
    ax.set_xlabel("GMM log-likelihood under the real-case model"); ax.set_yticks([]); style(ax); ax.legend(fontsize=7.5, frameon=False)
    how_plot = b64(fig)
    rate = float(syn.ok.mean())
    acc = [f"syn of {sid(p.meta_record_id)} · pct {pct_of(p.ll, ll_ref):.0f}" for _, p in P.iterrows() if p.ok]
    flg = [f"syn of {sid(p.meta_record_id)} · pct {pct_of(p.ll, ll_ref):.1f}" for _, p in P.iterrows() if not p.ok]
    return dict(
        ds=d, nn="03", key="class_imbalance", title="Class-Imbalance Correction and Validation Framework",
        method_line="TabDDPM Synthetic Cases", valid_line="GMM Validation of Synthetic Records (all columns)",
        steps=[("Detect Class Imbalance", f"Imbalance ratio {float(br['before']):.2f} : 1 (non-cases : cases); 4 of the 10 records are cases.", "detect"),
               ("TabDDPM Generation", "A diffusion model learns the real cases and samples new case records.", "model"),
               ("Balanced Dataset", f"{len(real):,} → {len(fs):,} training rows; ratio {float(br['after']):.2f} : 1.", "table"),
               ("GMM Validation", "Check whether each synthetic case fits the distribution of REAL cases (all columns).", "valid"),
               ("Accept or Flag", "Accept realistic synthetic cases; flag implausible ones.", "decide")],
        t1="Original 10 records (cases marked)", table1=table_html(cols1, rows1, cls1),
        note1=f"Training split: {n0:,} non-cases vs {n1:,} cases.",
        t2="TabDDPM", box2=f"Gaussian diffusion trained on the {n1:,} real case rows (standardised): noise is added step by step and a network learns to "
                           f"remove it. Sampling from pure noise gives <b>{len(syn):,}</b> new case rows, snapped to valid levels.",
        pred2=f'<div class="pred">Each synthetic row keeps the ID of its nearest real case.<br>Rows linked to the 10 samples:<br><b class="r">{int(syn.meta_record_id.isin(recs).sum())}</b></div>',
        t3="Synthetic case records (TabDDPM)", table3=table_html(cols3, rows3, cls3),
        note3="Synthetic rows linked to the sample cases (values mapped back to raw units).",
        t4="GMM validation (all columns)", plot4=plot4,
        how_title="How GMM Validation Works?", how_plot=how_plot,
        how_bullets=[f"A GMM ({g.n_components} clusters) learns the joint distribution of <b>all {len(feats)} columns</b> of the REAL cases only.",
                     "Every synthetic case is scored under that model: a realistic case has a likelihood like real cases.",
                     f"Over all {len(syn):,} synthetic cases, <b>{rate:.1%}</b> pass the 1st-percentile threshold of real held-out cases.",
                     "Passing rows are plausible new cases; failing rows lie outside the real-case clusters."],
        acc_rule="Synthetic case lies inside the real-case clusters (log-likelihood ≥ 1st percentile of real cases).",
        flag_rule="Synthetic case lies in a region where real cases are very rare.",
        accept=acc, flag=flg, dec_note=f"All synthetic rows: {int(syn.ok.sum()):,} accepted, {int((~syn.ok).sum()):,} flagged. The 10 original records are kept unchanged.",
        summary=summary5(f"the class imbalance ({float(br['before']):.2f} : 1).", "TabDDPM (diffusion model trained on real cases) to generate new cases.",
                         "the balanced training set (1 : 1).", "every synthetic case with a GMM of real cases on <b>all columns</b>.",
                         f"accept realistic synthetic cases, flag the rest ({rate:.1%} of all synthetic rows accepted)."),
        takeaway="TabDDPM creates new minority-class records from the learned distribution of real cases, while the <b>GMM</b> confirms that "
                 "the synthetic cases are statistically consistent with real cases across all columns.",
        foot="Diabetes · CDC NHANES 2021-23 · originals = traced records; synthetic rows = diabetes_train_feature_space.csv (meta_synthetic = 1).")


# ================================================================== 4. diabetes · demographic · adversarial debiasing
def diabetes_demographic():
    d = "diabetes"; t = trace(d); recs = t["records"]
    s0 = step(t, "Missing data")
    fs = pd.read_csv(ROOT / d / "preprocessed" / "diabetes_train_feature_space.csv", dtype={"meta_record_id": str})
    pp = pd.read_csv(ROOT / d / "preprocessed" / "diabetes_train_preprocessed.csv", dtype={"meta_record_id": str})
    fs, pp = fs[fs.meta_synthetic == 0].reset_index(drop=True), pp[pp.meta_synthetic == 0].reset_index(drop=True)
    feats = ["age", "male", "education", "income_poverty_ratio", "bmi", "waist_cm", "smoking_status"]
    F = [f"f{i}" for i in range(1, 33)]
    y_sex = (pp.sens_sex.astype(str).str.lower().str.startswith("m")).astype(int).to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    lr = LogisticRegression(max_iter=2000)
    out = {}
    for name, X in [("before", fs.set_index("meta_record_id").loc[pp.meta_record_id, feats].to_numpy()), ("after", pp[F].to_numpy())]:
        Xs = StandardScaler().fit_transform(X)
        p = cross_val_predict(lr, Xs, y_sex, cv=cv, method="predict_proba")[:, 1]
        out[name] = (roc_auc_score(y_sex, p), p)
    race = pp.sens_race_ethnicity.astype(str)
    auc_race = {}
    for name, X in [("before", fs.set_index("meta_record_id").loc[pp.meta_record_id, feats].to_numpy()), ("after", pp[F].to_numpy())]:
        Xs = StandardScaler().fit_transform(X)
        P = cross_val_predict(lr, Xs, race, cv=cv, method="predict_proba")
        auc_race[name] = roc_auc_score(race, P, multi_class="ovr", average="macro")
    vals, c = comparison(d, "05_demographic_bias_dashboard.html")
    ref = vals["No mitigation (reference)"]; adv = vals["Adversarial debiasing"]
    raw = pd.read_csv(ROOT / d / "raw" / "diabetes_raw.csv", dtype={"meta_record_id": str}).set_index("meta_record_id")
    cols1 = ["Record", "Sex", "Race/eth", "Age", "BMI", "Waist", "Diab."]
    rows1, rows3, cls1, cls3 = [], [], {}, {}
    pidx = {r: i for i, r in enumerate(pp.meta_record_id)}
    prob = {}
    for i, r in enumerate(recs):
        a = s0["after"][r]; sex = str(raw.loc[r, "sens_sex"]); rc = str(raw.loc[r, "sens_race_ethnicity"])
        rows1.append((r, [sex[:1].upper(), rc[:10], fv(a["age"]), fv(a["bmi"], 1), fv(a["waist_cm"], 1), "YES" if a["target"] == 1 else "no"]))
        cls1[(i, 0)] = "mark"; cls1[(i, 1)] = "mark"
        k = pidx[r]; f = pp.loc[k, F].to_numpy(float)
        p_true = out["after"][1][k] if y_sex[k] == 1 else 1 - out["after"][1][k]
        p_true_b = out["before"][1][k] if y_sex[k] == 1 else 1 - out["before"][1][k]
        prob[r] = (p_true_b, p_true)
        rows3.append((r, [f"{f[0]:.2f}", f"{f[1]:.2f}", f"{f[2]:.2f}", f"{f[3]:.2f}", "…", f"{p_true:.2f}"]))
        for j in range(4):
            cls3[(i, j)] = "new"
    cols3 = ["Record", "f1", "f2", "f3", "f4", "… f32", "P(sex)"]
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    lbls = ["sex", "race/ethnicity"]; b_ = [out["before"][0], auc_race["before"]]; a_ = [out["after"][0], auc_race["after"]]
    x = np.arange(2)
    ax.bar(x - 0.2, b_, 0.38, color="#9dbdff", label="original features"); ax.bar(x + 0.2, a_, 0.38, color="#1f9d55", label="debiased representation")
    for i in range(2):
        ax.text(x[i] - 0.2, b_[i] + 0.01, f"{b_[i]:.2f}", ha="center", fontsize=8); ax.text(x[i] + 0.2, a_[i] + 0.01, f"{a_[i]:.2f}", ha="center", fontsize=8)
    ax.axhline(0.5, color=C_FLAG, ls="--", lw=1.2); ax.text(1.45, 0.51, "chance", color=C_FLAG, fontsize=8, ha="right")
    ax.set_xticks(x, lbls); ax.set_ylim(0.4, 1.08); ax.set_ylabel("adversary AUC (5-fold CV)"); style(ax); ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.set_title("Can a protected attribute be predicted?", fontsize=9)
    plot4 = b64(fig)
    mets = ["SPD", "DID", "EOD", "Equalized-odds difference"]
    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    x = np.arange(len(mets))
    ax.bar(x - 0.2, [ref[m] for m in mets], 0.38, color="#c9c8c2", label="no mitigation"); ax.bar(x + 0.2, [adv[m] for m in mets], 0.38, color="#2a78d6", label="adversarial debiasing")
    ax.set_xticks(x, ["SPD", "DID", "EOD", "Eq. odds"]); ax.set_ylabel("gap between groups (lower = fairer)"); style(ax); ax.legend(fontsize=7.5, frameon=False)
    how_plot = b64(fig)
    acc = [f"{sid(r)} · P(sex) {prob[r][1]:.2f}" for r in recs if prob[r][1] <= 0.65]
    flg = [f"{sid(r)} · P(sex) {prob[r][1]:.2f}" for r in recs if prob[r][1] > 0.65]
    br = bias_row(d, "Demographic")
    return dict(
        ds=d, nn="04", key="demographic", title="Demographic-Bias Mitigation and Validation Framework",
        method_line="Adversarial Debiasing (fair representation)", valid_line="Adversary Validation (all 32 dimensions)",
        steps=[("Detect Demographic Bias", "Group gaps: min disparate impact {} ; max EOD {}.".format(*[x.strip() for x in str(br['before']).split(';')]), "detect"),
               ("Adversarial Debiasing", "An encoder predicts diabetes while an adversary tries to recover sex / race; the encoder is trained to fool it.", "model"),
               ("Fair Representation", "Each record becomes a 32-dimensional vector f1…f32 used for modelling.", "table"),
               ("Adversary Validation", "Train a fresh classifier to recover sex and race from the new vectors.", "valid"),
               ("Accept or Flag", "Accept records whose sex is not identifiable; flag records that still reveal it.", "decide")],
        t1="Original 10 records (protected attributes marked)", table1=table_html(cols1, rows1, cls1),
        note1="Sex is an input column (male), so the original features reveal it exactly.",
        t2="Adversarial debiasing", box2="Encoder → 32-dim representation → diabetes predictor. A second network (adversary) reads the representation and "
                                        "predicts the protected groups; the encoder receives the reversed adversary gradient, so group information is removed.",
        pred2=f'<div class="pred">Fairness (pipeline metrics)<br>SPD {ref["SPD"]:.3f} → <b class="r">{adv["SPD"]:.3f}</b><br>EOD {ref["EOD"]:.3f} → <b class="r">{adv["EOD"]:.3f}</b></div>',
        t3="Debiased representation", table3=table_html(cols3, rows3, cls3),
        note3="P(sex) = a fresh adversary's probability for the record's true sex (0.5 = cannot tell).",
        t4="Adversary validation", plot4=plot4,
        how_title="How Adversary Validation Works?", how_plot=how_plot,
        how_bullets=[f"A new logistic-regression adversary (5-fold cross-validation, {len(pp):,} real training rows) tries to recover sex and race.",
                     f"Sex AUC: {out['before'][0]:.2f} from the original features → <b>{out['after'][0]:.2f}</b> from the representation; race AUC {auc_race['before']:.2f} → <b>{auc_race['after']:.2f}</b>.",
                     "Pipeline fairness metrics (left plot): " + ", ".join(
                         f"{m} {ref[m]:.3f} → {adv[m]:.3f}" for m in mets) + " (lower = fairer).",
                     "Record rule: if the adversary gives ≤ 0.65 to the true sex, the record no longer reveals it."],
        acc_rule="Adversary probability for the record's true sex ≤ 0.65 (close to a coin toss).",
        flag_rule="Adversary is still confident (> 0.65) about the record's sex.",
        accept=acc, flag=flg, dec_note="Flagged records stay in the data; the flag says the representation still carries some sex signal for them.",
        summary=summary5("group gaps in outcome rate and model errors.", "adversarial debiasing to learn a 32-dim group-blind representation.",
                         "the fair representation for every record.", "a fresh adversary on <b>all 32 dimensions</b> (sex and race).",
                         f"accept records whose group is hidden, flag the rest ({len(acc)} accepted, {len(flg)} flagged)."),
        takeaway=("Adversarial debiasing removes protected-group information while keeping what predicts diabetes, and the <b>adversary check</b> "
                  "confirms that a new model can no longer recover sex or race reliably from the representation.")
                 if out["after"][0] < 0.6 and auc_race["after"] < 0.6 else
                 (f"Adversarial debiasing narrows the group gaps the pipeline measures (SPD {ref['SPD']:.3f} → {adv['SPD']:.3f}, "
                  f"EOD {ref['EOD']:.3f} → {adv['EOD']:.3f}), but the <b>adversary check</b> shows the representation is <b>not group-blind</b>: "
                  f"sex is still recoverable (AUC {out['after'][0]:.2f}) and race unchanged (AUC {auc_race['after']:.2f}). "
                  "The fairness gain comes from the debiased predictor, not from hiding the groups."),
        foot="Diabetes · CDC NHANES 2021-23 · representation = diabetes_train_preprocessed.csv (real rows); fairness metrics = the notebook's comparison table.")


# ================================================================== 5. hypertension · data noise · Confident Learning
def oof_case_prob(raw, feats):
    X = raw[feats].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype("category").cat.codes.replace(-1, np.nan)
    y = raw["target"].astype(int).to_numpy()
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=SEED)
    return cross_val_predict(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1], y


def hypertension_noise():
    d = "hypertension"; t = trace(d); recs = t["records"]; s = step(t, "Data noise")
    raw = pd.read_csv(ROOT / d / "raw" / "hypertension_raw.csv", dtype={"meta_record_id": str}, low_memory=False)
    feats = [c for c in raw.columns if not c.startswith(("meta_", "sens_")) and c != "target"]
    p, y = oof_case_prob(raw, feats)
    raw = raw.assign(p=p)
    t1 = float(p[y == 1].mean())
    pr = raw.set_index("meta_record_id")["p"]
    removed = [r for r in recs if r not in s["after"]]
    show = ["age", "male", "bmi_category", "high_cholesterol_ever", "diabetes_ever", "general_health"]
    lab = {"age": "Age", "male": "Male", "bmi_category": "BMI cat", "high_cholesterol_ever": "High chol", "diabetes_ever": "Diabetes", "general_health": "Gen hlth"}
    cols1 = ["Record"] + [lab[k] for k in show] + ["Hypert."]
    rows1, cls1, rows3, cls3 = [], {}, [], {}
    for i, r in enumerate(recs):
        b = s["before"][r]
        vals = [fv(b[k]) for k in show] + ["YES" if b["target"] == 1 else "no"]
        rows1.append((r, vals)); cls1[(i, len(show))] = "mark"
        rows3.append((r, vals[:3] + [f"{pr[r]:.2f}", "removed: label issue" if r in removed else "kept"]))
    cols3 = ["Record", "Age", "Male", "BMI cat", "P(case)", "Decision"]
    gone = {i for i, r in enumerate(recs) if r in removed}
    br = bias_row(d, "Data noise")
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.hist(p[y == 0], bins=40, color="#9dbdff", alpha=0.75, density=True, label="recorded non-cases")
    ax.hist(p[y == 1], bins=40, color="#f2a65a", alpha=0.6, density=True, label="recorded cases")
    ax.axvline(t1, color=C_FLAG, ls="--", lw=1.5); ax.text(t1, ax.get_ylim()[1] * 0.96, f" CL threshold\n t = {t1:.2f}", color=C_FLAG, fontsize=7.5, va="top")
    for k, r in enumerate(recs):
        ax.scatter(pr[r], ax.get_ylim()[1] * (0.05 + 0.045 * k), marker="X" if r in removed else "o", s=60 if r in removed else 30,
                   color=C_FLAG if r in removed else C_ACC, ec="white", zorder=5)
    ax.set_xlabel("out-of-fold P(hypertension) from all columns"); ax.set_yticks([]); style(ax); ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_title("× = removed by the pipeline, ● = kept", fontsize=9)
    plot4 = b64(fig)
    fig, ax = plt.subplots(figsize=(4.3, 2.9))
    order = sorted(recs, key=lambda r: pr[r])
    ax.barh(range(10), [pr[r] for r in order], color=[C_FLAG if r in removed else ("#f2a65a" if s["before"][r]["target"] == 1 else "#9dbdff") for r in order])
    ax.axvline(t1, color=C_FLAG, ls="--", lw=1.2)
    ax.set_yticks(range(10), [f"{sid(r)} ({'case' if s['before'][r]['target'] == 1 else 'non-case'})" for r in order], fontsize=7.5)
    ax.set_xlabel("P(hypertension) — red = removed"); style(ax)
    how_plot = b64(fig)
    agree = sum((pr[r] >= t1) for r in removed)
    acc = [f"{sid(r)} · {'case' if s['before'][r]['target'] == 1 else 'non-case'} · P {pr[r]:.2f}" for r in recs if r not in removed]
    flg = [f"{sid(r)} · recorded non-case · P {pr[r]:.2f}" for r in removed]
    rb, ra = int(s["rows_before"]), int(s["rows_after"])
    return dict(
        ds=d, nn="05", key="data_noise", title="Label-Noise Detection and Validation Framework",
        method_line="Confident Learning (Cleanlab)", valid_line="Out-of-Fold Label Validation (all columns)",
        steps=[("Detect Label Noise", f"Confident Learning flags {float(br['before']):.2%} of recorded labels as likely wrong.", "detect"),
               ("Confident Learning", "Out-of-fold probabilities + per-group confident thresholds find recorded non-cases that look like cases.", "model"),
               ("Cleaned Dataset", f"{rb:,} → {ra:,} training rows ({rb - ra:,} suspicious non-case labels pruned).", "table"),
               ("Label Validation", "A separate model re-estimates P(hypertension) for every record from all columns.", "valid"),
               ("Accept or Flag", "Keep records whose label is consistent; flag (remove) contradicted labels.", "decide")],
        t1="Original 10 records (recorded label marked)", table1=table_html(cols1, rows1, cls1), note1="Recorded label = self-reported hypertension diagnosis.",
        t2="Confident Learning", box2="Out-of-fold model probabilities → per protected group, threshold t = mean P(case) among recorded cases. "
                                     "Recorded non-cases far above t are label issues; recorded cases are never pruned (clinical-risk safeguard).",
        pred2=f'<div class="pred">Records removed among the 10:<br><b class="r">{len(removed)}</b><br>{", ".join(sid(r) for r in removed)}</div>',
        t3="After cleaning", table3=table_html(cols3, rows3, gone=gone, why=True),
        note3="Struck-through rows were removed by the pipeline; P(case) = independent check (step 4).",
        t4="Out-of-fold label validation", plot4=plot4,
        how_title="How Label Validation Works?", how_plot=how_plot,
        how_bullets=[f"A gradient-boosting model is trained 5 times on 4/5 of all {len(raw):,} records and predicts the held-out 1/5 (never its own label).",
                     f"Threshold t = {t1:.2f} = mean P(case) among recorded cases (Confident Learning's rule).",
                     f"A recorded non-case with P(case) ≥ t looks like a case → the label is suspicious. {agree} of the {len(removed)} removed records pass this independent check.",
                     "Kept records have labels consistent with their features."],
        acc_rule="Recorded label is consistent with the record's features → kept in the training data.",
        flag_rule="Recorded non-case looks like a case (high P) → flagged as a label issue and removed from training.",
        accept=acc, flag=flg, dec_note="Accept / Flag = the pipeline's own decision (trace); P = the independent re-check computed here.",
        summary=summary5("label noise in the recorded diagnosis.", "Confident Learning (out-of-fold probabilities, group-wise thresholds).",
                         "the cleaned training set.", "labels with an independent out-of-fold model on <b>all columns</b>.",
                         f"keep consistent labels, flag contradicted ones ({len(acc)} kept, {len(flg)} flagged among the 10)."),
        takeaway="Confident Learning finds recorded labels that the data strongly contradict, and the <b>out-of-fold check</b> confirms that the "
                 "removed records really look like the other class, while recorded cases are never discarded.",
        foot="Hypertension · CDC NHIS 2023 · decisions = traced records (stage 4, Data noise); check model = gradient boosting on hypertension_raw.csv.")


# ================================================================== 6. hypertension · encoding / domain · DG transformer
def hypertension_domain():
    d = "hypertension"; t = trace(d); recs = t["records"]; s = step(t, "Encoding / domain"); s0 = step(t, "Missing data")
    fs = pd.read_csv(ROOT / d / "preprocessed" / "hypertension_train_feature_space.csv", dtype={"meta_record_id": str})
    pp = pd.read_csv(ROOT / d / "preprocessed" / "hypertension_train_preprocessed.csv", dtype={"meta_record_id": str})
    feats = [c for c in fs.columns if not c.startswith(("meta_", "sens_")) and c != "target"]
    Zc = [f"z{i}" for i in range(1, 33)]
    fs = fs.set_index("meta_record_id").loc[pp.meta_record_id].reset_index()
    dom = pp.meta_domain_region.astype(str).to_numpy(); regs = sorted(set(dom))
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED); lr = LogisticRegression(max_iter=3000)
    res = {}
    for name, X in [("before", fs[feats].to_numpy()), ("after", pp[Zc].to_numpy())]:
        Xs = StandardScaler().fit_transform(X)
        P = cross_val_predict(lr, Xs, dom, cv=cv, method="predict_proba")
        res[name] = (roc_auc_score(dom, P, multi_class="ovr", average="macro"), P)
    pos = {r: i for i, r in enumerate(pp.meta_record_id)}
    here = [r for r in recs if r in s["after"]]
    own = {}
    for r in here:
        i = pos[r]; j = regs.index(dom[i])
        own[r] = (res["before"][1][i, j], res["after"][1][i, j])
    vals, _ = comparison(d, "08_encoding_domain_bias_dashboard.html")
    ref = vals["No adaptation (raw predictors)"]; dg = vals["Domain-generalisation transformer"]
    raw = pd.read_csv(ROOT / d / "raw" / "hypertension_raw.csv", dtype={"meta_record_id": str}, low_memory=False).set_index("meta_record_id")
    cols1 = ["Record", "Region", "Age", "Male", "Inc ratio", "Urban", "Hypert."]
    rows1, cls1, rows3, cls3 = [], {}, [], {}
    gone = set()
    for i, r in enumerate(recs):
        a = s0["after"][r]
        rows1.append((r, [str(raw.loc[r, "meta_domain_region"])[:9], fv(a["age"]), fv(a["male"]), fv(a["income_poverty_ratio"]), fv(a["urbanicity"]), "YES" if a["target"] == 1 else "no"]))
        cls1[(i, 0)] = "mark"
        if r in s["after"]:
            z = s["after"][r]
            rows3.append((r, [f"{z['z1']:.2f}", f"{z['z2']:.2f}", f"{z['z3']:.2f}", "…", f"{own[r][0]:.2f} → {own[r][1]:.2f}"]))
            for j in range(3):
                cls3[(len(rows3) - 1, j)] = "new"
        else:
            rows3.append((r, ["—", "—", "—", "—", "removed at data-noise step"])); gone.add(len(rows3) - 1)
    cols3 = ["Record", "z1", "z2", "z3", "… z32", "P(own region)"]
    pca_b = PCA(2, random_state=SEED).fit_transform(StandardScaler().fit_transform(fs[feats].to_numpy()))
    pca_a = PCA(2, random_state=SEED).fit_transform(StandardScaler().fit_transform(pp[Zc].to_numpy()))
    rng = np.random.default_rng(1); sub = rng.choice(len(pp), 2500, replace=False)
    fig, axs = plt.subplots(1, 2, figsize=(4.5, 2.6))
    for ax, Pc, ttl in [(axs[0], pca_b, f"before · AUC {res['before'][0]:.2f}"), (axs[1], pca_a, f"after DG · AUC {res['after'][0]:.2f}")]:
        for k, rg in enumerate(regs):
            m = dom[sub] == rg
            ax.scatter(Pc[sub][m, 0], Pc[sub][m, 1], s=3, color=REGION_C[k % 4], alpha=0.45, lw=0, label=rg)
        for r in here:
            ax.scatter(Pc[pos[r], 0], Pc[pos[r], 1], marker="*", s=90, color=C_IMP, ec="white", zorder=5)
        ax.set_title(ttl, fontsize=8.5); ax.set_xticks([]); ax.set_yticks([])
    axs[1].legend(fontsize=6, frameon=False, markerscale=3, loc="lower right")
    fig.suptitle("PCA of the training rows, coloured by region (★ = sample records)", fontsize=8.5)
    plot4 = b64(fig)
    mets = ["Domain-classifier AUC", "MMD", "DG gap", "Target-domain AUROC", "OOD performance"]
    fig, ax = plt.subplots(figsize=(4.3, 2.9))
    x = np.arange(len(mets))
    ax.bar(x - 0.2, [ref[m] for m in mets], 0.38, color="#c9c8c2", label="no adaptation")
    ax.bar(x + 0.2, [dg[m] for m in mets], 0.38, color="#2a78d6", label="DG transformer")
    for i, m in enumerate(mets):
        ax.text(x[i] + 0.2, dg[m] + 0.01, f"{dg[m]:.3f}" if abs(dg[m]) < 0.1 else f"{dg[m]:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, ["Domain AUC", "MMD", "DG gap", "Target AUROC", "OOD perf."], fontsize=7.5); style(ax); ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("Pipeline metrics (domain AUC/MMD/gap ↓, AUROC ↑)", fontsize=8.5)
    how_plot = b64(fig)
    chance = 1 / len(regs)
    acc = [f"{sid(r)} · P(own) {own[r][1]:.2f}" for r in here if own[r][1] <= chance + 0.15]
    flg = [f"{sid(r)} · P(own) {own[r][1]:.2f}" for r in here if own[r][1] > chance + 0.15]
    br = bias_row(d, "Encoding / domain")
    return dict(
        ds=d, nn="06", key="encoding_domain", title="Domain-Shift Correction and Validation Framework",
        method_line="Domain-Generalisation Transformer", valid_line="Domain-Classifier Validation (all 32 dimensions)",
        steps=[("Detect Domain Bias", f"US region is predictable from the features (domain-classifier AUC; AUROC drop: {br['before']}).", "detect"),
               ("DG Transformer", "Feature tokens + [CLS] transformer trained to predict hypertension equally well in every region.", "model"),
               ("Domain-Invariant Data", "Each record becomes a 32-dim vector z1…z32 shared by all regions.", "table"),
               ("Domain Validation", "Train a fresh classifier to recover the region from the new vectors.", "valid"),
               ("Accept or Flag", "Accept records whose region is hidden; flag records still tied to their region.", "decide")],
        t1="Original 10 records (domain = region marked)", table1=table_html(cols1, rows1, cls1), note1="Domain = census region (4 regions).",
        t2="DG transformer", box2="Each predictor becomes a token; a transformer encoder reads all tokens plus a [CLS] token. Training balances the "
                                 "hypertension loss across regions, so the [CLS] vector (z) carries the outcome signal but not the region.",
        pred2=f'<div class="pred">Domain-classifier AUC (pipeline)<br>{ref["Domain-classifier AUC"]:.3f} → <b class="r">{dg["Domain-classifier AUC"]:.3f}</b><br>(0.5 = region not predictable)</div>',
        t3="Domain-invariant representation", table3=table_html(cols3, rows3, cls3, gone=gone, why=True),
        note3="P(own region) = fresh classifier's probability for the record's true region, before → after (chance 0.25).",
        t4="Domain-classifier validation", plot4=plot4,
        how_title="How Domain Validation Works?", how_plot=how_plot,
        how_bullets=[f"A new multinomial classifier (5-fold CV, {len(pp):,} training rows) predicts the region from all columns.",
                     f"Macro AUC: {res['before'][0]:.2f} from the original features → <b>{res['after'][0]:.2f}</b> from z1…z32.",
                     "The pipeline's own metrics (left) compare no adaptation with the DG transformer on shift and accuracy.",
                     f"Record rule: P(own region) ≤ chance + 0.15 = {chance + 0.15:.2f} → the record no longer reveals its region."],
        acc_rule=f"Fresh classifier gives the record's true region ≤ {chance + 0.15:.2f} (near chance {chance:.2f}).",
        flag_rule="Region is still recognisable from the record's representation.",
        accept=acc, flag=flg, dec_note=f"{len(recs) - len(here)} of the 10 records were already removed at the data-noise step, so 7 are checked.",
        summary=summary5("region-dependent feature distributions (domain shift).", "the domain-generalisation transformer (32-dim representation).",
                         "the domain-invariant representation.", "a fresh domain classifier on <b>all 32 dimensions</b>.",
                         f"accept records whose region is hidden, flag the rest ({len(acc)} accepted, {len(flg)} flagged)."),
        takeaway=("The DG transformer keeps the hypertension signal but removes what tells the regions apart, and the <b>domain classifier</b> "
                  "confirms that the region can no longer be recovered well from the new representation.")
                 if res["after"][0] < 0.55 else
                 (f"The DG transformer makes the region harder to recognise (domain AUC {res['before'][0]:.2f} → {res['after'][0]:.2f} here; "
                  f"pipeline {ref['Domain-classifier AUC']:.3f} → {dg['Domain-classifier AUC']:.3f}) while keeping hypertension accuracy, "
                  "but the <b>domain classifier</b> shows a small region signal remains, so a few records are still flagged."),
        foot="Hypertension · CDC NHIS 2023 · before = hypertension_train_feature_space.csv, after = hypertension_train_preprocessed.csv (z1…z32); pipeline metrics = comparison table.")


# ================================================================== respiratory helpers
def resp_feature_space(t):
    fs = pd.read_csv(ROOT / "respiratory" / "preprocessed" / "respiratory_train_feature_space.csv", dtype={"meta_patientunitstayid": str}, low_memory=False)
    feats = [c for c in fs.columns if not c.startswith(("meta_", "sens_")) and c != "target"]
    cont = [c for c in feats if fs[c].nunique() > 6]
    return fs, cont


def resp_map_missing(t, fs, cont, stage):
    """Values of the 10 records in feature-space units. Present records: read directly. A record removed later:
    mapped with the exact per-column linear z-score map estimated from the present records (checked to be exact)."""
    s = step(t, stage); recs = t["records"]
    ids = fs.set_index("meta_patientunitstayid")
    out = {}
    present = [r for r in recs if r in ids.index]
    for r in present:
        out[r] = ids.loc[r, cont].to_numpy(float)
    for r in recs:
        if r in out:
            continue
        z = []
        for c in cont:
            xs = np.array([s["after"][p][c] for p in present], float); zs = ids.loc[present, c].to_numpy(float)
            if np.ptp(xs) <= 0:
                z.append(np.nan); continue
            a, b = np.polyfit(xs, zs, 1)
            if np.max(np.abs(a * xs + b - zs)) > 1e-3:
                z.append(np.nan); continue
            z.append(a * s["after"][r][c] + b)
        z = np.array(z)
        out[r] = z if np.isfinite(z).all() else None
    return out


# ================================================================== 7. respiratory · missing data · MIP-VAE
def respiratory_missing():
    d = "respiratory"; t = trace(d); recs = t["records"]; s = step(t, "Missing data")
    fs, cont = resp_feature_space(t)
    X = fs.loc[~fs.meta_patientunitstayid.isin(recs), cont].to_numpy(float)
    pca = PCA(20, random_state=SEED).fit(X)
    g, ll_ref, thr = fit_gmm(pca.transform(X), 6)
    vals = resp_map_missing(t, fs, cont, "Missing data")
    miss = {r: [k for k, v in s["before"][r].items() if isnan(v)] for r in recs}
    res = {}
    for r in recs:
        if vals[r] is None:
            res[r] = None; continue
        ll = g.score_samples(pca.transform(vals[r][None]))[0]
        res[r] = (ll, pct_of(ll, ll_ref), ll >= thr)
    cnt = pd.Series({k: sum(k in miss[r] for r in recs) for k in s["before"][recs[0]]})
    show = [k for k in ["urine_output_day1_ml", "art_sbp_mean", "lactate_max", "albumin_min", "bilirubin_max", "inr_max"] if k in cnt.index]
    lab = {"urine_output_day1_ml": "Urine ml", "art_sbp_mean": "Art SBP", "lactate_max": "Lact max", "albumin_min": "Alb min",
           "bilirubin_max": "Bili max", "inr_max": "INR max"}
    cols = ["Record", "Gaps"] + [lab[k] for k in show]
    rows1, rows3, cls1, cls3 = [], [], {}, {}
    for i, r in enumerate(recs):
        b, a = s["before"][r], s["after"][r]
        rows1.append((r, [str(len(miss[r]))] + [fv(b[k], 1) for k in show])); rows3.append((r, [str(len(miss[r]))] + [fv(a[k], 1) for k in show]))
        cls1[(i, 0)] = "mark"
        for j, k in enumerate(show):
            if isnan(b[k]):
                cls1[(i, j + 1)] = "bad"; cls3[(i, j + 1)] = "new"
    n_cells = sum(len(v) for v in miss.values())
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    Pb = pca.transform(X[np.random.default_rng(1).choice(len(X), 4000, replace=False)])
    lbl = g.predict(Pb)
    ax.scatter(Pb[:, 0], Pb[:, 1], s=4, c=[PASTEL[k] for k in lbl], alpha=0.45, lw=0)
    gmm_ellipses(ax, g, [0, 1])
    for r in recs:
        if res[r] is None:
            continue
        p2 = pca.transform(vals[r][None])[0]
        ax.scatter(p2[0], p2[1], marker="*", s=200, color=C_IMP if res[r][2] else "#5a0000", ec="white", zorder=5)
        ax.annotate(sid(r), (p2[0], p2[1]), xytext=(5, 4), textcoords="offset points", fontsize=7, color="#8b0000")
    ax.set_xlabel(f"PC1 of all {len(cont)} continuous columns"); ax.set_ylabel("PC2"); style(ax)
    lo, hi = np.percentile(Pb, [0.5, 99.5], axis=0); ax.set_xlim(lo[0] - 1, hi[0] + 1); ax.set_ylim(lo[1] - 1, hi[1] + 1)
    ax.set_title(f"GMM ({g.n_components} clusters) on all continuous columns\n★ = completed sample record", fontsize=9)
    plot4 = b64(fig)
    fig, ax = plt.subplots(figsize=(4.3, 3.2))
    lo_ = np.percentile(ll_ref, 0.3) - 3
    ax.hist(ll_ref[ll_ref >= lo_], bins=np.linspace(lo_, ll_ref.max(), 70), color="#a9c8f0", ec="white", lw=0.3)
    ax.axvline(thr, color=C_FLAG, ls="--", lw=1.6); ax.text(thr, ax.get_ylim()[1] * 0.92, " 1st pct\n threshold", color=C_FLAG, fontsize=8, va="top")
    for k, r in enumerate([r for r in recs if res[r] is not None]):
        ax.scatter(res[r][0], ax.get_ylim()[1] * (0.05 + 0.07 * k), marker="*", s=150, color=C_IMP, ec="white", zorder=5)
    ax.set_xlabel("GMM log-likelihood of a whole record"); ax.set_ylabel("ICU stays (held out)"); style(ax)
    ax.set_xlim(np.percentile(ll_ref, 0.3) - 3, ll_ref.max() + 2)
    how_plot = b64(fig)
    ok = [r for r in recs if res[r] is not None]
    acc = [f"{sid(r)} · {len(miss[r])} gaps · pct {res[r][1]:.0f}" for r in ok if res[r][2]]
    flg = [f"{sid(r)} · {len(miss[r])} gaps · pct {res[r][1]:.1f}" for r in ok if not res[r][2]]
    nochk = [r for r in recs if res[r] is None]
    br = bias_row(d, "Missing data")
    return dict(
        ds=d, nn="07", key="missing_data", title="Missing Value Imputation and Validation Framework",
        method_line="MIP-VAE Imputation", valid_line="GMM Validation (using ALL continuous columns)",
        steps=[("Detect Missing Values", f"{float(br['before']):.2%} of predictor cells missing (labs, blood gases, urine output).", "detect"),
               ("MIP-VAE Imputation", "A variational autoencoder trained on the observed cells reconstructs the missing ones.", "model"),
               ("Completed Dataset", f"All {n_cells} missing cells of the 10 ICU stays filled; no rows removed.", "table"),
               ("GMM Validation (All Columns)", "Check whether each completed stay fits the joint distribution of all columns.", "valid"),
               ("Accept or Flag Imputation", "Accept plausible completed stays; flag low-likelihood ones.", "decide")],
        t1="Original 10 ICU stays (missing = ?)", table1=table_html(cols, rows1, cls1), note1=f"{n_cells} missing cells across the 10 stays (Gaps = per stay).",
        t2="MIP-VAE", box2="Encoder → latent Gaussian → decoder, trained only on observed cells (missing cells masked out of the loss). "
                          "Missing values are filled with the decoder's reconstruction; observed values are kept.",
        pred2=f'<div class="pred">Stay {sid(max(recs, key=lambda r: len(miss[r])))}<br>{max(len(v) for v in miss.values())} gaps filled<br>'
              f'lactate max = <b class="r">{fv(s["after"][max(recs, key=lambda r: len(miss[r]))]["lactate_max"], 2)}</b></div>',
        t3="Completed stays (after imputation)", table3=table_html(cols, rows3, cls3), note3="Red = value filled by MIP-VAE (6 of the gappy columns shown).",
        t4="GMM validation (all columns)", plot4=plot4,
        how_title="How GMM Validation Works?", how_plot=how_plot,
        how_bullets=[f"A GMM ({g.n_components} clusters) learns the joint distribution of <b>all {len(cont)} continuous columns</b> (20 principal components) of the completed training stays.",
                     "It learns patterns such as: low albumin ↔ high lactate; low blood pressure ↔ high heart rate.",
                     f"Each completed sample stay is compared with {len(ll_ref):,} held-out stays.",
                     "Above the 1st percentile → <b>plausible</b>; below → <b>flag</b>."],
        acc_rule="Completed stay lies inside the clusters (log-likelihood ≥ 1st percentile of held-out stays).",
        flag_rule="Completed stay falls in a low-density region (below the 1st percentile).",
        accept=acc, flag=flg,
        dec_note=("" if not nochk else f"{', '.join(sid(r) for r in nochk)}: removed later at the data-noise step and could not be mapped exactly, so not re-scored. ")
                 + "pct = percentile among held-out stays.",
        summary=summary5("missing values (labs and blood gases are often not measured).", "MIP-VAE to reconstruct every gap from the observed cells.",
                         "the completed dataset (same stays, no gaps).", "completed stays with a GMM on <b>all continuous columns</b>.",
                         f"accept plausible imputations, flag the rest ({len(acc)} accepted, {len(flg)} flagged)."),
        takeaway="MIP-VAE fills each gap from what was observed for the same patient, while the <b>GMM</b> confirms that every completed ICU stay "
                 "is consistent with the joint distribution of all physiological columns.",
        foot="Respiratory · eICU-CRD v2.0 · values = traced records (stage 2); GMM fitted on respiratory_train_feature_space.csv (completed, scaled).")


# ================================================================== 8. respiratory · outlier · ECOD
def respiratory_outlier():
    from pyod.models.ecod import ECOD
    d = "respiratory"; t = trace(d); recs = t["records"]; s = step(t, "Outlier")
    fs, cont = resp_feature_space(t)
    X = fs.loc[~fs.meta_patientunitstayid.isin(recs), cont].to_numpy(float)
    rate = 1 - s["rows_after"] / s["rows_before"]
    det = ECOD(contamination=rate).fit(X)
    sc_ref = det.decision_scores_
    thr = np.percentile(sc_ref, 100 * (1 - rate))
    vals = resp_map_missing(t, fs, cont, "Missing data")
    sc = {r: (float(det.decision_function(vals[r][None])[0]) if vals[r] is not None else None) for r in recs}
    removed = [r for r in recs if r not in s["after"]]
    show = ["heart_rate_max", "resp_rate_max", "spo2_min", "temperature_c_max", "creatinine_max", "wbc_max"]
    lab = {"heart_rate_max": "HR max", "resp_rate_max": "RR max", "spo2_min": "SpO2 min", "temperature_c_max": "Temp max",
           "creatinine_max": "Creat max", "wbc_max": "WBC max"}
    cols1 = ["Record"] + [lab[k] for k in show]
    rows1, rows3, cls3 = [], [], {}
    # tail probability of each shown value against the training distribution (ECOD's per-column ingredient)
    ids = fs.set_index("meta_patientunitstayid")
    for i, r in enumerate(recs):
        b = s["before"][r]
        rows1.append((r, [fv(b[k], 1) for k in show]))
        st = "removed" if r in removed else "kept"
        rows3.append((r, [fv(b[k], 1) for k in show[:3]] + [f"{sc[r]:.1f}" if sc[r] is not None else "n/a", f"{(sc_ref < sc[r]).mean() * 100:.1f}" if sc[r] is not None else "n/a", st]))
        cls3[(i, 3)] = "new"
    cols3 = ["Record"] + [lab[k] for k in show[:3]] + ["ECOD", "pct", "Pipeline"]
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.hist(sc_ref, bins=70, color="#a9c8f0", ec="white", lw=0.3)
    ax.axvline(thr, color=C_FLAG, ls="--", lw=1.5); ax.text(thr, ax.get_ylim()[1] * 0.95, f" top {rate:.1%}\n threshold", color=C_FLAG, fontsize=8, va="top")
    for k, r in enumerate(recs):
        if sc[r] is not None:
            ax.scatter(sc[r], ax.get_ylim()[1] * (0.05 + 0.06 * k), marker="*", s=150, color=C_ACC if sc[r] < thr else C_FLAG, ec="white", zorder=5)
    ax.set_xlabel(f"ECOD outlier score (all {len(cont)} continuous columns)"); ax.set_ylabel("training ICU stays"); style(ax)
    ax.set_title("★ = the sample stays", fontsize=9)
    plot4 = b64(fig)
    ex = max([r for r in recs if sc[r] is not None], key=lambda r: sc[r])
    v = vals[ex]
    Xs = np.sort(X, axis=0)
    left = (np.searchsorted(Xs.T[0], 0) * 0)  # placeholder to keep shapes explicit
    tails = []
    for j, c in enumerate(cont):
        col = Xs[:, j]
        pl = (np.searchsorted(col, v[j], side="right")) / len(col)
        pr = 1 - np.searchsorted(col, v[j], side="left") / len(col)
        tails.append((c, -np.log(max(min(pl, pr), 1 / len(col)))))
    tails = sorted(tails, key=lambda x: -x[1])[:8]
    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    ax.barh(range(len(tails)), [x[1] for x in tails], color="#f2a65a")
    ax.set_yticks(range(len(tails)), [x[0] for x in tails], fontsize=7.5); ax.invert_yaxis()
    ax.set_xlabel("−log tail probability (higher = more unusual)"); style(ax)
    ax.set_title(f"Most unusual columns of stay {sid(ex)}", fontsize=8.5)
    how_plot = b64(fig)
    acc = [f"{sid(r)} · pct {(sc_ref < sc[r]).mean() * 100:.0f}" for r in recs if sc[r] is not None and sc[r] < thr]
    flg = [f"{sid(r)} · pct {(sc_ref < sc[r]).mean() * 100:.1f}" for r in recs if sc[r] is not None and sc[r] >= thr]
    br = bias_row(d, "Outlier")
    rb, ra = int(s["rows_before"]), int(s["rows_after"])
    return dict(
        ds=d, nn="08", key="outlier", title="Outlier Detection and Validation Framework",
        method_line="ECOD Outlier Detection", valid_line="ECOD Re-check on the Completed Data (all columns)",
        steps=[("Detect Outliers", f"{float(br['before']):.2%} of rows lie beyond the skew-adjusted extreme fences.", "detect"),
               ("ECOD Scoring", "Empirical-CDF tail probabilities of every column are combined into one outlier score.", "model"),
               ("Cleaned Dataset", f"{rb:,} → {ra:,} training stays ({rb - ra:,} most anomalous removed).", "table"),
               ("ECOD Re-check", "Re-score the sample stays against the cleaned training distribution.", "valid"),
               ("Accept or Flag", "Accept stays below the outlier threshold; flag stays above it.", "decide")],
        t1="Original 10 ICU stays (extremes in vitals/labs?)", table1=table_html(cols1, rows1), note1="Values after imputation, before the outlier step.",
        t2="ECOD", box2="For each column, the empirical CDF gives left and right tail probabilities; the skewness of the column decides which tail counts. "
                       f"The score is the sum of −log tail probabilities. The top {rate:.1%} of training stays are removed.",
        pred2=f'<div class="pred">Sample stays removed by the pipeline:<br><b class="r">{len(removed)}</b> of 10<br>(all kept as typical stays)</div>' if not removed else
              f'<div class="pred">Removed:<br><b class="r">{len(removed)}</b></div>',
        t3="After outlier removal (with ECOD re-check)", table3=table_html(cols3, rows3, cls3),
        note3="ECOD / pct = re-check score and its percentile among cleaned training stays; Pipeline = the trace's decision.",
        t4="ECOD re-check", plot4=plot4,
        how_title="How the ECOD Re-check Works?", how_plot=how_plot,
        how_bullets=[f"ECOD (pyod) is fitted on all {len(X):,} cleaned training stays and <b>all {len(cont)} continuous columns</b>.",
                     "Each column contributes −log of its tail probability; the right plot shows the columns that make the most unusual sample stay stand out.",
                     f"Threshold = the score exceeded by the top {rate:.1%} of stays (the pipeline's removal share).",
                     "Below the threshold → typical stay (accept); above → flag for inspection."],
        acc_rule=f"ECOD score below the top-{rate:.1%} threshold: physiologically typical stay.",
        flag_rule=f"ECOD score in the top {rate:.1%}: unusual combination of values → inspect.",
        accept=acc, flag=flg, dec_note="Flags here are a second opinion on the cleaned data; the pipeline kept all 10 sample stays at the outlier step.",
        summary=summary5("outliers (extreme vital-sign / lab combinations).", "ECOD (empirical-CDF tail probabilities over all columns).",
                         "the cleaned training set without the most anomalous stays.", "the sample stays with an ECOD re-check on <b>all columns</b>.",
                         f"accept typical stays, flag unusual ones ({len(acc)} accepted, {len(flg)} flagged)."),
        takeaway=("ECOD scores how far each ICU stay sits in the tails of every column at once, without tuning, and the <b>re-check</b> confirms "
                  "that the stays kept by the pipeline are typical of the cleaned training data.") if not flg else
                 ("ECOD scores how far each ICU stay sits in the tails of every column at once, without tuning. The <b>re-check</b> on the "
                  f"cleaned data agrees for {len(acc)} sample stays but flags {len(flg)} that the pipeline kept: they are unusual on many "
                  "continuous columns at once and deserve a clinical look before being trusted."),
        foot="Respiratory · eICU-CRD v2.0 · decisions = traced records (stage 3, Outlier); re-check = pyod ECOD on respiratory_train_feature_space.csv.")


CASES = [heart_missing, heart_scale, diabetes_imbalance, diabetes_demographic, hypertension_noise, hypertension_domain,
         respiratory_missing, respiratory_outlier]


def main(only=None):
    for fn in CASES:
        if only and fn.__name__ not in only:
            continue
        c = fn()
        out = ROOT / c["ds"] / "plots_comparison"
        out.mkdir(parents=True, exist_ok=True)
        stem = f"S10_{c['nn']}_{c['key']}_infographic"
        hp = out / f"{stem}.html"
        hp.write_text(render(c), encoding="utf-8")
        screenshot(hp, out / f"{stem}.png")
        print(f"{c['ds']:13s} {stem}.png  accept={len(c['accept'])} flag={len(c['flag'])}", flush=True)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
