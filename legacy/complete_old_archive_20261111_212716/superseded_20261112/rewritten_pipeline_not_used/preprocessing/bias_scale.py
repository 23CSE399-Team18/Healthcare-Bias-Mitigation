"""Feature-scale bias: Quantile transformation, Min-Max scaling, Robust scaling (traditional);
Self-normalizing neural network (SELU), FT-Transformer, PowerTransform Yeo-Johnson (modern).

Transforms act on continuous and ordinal predictors (binary / one-hot stay 0/1).
SNN and FT-Transformer are model architectures: their data-side preprocessing is z-score
standardisation (required by SELU self-normalisation and FT-Transformer's linear numeric
embeddings), so their data metrics are those of z-scoring and their training metrics come
from the architecture itself.
Metrics (training sub-sample; probes trained on it, activations measured on validation):
  Wasserstein distance  mean pairwise 1-D W1 between transformed predictors (scale disparity)
  Correlation preservation  mean |Pearson after - Pearson before|
  Skewness reduction    mean|skew| before - mean|skew| after
  Kurtosis              mean |excess kurtosis| after
  Activation variance   mean over hidden layers of |log(variance of activations)| (0 = unit variance)
  Gradient stability    coefficient of variation of the gradient norm during training
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler
from torch import nn

from common import SEED, select, stratified_sample, train_classifier, wasserstein_1d

FAMILY = {"Quantile transform": "Traditional", "Min-Max scaler": "Traditional", "Robust scaler": "Traditional",
          "SNN (SELU)": "Modern", "FT-Transformer": "Modern", "PowerTransform (Yeo-Johnson)": "Modern"}
NOTES = {"SNN (SELU)": "Self-normalizing network (Klambauer et al. 2017): SELU, LeCun-normal init, alpha dropout; data-side transform = z-score.",
         "FT-Transformer": "rtdl FT-Transformer (2 blocks, d=64); data-side transform = z-score.",
         "Quantile transform": "QuantileTransformer to a normal distribution (1,000 quantiles).",
         "PowerTransform (Yeo-Johnson)": "Yeo-Johnson power transform followed by standardisation."}
DIRECTIONS = {"Wasserstein distance": "min", "Correlation preservation |dr|": "min", "Skewness reduction": "max",
              "Kurtosis": "min", "Activation variance": "min", "Gradient stability": "min"}


def make_transform(name):
    return {"Quantile transform": lambda: QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                                               subsample=200000, random_state=SEED),
            "Min-Max scaler": MinMaxScaler, "Robust scaler": RobustScaler,
            "PowerTransform (Yeo-Johnson)": lambda: PowerTransformer(method="yeo-johnson", standardize=True),
            "SNN (SELU)": StandardScaler, "FT-Transformer": StandardScaler}[name]()


def detect(train, schema):
    X = train.X[schema.scalable]
    sd = X.std()
    sd = sd[sd > 0]
    ratio = float(sd.max() / sd.min()) if len(sd) else 1.0
    sk = float(np.nanmax(np.abs(skew(X.to_numpy(float), axis=0)))) if len(X.columns) else 0.0
    return {"statistic": "max/min predictor SD ratio; max |skewness| (training)", "before": ratio, "max_abs_skew": sk,
            "per_feature_sd": sd.to_dict(), "rule": "present if SD ratio > 10 or max |skewness| > 2",
            "detected": bool(ratio > 10 or sk > 2)}


class SNN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.SELU(), nn.AlphaDropout(0.05), nn.Linear(64, 32), nn.SELU(),
                                 nn.Linear(32, 1))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1 / np.sqrt(m.in_features))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class FTT(nn.Module):
    def __init__(self, d):
        super().__init__()
        from rtdl_revisiting_models import FTTransformer
        self.m = FTTransformer(n_cont_features=d, cat_cardinalities=[], d_out=1, n_blocks=2, d_block=64,
                               attention_n_heads=8, attention_dropout=0.1, ffn_d_hidden=None,
                               ffn_d_hidden_multiplier=4 / 3, ffn_dropout=0.1, residual_dropout=0.0)

    def forward(self, x):
        return self.m(x, None)


def _activation_variance(net, X):
    store = []
    if isinstance(net, FTT):
        layers = list(net.m.backbone.blocks)
    else:
        layers = [m for m in net.modules() if isinstance(m, (nn.ReLU, nn.SELU))]
    hooks = [layer.register_forward_hook(lambda _m, _i, out: store.append(out.detach())) for layer in layers]
    net.eval()
    with torch.no_grad():
        net(torch.from_numpy(X[:2048].astype(np.float32)))
    for h in hooks:
        h.remove()
    return float(np.mean([abs(np.log(max(float(a.var()), 1e-12))) for a in store]))


def probe(name, Xtr, ytr, Xva):
    d = Xtr.shape[1]
    model = SNN(d) if name == "SNN (SELU)" else FTT(d) if name == "FT-Transformer" else None
    net = train_classifier(Xtr, ytr, steps=400, batch=256, record_gradients=True, model=model,
                           hidden=(64, 32))
    g = np.array(net.gradient_norms)
    return _activation_variance(net, Xva), float(g.std() / g.mean()) if g.mean() > 0 else np.nan


def _assemble(frame_X, schema, transformed):
    X = frame_X[schema.columns()].copy()
    X[schema.scalable] = transformed
    return X.to_numpy(float)


def benchmark(frames, schema, log, n_train=10000, n_val=4000):
    train, val = frames["train"], frames["validation"]
    ids = stratified_sample(None, train.meta.target.to_numpy(), n_train, seed=SEED + 5)
    vids = stratified_sample(None, val.meta.target.to_numpy(), n_val, seed=SEED + 6)
    Btr, Bva = train.X.iloc[ids], val.X.iloc[vids]
    raw = Btr[schema.scalable].to_numpy(float)
    corr_before = np.nan_to_num(pd.DataFrame(raw).corr().to_numpy())
    skew_before = np.nanmean(np.abs(skew(raw, axis=0)))
    ytr = train.meta.target.to_numpy()[ids]
    rows, pairs = {}, list(itertools.combinations(range(raw.shape[1]), 2))
    rng = np.random.default_rng(SEED)
    if len(pairs) > 300:
        pairs = [pairs[i] for i in rng.choice(len(pairs), 300, replace=False)]
    for name in FAMILY:
        t0 = time.time()
        tf = make_transform(name).fit(raw)
        after = tf.transform(raw)
        sample = after[rng.choice(len(after), min(2000, len(after)), replace=False)]
        wd = np.mean([wasserstein_1d(sample[:, i], sample[:, j]) for i, j in pairs]) if pairs else 0.0
        corr_after = np.nan_to_num(pd.DataFrame(after).corr().to_numpy())
        Xtr = _assemble(Btr, schema, after)
        Xva = _assemble(Bva, schema, tf.transform(Bva[schema.scalable].to_numpy(float)))
        torch.manual_seed(SEED)
        act, grad = probe(name, Xtr, ytr, Xva)
        rows[name] = {"Wasserstein distance": wd, "Correlation preservation |dr|": float(np.abs(corr_after - corr_before).mean()),
                      "Skewness reduction": float(skew_before - np.nanmean(np.abs(skew(after, axis=0)))),
                      "Kurtosis": float(np.nanmean(np.abs(kurtosis(after, axis=0)))),
                      "Activation variance": act, "Gradient stability": grad}
        log(f"  scale/{name}: W {wd:.3f} act {act:.3f} grad-CV {grad:.3f} ({time.time()-t0:.0f}s)")
    table = pd.DataFrame(rows).T
    table.insert(0, "family", [FAMILY[m] for m in table.index])
    table, best = select(table, DIRECTIONS)
    protocol = (f"{len(ids):,} training rows; {len(schema.scalable)} continuous/ordinal predictors transformed. Probes: "
                "identical 2-hidden-layer ReLU MLP for data transforms, the SNN or FT-Transformer for those methods; "
                f"400 Adam steps; activation variance measured on {len(vids):,} validation rows.")
    return table, best, protocol


def apply(frames, schema, method, log):
    tf = make_transform(method).fit(frames["train"].X[schema.scalable].to_numpy(float))
    new = {}
    for split, frame in frames.items():
        f = frame.copy()
        f.X[schema.scalable] = tf.transform(frame.X[schema.scalable].to_numpy(float))
        new[split] = f
    # ordinal levels move with the transform
    for c in schema.ordinal:
        j = schema.scalable.index(c)
        grid = np.zeros((len(schema.levels[c]), len(schema.scalable)))
        grid[:, j] = schema.levels[c]
        schema.levels[c] = np.sort(tf.transform(grid)[:, j])        # column-wise transforms
    log(f"  {method}: transformed {len(schema.scalable)} predictors in every split (fitted on training)")
    return new, {"transform": tf, "data_side": "z-score" if method in ("SNN (SELU)", "FT-Transformer") else method}
