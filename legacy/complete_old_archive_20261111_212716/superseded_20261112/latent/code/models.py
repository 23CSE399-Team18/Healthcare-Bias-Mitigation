"""Explicit, inductive research implementations used by the latent pipeline."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LATENT_DIM = 8


def mlp(din, dout):
    return nn.Sequential(nn.Linear(din, 64), nn.GELU(), nn.Linear(64, dout))


class MultiViewDAE(nn.Module):
    def __init__(self, dimension, views, groups=None, reference=None):
        super().__init__()
        self.views = views
        self.encoders = nn.ModuleList([mlp(len(v), 16) for v in views])
        self.fuse = nn.Linear(16*len(views), LATENT_DIM)
        self.decoder = mlp(LATENT_DIM, dimension)

    def encode(self, x, corrupt=False):
        h = []
        # A row always retains its first view. Additional view dropout is training-only.
        for i, (v, enc) in enumerate(zip(self.views, self.encoders)):
            a = x[:, v]
            if corrupt:
                a = a * (torch.rand_like(a) > .15)
                if i:
                    a = a * (torch.rand((len(a), 1), device=a.device) > .20)
            h.append(enc(a))
        return self.fuse(torch.cat(h, dim=1))

    def objective(self, x, corrupt=True):
        z = self.encode(x, corrupt=corrupt)
        reconstructed = self.decoder(z)
        return torch.stack([F.mse_loss(reconstructed[:, v], x[:, v]) for v in self.views]).mean()


class DeepGCCA(nn.Module):
    """MAXVAR DGCCA on paired within-record feature views, plus reconstruction.

The correlation objective uses the leading eigenvalues of the sum of regularized
view projection matrices. Its nonzero eigenvalues are computed from the singular
values of the concatenated whitened views. Final projection is fit globally on
training rows, never with a per-test-batch SVD.
"""
    def __init__(self, dimension, views, groups=None, reference=None):
        super().__init__()
        self.views = views
        self.encoders = nn.ModuleList([mlp(len(v), 16) for v in views])
        self.decoders = nn.ModuleList([mlp(16, len(v)) for v in views])

    def hidden(self, x):
        return [enc(x[:, v]) for enc, v in zip(self.encoders, self.views)]

    def objective(self, x, corrupt=True):
        hs = self.hidden(x)
        whitened = []
        for h in hs:
            h = (h - h.mean(0)) / np.sqrt(max(len(h)-1, 1))
            chol = torch.linalg.cholesky(h.T @ h + .01*torch.eye(h.shape[1], device=h.device))
            whitened.append(torch.linalg.solve_triangular(chol, h.T, upper=False).T)
        sv = torch.linalg.svdvals(torch.cat(whitened, dim=1))
        correlation = sv[:LATENT_DIM].square().mean()/len(hs)
        reconstruction = torch.stack([F.mse_loss(dec(h), x[:, v])
            for dec, h, v in zip(self.decoders, hs, self.views)]).mean()
        return reconstruction - .10*correlation


class SaintFixedContext(nn.Module):
    """SAINT-style feature and intersample attention with a frozen train context.

One token per original feature; its one-hot levels and missing flag stay together.
Rows query the fixed training bank, not other query/validation/test rows. This is
an explicitly documented inductive adaptation, not the original batch-context
implementation or a claim to reproduce its benchmark scores.
"""
    def __init__(self, dimension, views, groups, reference):
        super().__init__()
        self.groups = groups
        width = 16
        self.embeddings = nn.ModuleList([nn.Linear(len(g), width) for g in groups])
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Parameter(torch.randn(1, len(groups)+1, width)*.02)
        self.column_attention = nn.MultiheadAttention(width, 2, batch_first=True, dropout=0)
        self.norm1 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width, width))
        self.norm2 = nn.LayerNorm(width)
        flat_dim = (len(groups)+1)*width
        self.q = nn.Linear(flat_dim, 32)
        self.k = nn.Linear(flat_dim, 32)
        self.v = nn.Linear(flat_dim, 32)
        self.row_out = nn.Linear(32, width)
        self.norm3 = nn.LayerNorm(width)
        self.latent = nn.Linear(width, LATENT_DIM)
        self.decoder = mlp(LATENT_DIM, dimension)
        self.register_buffer('reference', torch.as_tensor(reference, dtype=torch.float32))

    def tokens(self, x):
        t = torch.stack([layer(x[:, g]) for g, layer in zip(self.groups, self.embeddings)], dim=1)
        t = torch.cat([self.cls.expand(len(x), -1, -1), t], dim=1) + self.position
        a, _ = self.column_attention(t, t, t, need_weights=False)
        t = self.norm1(t+a)
        return self.norm2(t+self.ff(t))

    def encode(self, x, corrupt=False, cached_context=None):
        if corrupt:
            x = x * (torch.rand_like(x) > .15)
        t = self.tokens(x)
        if cached_context is None:
            r = self.tokens(self.reference).flatten(1)
            k, v = self.k(r), self.v(r)
        else:
            k, v = cached_context
        q = self.q(t.flatten(1))
        weights = torch.softmax(q @ k.T / np.sqrt(32), dim=1)
        return self.latent(self.norm3(t[:, 0]+self.row_out(weights @ v)))

    def context(self):
        r = self.tokens(self.reference).flatten(1)
        return self.k(r), self.v(r)

    def objective(self, x, corrupt=True):
        z = self.encode(x, corrupt=corrupt)
        loss = F.mse_loss(self.decoder(z), x)
        if corrupt:
            z2 = self.encode(x, corrupt=True)
            a, b = F.normalize(z, dim=1), F.normalize(z2, dim=1)
            logits = a @ b.T / .2
            labels = torch.arange(len(x), device=x.device)
            loss = loss + .01*(F.cross_entropy(logits, labels)+F.cross_entropy(logits.T, labels))/2
        return loss


MODELS = {'DGCCA': DeepGCCA, 'SAINT': SaintFixedContext, 'MV_DAE': MultiViewDAE}


@torch.no_grad()
def hidden_all(model, x, batch_size=1024):
    model.eval()
    arrays = [[] for _ in model.views]
    for start in range(0, len(x), batch_size):
        hs = model.hidden(torch.from_numpy(x[start:start+batch_size]))
        for dest, h in zip(arrays, hs):
            dest.append(h.numpy())
    return [np.concatenate(a).astype(np.float64) for a in arrays]


def fit_gcca_projection(model, x):
    hs = hidden_all(model, x)
    means = [h.mean(0) for h in hs]
    hs = [h-m for h, m in zip(hs, means)]
    ridges = [.01*len(x)]*len(hs)
    invs, whitened = [], []
    for h, ridge in zip(hs, ridges):
        cov = h.T @ h + ridge*np.eye(h.shape[1])
        vals, vecs = np.linalg.eigh(cov)
        invroot = (vecs/np.sqrt(vals)) @ vecs.T
        invs.append(np.linalg.inv(cov))
        whitened.append(h @ invroot)
    a = np.concatenate(whitened, axis=1)
    vals, vecs = np.linalg.eigh(a.T @ a)
    ix = np.argsort(vals)[-LATENT_DIM:][::-1]
    g = a @ (vecs[:, ix]/np.sqrt(np.maximum(vals[ix], 1e-12)))
    projections = [inv @ h.T @ g for inv, h in zip(invs, hs)]
    return {'means': means, 'projections': projections, 'eigenvalues': vals[ix], 'fit_rows': len(x)}


@torch.no_grad()
def encode_all(model, x, projection=None, batch_size=512):
    model.eval()
    if isinstance(model, DeepGCCA):
        hs = hidden_all(model, x)
        return sum((h-m)@w for h, m, w in zip(hs, projection['means'], projection['projections']))/len(hs)
    result = []
    context = model.context() if isinstance(model, SaintFixedContext) else None
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[start:start+batch_size])
        z = model.encode(batch, cached_context=context) if isinstance(model, SaintFixedContext) else model.encode(batch)
        result.append(z.numpy())
    return np.concatenate(result).astype(np.float64)
