"""Stage 2 over sentence log-probs (16 Oct): post-processing / document-level sequence models fitted on out-of-fold
(OOF) CV predictions, scored cross-fold (fit on the OOF predictions of one CV half's docs, score on the other half's,
pool both directions), so no label of a scored doc is used to fit what scores it.

  python stage2.py --feats ../runs_p3/cv/ctx_win1024:17 ../runs_p3/cv/lorma_mlp_la:7 --methods argmax bias viterbi gru

Feature spec <cv run prefix>:<epoch> reads <prefix>_f0/_f1/dev_logp_ep<epoch>.npz (OOF for every labelled doc);
several specs = their mean probability (ensemble) for argmax/bias/viterbi, concatenated log-probs for the GRU.
Methods:
  argmax  - the base prediction
  bias    - per-label additive log-prob bias, coordinate ascent on macro-F1 of the fit half
  viterbi - first-order label chain: log p(y_t|y_t-1) from the fit half's gold sequences, weight lam (grid on the fit half)
  gru     - 2-layer BiGRU over the doc's sentence sequence: input = OOF log-probs (+ relative position), CE loss
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

import rr_data as D

K = len(D.LABELS)


def load_feats(specs, kind="dev", test_ep=None):
    """-> {doc_id: [ (n, K) logp per spec ]}. kind='dev': OOF (both halves); kind='test': mean over the two halves."""
    out = {}
    for spec in specs:
        prefix, ep = spec.rsplit(":", 1)
        if kind == "dev":
            z = {}
            for f in (0, 1):
                zz = np.load(f"{prefix}_f{f}/dev_logp_ep{ep}.npz")
                z.update({k: zz[k].astype(np.float32) for k in zz.files})
        else:
            zs = [np.load(f"{prefix}_f{f}/test_logp_ep{ep}.npz") for f in (0, 1)]
            z = {k: np.log(np.mean([np.exp(x[k].astype(np.float32)) for x in zs], 0) + 1e-9) for k in zs[0].files}
        for k, v in z.items():
            out.setdefault(k, []).append(v)
    return out


def mean_logp(fs):
    return np.log(np.mean([np.exp(f) for f in fs], 0) + 1e-9)


def f1(ys, ps):
    return D.macro_scores(np.concatenate(ys), np.concatenate(ps))[0]


def fit_bias(lps, ys, iters=3, grid=np.linspace(-3, 3, 25)):
    b = np.zeros(K, dtype=np.float32)
    best = f1(ys, [lp.argmax(-1) for lp in lps])
    for _ in range(iters):
        for k in range(K):
            for g in grid:
                bb = b.copy()
                bb[k] = g
                s = f1(ys, [(lp + bb).argmax(-1) for lp in lps])
                if s > best + 1e-6:
                    best, b = s, bb
    return b


def trans_matrix(ys, smooth=1.0):
    c = np.full((K, K), smooth)
    s = np.full(K, smooth)
    for y in ys:
        for a, b in zip(y[:-1], y[1:]):
            c[a, b] += 1
        s[y[0]] += 1
    return np.log(c / c.sum(1, keepdims=True)), np.log(s / s.sum())


def viterbi(lp, T, S, lam):
    n = len(lp)
    score = lp[0] + lam * S
    back = np.zeros((n, K), dtype=np.int64)
    for t in range(1, n):
        cand = score[:, None] + lam * T
        back[t] = cand.argmax(0)
        score = cand.max(0) + lp[t]
    path = [int(score.argmax())]
    for t in range(n - 1, 0, -1):
        path.append(int(back[t][path[-1]]))
    return np.array(path[::-1])


class GRU(nn.Module):
    def __init__(self, d_in, h=128, dropout=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Dropout(dropout))
        self.rnn = nn.GRU(h, h, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.out = nn.Linear(2 * h, K)
        nn.init.zeros_(self.out.weight)            # residual: start from the mean input log-prob, learn a correction
        nn.init.zeros_(self.out.bias)
        self.drop = nn.Dropout(dropout)
        self.n_spec = (d_in - 2) // K

    def forward(self, x):
        base = torch.logsumexp(x[..., :self.n_spec * K].unflatten(-1, (self.n_spec, K)), -2)
        return base + self.out(self.drop(self.rnn(self.inp(x))[0]))


def gru_input(fs):
    x = np.concatenate(fs, -1)
    n = len(x)
    pos = (np.arange(n) / max(1, n - 1))[:, None]
    return torch.tensor(np.concatenate([x, pos, np.full((n, 1), np.log(n) / 8)], -1), dtype=torch.float32)


DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pad(X, Y=None):
    n = max(len(x) for x in X)
    xb = torch.zeros(len(X), n, X[0].shape[1])
    yb = torch.full((len(X), n), -100, dtype=torch.long)
    for i, x in enumerate(X):
        xb[i, :len(x)] = x
        if Y is not None:
            yb[i, :len(x)] = torch.as_tensor(Y[i])
    return xb.to(DEV), yb.to(DEV)


def fit_gru(X, Y, epochs=40, seed=0, la_tau=0.0, lr=2e-3, bs=4):
    """Padded mini-batches of docs; the bidirectional GRU sees padding at the end of short docs (zeros, masked loss)."""
    torch.manual_seed(seed)
    m = GRU(X[0].shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    cnt = np.bincount(np.concatenate(Y), minlength=K) + 1
    off = torch.tensor(la_tau * np.log(cnt / cnt.sum()), dtype=torch.float32, device=DEV)
    order = np.argsort([len(x) for x in X])
    groups = [order[i:i + bs] for i in range(0, len(order), bs)]
    data = [pad([X[i] for i in g], [Y[i] for i in g]) for g in groups]
    m.train()
    for ep in range(epochs):
        for j in np.random.RandomState(seed + ep).permutation(len(data)):
            xb, yb = data[j]
            logits = m(xb) + off
            loss = nn.functional.cross_entropy(logits.reshape(-1, K), yb.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
    m.eval()
    return m


def gru_logp(models, X):
    with torch.no_grad():
        return [torch.stack([m(x[None].to(DEV))[0].log_softmax(-1) for m in models]).exp().mean(0).log().cpu().numpy()
                for x in X]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=["argmax", "bias", "viterbi", "gru"])
    ap.add_argument("--gru_seeds", type=int, default=3)
    ap.add_argument("--gru_epochs", type=int, default=40)
    ap.add_argument("--la_tau", type=float, default=0.0)
    ap.add_argument("--out", default=None, help="json with the results")
    args = ap.parse_args()
    torch.set_num_threads(8)
    feats = load_feats(args.feats)
    gold = {d["id"]: np.array([D.LAB2ID[l] for l in d["labels"]])
            for dom in D.DOMAINS for p in ("train", "dev") for d in D.load_split(f"{dom}_{p}")}
    folds = D.cv_folds(2)
    res = {}
    preds = {m: {} for m in args.methods}
    for a in (0, 1):                                   # fit on fold a's docs, predict fold 1 - a's docs
        fit, ev = folds[a], folds[1 - a]
        lp_fit = [mean_logp(feats[k]) for k in fit]
        y_fit = [gold[k] for k in fit]
        lp_ev = [mean_logp(feats[k]) for k in ev]
        if "argmax" in args.methods:
            preds["argmax"].update({k: lp.argmax(-1) for k, lp in zip(ev, lp_ev)})
        if "bias" in args.methods:
            b = fit_bias(lp_fit, y_fit)
            res[f"bias_fit{a}"] = b.round(2).tolist()
            preds["bias"].update({k: (lp + b).argmax(-1) for k, lp in zip(ev, lp_ev)})
        if "viterbi" in args.methods:
            T, S = trans_matrix(y_fit)
            lams = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
            sc = [f1(y_fit, [viterbi(lp, T, S, l) for lp in lp_fit]) for l in lams]
            lam = lams[int(np.argmax(sc))]
            res[f"viterbi_lam_fit{a}"] = lam
            preds["viterbi"].update({k: viterbi(lp, T, S, lam) for k, lp in zip(ev, lp_ev)})
        if "gru" in args.methods:
            Xf = [gru_input(feats[k]) for k in fit]
            Xe = [gru_input(feats[k]) for k in ev]
            ms = [fit_gru(Xf, y_fit, args.gru_epochs, s, args.la_tau) for s in range(args.gru_seeds)]
            preds["gru"].update({k: lp.argmax(-1) for k, lp in zip(ev, gru_logp(ms, Xe))})
    ids = sorted(gold)
    y = np.concatenate([gold[k] for k in ids])
    for m in args.methods:
        p = np.concatenate([preds[m][k] for k in ids])
        f, pr, rc, per = D.macro_scores(y, p)
        res[m] = {"f1": round(f, 4), "acc": round(float((p == y).mean()), 4),
                  "per_label": {l: round(v[2], 3) for l, v in per.items()}}
        print(f"{m:8s} cross-fold macro-F1 {f:.4f}  acc {(p == y).mean():.3f}  "
              + " ".join(f"{l[:6]}={v[2]:.2f}" for l, v in per.items()), flush=True)
    if args.out:
        json.dump({"feats": args.feats, **res}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
