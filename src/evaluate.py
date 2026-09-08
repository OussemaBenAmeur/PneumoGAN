"""Evaluate a trained conditional DCGAN, following the evaluation taxonomy of
Makhlouf et al. (2023, Table 3):

  DIRECT quantitative   -> FID (Inception features) and a domain-specific FID
                           computed with a chest-X-ray classifier's features,
                           each reported next to a real-vs-real reference value.
  INDIRECT quantitative -> does adding synthetic images improve a downstream
                           pneumonia classifier? (the most common evaluation
                           method in the reviewed studies), plus
                           Train-on-Synthetic / Test-on-Real (TSTR).
  QUALITATIVE           -> side-by-side real vs. synthetic image grids
                           (for visual inspection / a blinded reader test).

Example:
    python src/evaluate.py --gan results/gan_full --out results/eval_full
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from data import CLASS_NAMES, load_split, subsample_class, to_tensor, to_uint8  # noqa: E402
from models import Generator, SmallCNN  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--gan", default="results/gan_full", help="folder with checkpoint.pt")
    p.add_argument("--out", default="results/eval_full")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--n-fid", type=int, default=2000, help="synthetic images per class for FID")
    p.add_argument("--skip-inception", action="store_true")
    return p.parse_args()


# ----------------------------------------------------------------- generation
@torch.no_grad()
def generate(G, cls, n, z_dim, bs=500):
    out = []
    for i in range(0, n, bs):
        m = min(bs, n - i)
        z = torch.randn(m, z_dim, device=DEV)
        y = torch.full((m,), cls, dtype=torch.long, device=DEV)
        out.append(G(z, y).cpu())
    return torch.cat(out) if out else torch.empty(0, 1, 64, 64)


# ----------------------------------------------------------------- GPU augmentation
def random_affine(x, deg=10, trans=0.08, scale=(0.9, 1.1)):
    """Classical geometric augmentation, applied per-sample on the GPU."""
    n = x.size(0)
    ang = (torch.rand(n, device=x.device) * 2 - 1) * math.radians(deg)
    s = torch.empty(n, device=x.device).uniform_(*scale)
    t = (torch.rand(n, 2, device=x.device) * 2 - 1) * trans * 2
    cos, sin = torch.cos(ang) / s, torch.sin(ang) / s
    theta = torch.stack([torch.stack([cos, -sin, t[:, 0]], 1),
                         torch.stack([sin, cos, t[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, padding_mode="border", align_corners=False)


# ----------------------------------------------------------------- classifier
def train_classifier(x, y, xval, yval, epochs, seed, classical_aug=False, bs=128):
    torch.manual_seed(seed); np.random.seed(seed)
    model = SmallCNN().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * math.ceil(len(x) / bs))
    x, y = x.to(DEV), torch.as_tensor(y).to(DEV)
    best_auc, best_state = -1, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(x), device=DEV)
        for i in range(0, len(x), bs):
            idx = perm[i:i + bs]
            xb = x[idx]
            if classical_aug:
                xb = random_affine(xb)
            loss = F.cross_entropy(model(xb), y[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        auc = predict_metrics(model, xval, yval)["auc"]
        if auc > best_auc:  # model selection on the validation split only
            best_auc = auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_proba(model, x, bs=512):
    model.eval()
    return torch.cat([F.softmax(model(x[i:i + bs].to(DEV)), 1)[:, 1].cpu()
                      for i in range(0, len(x), bs)]).numpy()


def predict_metrics(model, x, y):
    p = predict_proba(model, x)
    pred = (p >= 0.5).astype(int)
    return dict(
        auc=roc_auc_score(y, p),
        accuracy=accuracy_score(y, pred),
        balanced_accuracy=balanced_accuracy_score(y, pred),
        f1_macro=f1_score(y, pred, average="macro"),
        recall_normal=float(((pred == 0) & (y == 0)).sum() / max(1, (y == 0).sum())),
        recall_pneumonia=float(((pred == 1) & (y == 1)).sum() / max(1, (y == 1).sum())),
    )


# ----------------------------------------------------------------- FID
def frechet(f1, f2):
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)
    if isinstance(covmean, tuple):  # older SciPy versions
        covmean = covmean[0]
    covmean = np.real(covmean)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


class InceptionFeatures:
    """2048-d pool features of torchvision's ImageNet Inception-v3.
    (Numbers are comparable within this project, not exactly with the TF reference FID.)"""

    def __init__(self):
        from torchvision.models import Inception_V3_Weights, inception_v3
        m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        m.fc = torch.nn.Identity()
        self.m = m.eval().to(DEV)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=DEV)[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=DEV)[None, :, None, None]

    @torch.no_grad()
    def __call__(self, x, bs=100):
        feats = []
        for i in range(0, len(x), bs):
            xb = (x[i:i + bs].to(DEV) + 1) / 2
            xb = F.interpolate(xb, size=(299, 299), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
            feats.append(self.m((xb - self.mean) / self.std).cpu())
        return torch.cat(feats).numpy().astype(np.float64)


@torch.no_grad()
def domain_features(model, x, bs=512):
    model.eval()
    return torch.cat([model.embed(x[i:i + bs].to(DEV)).cpu() for i in range(0, len(x), bs)]).numpy().astype(np.float64)


# ----------------------------------------------------------------- figures
def save_grids(real_by_cls, fake_by_cls, path, n=16):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(n * 0.9, 4 * 1.15))
    rows = [("Real – normal", real_by_cls[0]), ("Synthetic – normal", fake_by_cls[0]),
            ("Real – pneumonia", real_by_cls[1]), ("Synthetic – pneumonia", fake_by_cls[1])]
    for ax, (title, imgs) in zip(axes, rows):
        strip = np.concatenate([to_uint8(imgs[i:i + 1])[0] for i in range(n)], axis=1)
        ax.imshow(strip, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, loc="left", fontsize=10); ax.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)


def save_loss_curve(log_path, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    log = json.load(open(log_path))["log"]
    it = [r["iter"] for r in log]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.2))
    a1.plot(it, [r["loss_d"] for r in log], label="D loss"); a1.plot(it, [r["loss_g"] for r in log], label="G loss")
    a1.set_xlabel("iteration"); a1.legend(); a1.set_title("Adversarial losses")
    a2.plot(it, [r["d_real"] for r in log], label="D(x) real"); a2.plot(it, [r["d_fake"] for r in log], label="D(G(z)) fake")
    a2.axhline(0.5, ls="--", c="gray", lw=0.8); a2.set_xlabel("iteration"); a2.legend(); a2.set_title("Discriminator outputs")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)


# ----------------------------------------------------------------- main
def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    ck = torch.load(f"{a.gan}/checkpoint.pt", map_location=DEV)
    gargs = ck["args"]
    G = Generator(z_dim=gargs["z_dim"]).to(DEV)
    G.load_state_dict(ck["G_ema"]); G.eval()

    # Same (possibly subsampled) training set the GAN saw
    tr_i, tr_y = load_split("train")
    tr_i, tr_y = subsample_class(tr_i, tr_y, 0, gargs["minority_fraction"], seed=gargs["seed"])
    va_i, va_y = load_split("val")
    te_i, te_y = load_split("test")
    xtr, xva, xte = to_tensor(tr_i), to_tensor(va_i), to_tensor(te_i)
    counts = np.bincount(tr_y, minlength=2)
    n_min, n_maj = counts[0], counts[1]
    print(f"train counts normal={n_min} pneumonia={n_maj}; val={len(va_y)} test={len(te_y)}")

    torch.manual_seed(1234)
    fake_by_cls = {c: generate(G, c, a.n_fid, gargs["z_dim"]) for c in (0, 1)}
    real_tr_by_cls = {c: xtr[tr_y == c] for c in (0, 1)}
    real_te_by_cls = {c: xte[te_y == c] for c in (0, 1)}

    # ---------- Qualitative
    save_grids(real_tr_by_cls, fake_by_cls, f"{a.out}/real_vs_synthetic.png")
    if os.path.exists(f"{a.gan}/train_log.json"):
        save_loss_curve(f"{a.gan}/train_log.json", f"{a.out}/gan_training_curves.png")

    results = {"gan": a.gan, "gan_args": gargs, "train_counts": counts.tolist()}

    # ---------- Indirect evaluation: downstream classifier
    def make_set(strategy):
        if strategy in ("baseline", "classical_aug"):
            return xtr, tr_y
        if strategy == "oversample":
            extra = np.random.default_rng(0).choice(np.where(tr_y == 0)[0], n_maj - n_min, replace=True)
            return torch.cat([xtr, xtr[extra]]), np.concatenate([tr_y, tr_y[extra]])
        if strategy in ("gan_balance", "gan_balance+classical_aug"):
            syn = generate(G, 0, n_maj - n_min, gargs["z_dim"])
            return torch.cat([xtr, syn]), np.concatenate([tr_y, np.zeros(len(syn), dtype=np.int64)])
        if strategy == "gan_double":  # balance, then add the same amount again of both classes
            syn0 = generate(G, 0, (n_maj - n_min) + n_maj, gargs["z_dim"])
            syn1 = generate(G, 1, n_maj, gargs["z_dim"])
            return (torch.cat([xtr, syn0, syn1]),
                    np.concatenate([tr_y, np.zeros(len(syn0), dtype=np.int64), np.ones(len(syn1), dtype=np.int64)]))
        if strategy == "synthetic_only_TSTR":  # train on synthetic, test on real
            syn0 = generate(G, 0, len(tr_y) // 2, gargs["z_dim"])
            syn1 = generate(G, 1, len(tr_y) // 2, gargs["z_dim"])
            return torch.cat([syn0, syn1]), np.array([0] * len(syn0) + [1] * len(syn1))
        raise ValueError(strategy)

    strategies = ["baseline", "classical_aug", "oversample", "gan_balance",
                  "gan_balance+classical_aug", "gan_double", "synthetic_only_TSTR"]
    clf = {}
    ref_model = None
    for s in strategies:
        runs = []
        for seed in range(a.seeds):
            torch.manual_seed(100 + seed)
            x, y = make_set(s)
            m = train_classifier(x, y, xva, va_y, a.epochs, seed, classical_aug="classical_aug" in s)
            r = predict_metrics(m, xte, te_y); r["n_train"] = len(y)
            runs.append(r)
            if s == "baseline" and seed == 0:
                ref_model = m  # reused as the feature extractor for the domain FID
            print(f"  {s:28s} seed={seed} AUC={r['auc']:.4f} bal.acc={r['balanced_accuracy']:.4f} "
                  f"recall_normal={r['recall_normal']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        keys = [k for k in runs[0] if k != "n_train"]
        clf[s] = {"n_train": runs[0]["n_train"], "runs": runs,
                  "mean": {k: float(np.mean([r[k] for r in runs])) for k in keys},
                  "std": {k: float(np.std([r[k] for r in runs])) for k in keys}}
    results["classifier"] = clf

    # ---------- Direct evaluation: FID (synthetic vs real test) with a real-vs-real reference
    fid = {}
    extractors = [("domain_fid", lambda x: domain_features(ref_model, x))]
    if not a.skip_inception:
        try:
            inc = InceptionFeatures()
            extractors.append(("inception_fid", inc))
        except Exception as e:  # e.g. no internet for the weights
            print("Inception FID skipped:", e)
    for name, fx in extractors:
        fid[name] = {}
        for c in (0, 1):
            f_test = fx(real_te_by_cls[c])
            f_fake = fx(fake_by_cls[c])
            f_train = fx(real_tr_by_cls[c][: a.n_fid])
            fid[name][CLASS_NAMES[c]] = {
                "synthetic_vs_real_test": frechet(f_fake, f_test),
                "real_train_vs_real_test (reference)": frechet(f_train, f_test),
            }
        print(name, json.dumps(fid[name], indent=1))
    results["fid"] = fid
    results["seconds"] = time.time() - t0

    with open(f"{a.out}/results.json", "w") as f:
        json.dump(results, f, indent=1)
    write_markdown(results, f"{a.out}/results.md")
    print(open(f"{a.out}/results.md").read())


def write_markdown(r, path):
    L = [f"# Results for `{r['gan']}`", "",
         f"Training set (normal, pneumonia): {r['train_counts']}", "",
         "## Indirect evaluation – downstream classifier on the real test set (mean ± std over seeds)", "",
         "| Strategy | # train | AUC | Accuracy | Balanced acc. | Macro-F1 | Recall normal | Recall pneumonia |",
         "|---|---|---|---|---|---|---|---|"]
    for s, v in r["classifier"].items():
        m, sd = v["mean"], v["std"]
        cell = lambda k: f"{m[k]:.3f} ± {sd[k]:.3f}"
        L.append(f"| {s} | {v['n_train']} | {cell('auc')} | {cell('accuracy')} | {cell('balanced_accuracy')} | "
                 f"{cell('f1_macro')} | {cell('recall_normal')} | {cell('recall_pneumonia')} |")
    L += ["", "## Direct evaluation – Fréchet distances (lower is better)", "",
          "| Feature space | Class | Synthetic vs real test | Real train vs real test (reference) |", "|---|---|---|---|"]
    for name, d in r["fid"].items():
        for c, v in d.items():
            L.append(f"| {name} | {c} | {v['synthetic_vs_real_test']:.2f} | {v['real_train_vs_real_test (reference)']:.2f} |")
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
