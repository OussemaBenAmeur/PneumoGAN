#!/usr/bin/env python3
"""French versions of the figures for the slides (same GAN checkpoint, same data)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import load_split, to_tensor, to_uint8  # noqa: E402
from models import Generator  # noqa: E402

OUT = ROOT / "presentation" / "fig"
OUT.mkdir(exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load(ROOT / "results/gan_full/checkpoint.pt", map_location=dev)
G = Generator(z_dim=ck["args"]["z_dim"]).to(dev)
G.load_state_dict(ck["G_ema"]); G.eval()
imgs, labels = load_split("train", root=str(ROOT / "data"))
x = to_tensor(imgs)
torch.manual_seed(1234)
n = 16
with torch.no_grad():
    fake = {c: G(torch.randn(n, G.z_dim, device=dev), torch.full((n,), c, device=dev)).cpu() for c in (0, 1)}
rows = [("Réelles : normal", x[labels == 0][:n]), ("Synthétiques (GAN) : normal", fake[0]),
        ("Réelles : pneumonie", x[labels == 1][:n]), ("Synthétiques (GAN) : pneumonie", fake[1])]
fig, axes = plt.subplots(4, 1, figsize=(n * 0.9, 4 * 1.15))
for ax, (title, im) in zip(axes, rows):
    ax.imshow(np.concatenate(list(to_uint8(im)), axis=1), cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold"); ax.axis("off")
plt.tight_layout(); plt.savefig(OUT / "reel_vs_synthetique.png", dpi=150); plt.close(fig)

log = json.load(open(ROOT / "results/gan_full/train_log.json"))["log"]
it = [r["iter"] for r in log]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.2))
a1.plot(it, [r["loss_d"] for r in log], label="perte D"); a1.plot(it, [r["loss_g"] for r in log], label="perte G")
a1.set_xlabel("itération"); a1.legend(); a1.set_title("Pertes antagonistes")
a2.plot(it, [r["d_real"] for r in log], label="D(x) réel"); a2.plot(it, [r["d_fake"] for r in log], label="D(G(z)) faux")
a2.axhline(0.5, ls="--", c="gray", lw=0.8); a2.set_xlabel("itération"); a2.legend(); a2.set_title("Sorties du discriminateur")
plt.tight_layout(); plt.savefig(OUT / "courbes_entrainement.png", dpi=150); plt.close(fig)
print("figures ->", OUT)
