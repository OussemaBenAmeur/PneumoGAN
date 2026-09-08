"""Train the conditional DCGAN on the PneumoniaMNIST training split.

Objective (Goodfellow et al., 2014; Eq. 1 in Akpinar et al., 2024):
    min_G max_D  E_x[log D(x|y)] + E_z[log(1 - D(G(z|y)|y))]
In practice G maximises log D(G(z|y)|y) (the "non-saturating" loss proposed in
the original GAN paper), which gives stronger gradients early in training.

Example:
    python src/train_gan.py --out results/gan_full
    python src/train_gan.py --out results/gan_normal20 --minority-fraction 0.2
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

sys.path.insert(0, os.path.dirname(__file__))
from data import load_split, subsample_class, to_tensor  # noqa: E402
from models import Discriminator, Generator, weights_init  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/gan_full")
    p.add_argument("--iters", type=int, default=15000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr-g", type=float, default=2e-4)
    p.add_argument("--lr-d", type=float, default=2e-4)
    p.add_argument("--z-dim", type=int, default=100)
    p.add_argument("--ema", type=float, default=0.999, help="EMA decay for generator weights")
    p.add_argument("--minority-fraction", type=float, default=1.0,
                   help="fraction of NORMAL training images kept (scarcity experiment)")
    p.add_argument("--diffaug", action="store_true",
                   help="apply random translation to real+fake inputs of D (helps small data)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def translate(x, max_shift=4):
    """Differentiable random translation applied identically to real and fake (DiffAugment-style)."""
    n = x.size(0)
    pad = F.pad(x, [max_shift] * 4, mode="reflect")
    dx = torch.randint(0, 2 * max_shift + 1, (n,), device=x.device)
    dy = torch.randint(0, 2 * max_shift + 1, (n,), device=x.device)
    H, W = x.shape[2:]
    rows = (dy[:, None] + torch.arange(H, device=x.device)[None])  # n,H
    cols = (dx[:, None] + torch.arange(W, device=x.device)[None])  # n,W
    idx_n = torch.arange(n, device=x.device)[:, None, None]
    return pad[idx_n, 0, rows[:, :, None], cols[:, None, :]].unsqueeze(1)


def main():
    a = parse()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)

    imgs, labels = load_split("train")
    imgs, labels = subsample_class(imgs, labels, cls=0, fraction=a.minority_fraction, seed=a.seed)
    counts = np.bincount(labels, minlength=2).tolist()
    print(f"training GAN on {len(labels)} images, class counts (normal, pneumonia) = {counts}")

    ds = TensorDataset(to_tensor(imgs), torch.from_numpy(labels))
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True, num_workers=2)

    G = Generator(z_dim=a.z_dim).to(dev); G.apply(weights_init)
    D = Discriminator().to(dev)
    G_ema = copy.deepcopy(G).eval()
    opt_g = torch.optim.Adam(G.parameters(), lr=a.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=a.lr_d, betas=(0.5, 0.999))

    aug = translate if a.diffaug else (lambda t: t)
    fixed_z = torch.randn(64, a.z_dim, device=dev)
    fixed_y = torch.arange(2, device=dev).repeat_interleave(32)  # top half normal, bottom pneumonia

    log, it, t0 = [], 0, time.time()
    while it < a.iters:
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            z = torch.randn(x.size(0), a.z_dim, device=dev)

            # ---- Discriminator: maximise log D(x|y) + log(1 - D(G(z|y)|y)) ----
            fake = G(z, y)
            d_real, d_fake = D(aug(x), y), D(aug(fake.detach()), y)
            loss_d = F.softplus(-d_real).mean() + F.softplus(d_fake).mean()
            opt_d.zero_grad(set_to_none=True); loss_d.backward(); opt_d.step()

            # ---- Generator: non-saturating loss, maximise log D(G(z|y)|y) ----
            loss_g = F.softplus(-D(aug(fake), y)).mean()
            opt_g.zero_grad(set_to_none=True); loss_g.backward(); opt_g.step()

            with torch.no_grad():  # exponential moving average of G's weights
                for pe, p in zip(G_ema.parameters(), G.parameters()):
                    pe.lerp_(p, 1 - a.ema)
                for be, b in zip(G_ema.buffers(), G.buffers()):
                    be.copy_(b)

            if it % 100 == 0:
                rec = dict(iter=it, loss_d=loss_d.item(), loss_g=loss_g.item(),
                           d_real=torch.sigmoid(d_real).mean().item(),
                           d_fake=torch.sigmoid(d_fake).mean().item())
                log.append(rec)
                if it % 1000 == 0:
                    print(f"[{it:6d}/{a.iters}] D={rec['loss_d']:.3f} G={rec['loss_g']:.3f} "
                          f"D(x)={rec['d_real']:.2f} D(G(z))={rec['d_fake']:.2f} "
                          f"({time.time() - t0:.0f}s)", flush=True)
            if it % 2500 == 0 or it == a.iters - 1:
                with torch.no_grad():
                    save_image(G_ema(fixed_z, fixed_y), f"{a.out}/samples_{it:06d}.png",
                               nrow=8, normalize=True, value_range=(-1, 1))
            it += 1
            if it >= a.iters:
                break

    torch.save({"G_ema": G_ema.state_dict(), "G": G.state_dict(), "D": D.state_dict(),
                "args": vars(a), "class_counts": counts}, f"{a.out}/checkpoint.pt")
    with open(f"{a.out}/train_log.json", "w") as f:
        json.dump({"args": vars(a), "class_counts": counts, "log": log,
                   "seconds": time.time() - t0}, f, indent=1)
    print(f"done in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
