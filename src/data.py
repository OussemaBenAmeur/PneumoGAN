"""Data loading for PneumoniaMNIST (MedMNIST v3, 64x64 chest X-rays).

Labels: 0 = NORMAL (minority class), 1 = PNEUMONIA (majority class).
The official train/val/test split is used. The GAN only ever sees the
training split, so synthetic images cannot leak test information.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

DATA_ROOT = "data"
CLASS_NAMES = ["normal", "pneumonia"]


def load_split(split, size=64, root=DATA_ROOT):
    """Return (images uint8 [N,H,W], labels int64 [N]) for one split."""
    import medmnist
    from medmnist import PneumoniaMNIST

    ds = PneumoniaMNIST(split=split, download=True, size=size, root=root)
    return ds.imgs.astype(np.uint8), ds.labels.reshape(-1).astype(np.int64)


def subsample_class(imgs, labels, cls, fraction, seed=0):
    """Keep only `fraction` of class `cls` (used to simulate stronger scarcity)."""
    if fraction >= 1.0:
        return imgs, labels
    rng = np.random.default_rng(seed)
    idx_c = np.where(labels == cls)[0]
    keep_c = rng.choice(idx_c, size=max(1, int(round(len(idx_c) * fraction))), replace=False)
    keep = np.sort(np.concatenate([np.where(labels != cls)[0], keep_c]))
    return imgs[keep], labels[keep]


def to_tensor(imgs):
    """uint8 [N,H,W] -> float tensor [N,1,H,W] scaled to [-1, 1] (tanh range)."""
    x = torch.from_numpy(imgs).float().unsqueeze(1) / 255.0
    return x * 2.0 - 1.0


def to_uint8(x):
    """float tensor in [-1,1] [N,1,H,W] -> uint8 numpy [N,H,W]."""
    x = ((x.clamp(-1, 1) + 1.0) * 127.5).round().byte()
    return x.squeeze(1).cpu().numpy()


class TensorImageDataset(Dataset):
    """Images in [-1,1] with optional on-the-fly transform."""

    def __init__(self, x, y, transform=None):
        self.x, self.y, self.transform = x, torch.as_tensor(y), transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = self.x[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y[i]
