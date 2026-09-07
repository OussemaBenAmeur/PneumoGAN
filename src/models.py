"""Conditional DCGAN (cGAN + DCGAN) and a small CNN classifier.

Generator  G(z, y): DCGAN transposed-conv stack (Radford et al., 2016).
                    The class label y is embedded and concatenated with the
                    noise z, as in the conditional GAN of Mirza & Osindero (2014).
Discriminator D(x, y): DCGAN conv stack with spectral normalisation for stable
                    training, conditioned via a projection head
                    (Miyato & Koyama, 2018): D(x,y) = psi(phi(x)) + <embed(y), phi(x)>.
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN


class Generator(nn.Module):
    def __init__(self, z_dim=100, n_classes=2, emb_dim=50, ngf=64, img_size=64):
        super().__init__()
        assert img_size == 64, "architecture is written for 64x64 images"
        self.z_dim = z_dim
        self.embed = nn.Embedding(n_classes, emb_dim)
        self.net = nn.Sequential(
            # (z+emb) x 1 x 1 -> 512 x 4 x 4
            nn.ConvTranspose2d(z_dim + emb_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            # -> 256 x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            # -> 128 x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            # -> 64 x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            # -> 1 x 64 x 64
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, y):
        h = torch.cat([z, self.embed(y)], dim=1)[:, :, None, None]
        return self.net(h)


class Discriminator(nn.Module):
    def __init__(self, n_classes=2, ndf=64):
        super().__init__()
        self.features = nn.Sequential(
            SN(nn.Conv2d(1, ndf, 4, 2, 1)), nn.LeakyReLU(0.2, True),            # 32
            SN(nn.Conv2d(ndf, ndf * 2, 4, 2, 1)), nn.LeakyReLU(0.2, True),      # 16
            SN(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)), nn.LeakyReLU(0.2, True),  # 8
            SN(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)), nn.LeakyReLU(0.2, True),  # 4
        )
        feat_dim = ndf * 8
        self.linear = SN(nn.Linear(feat_dim, 1))
        self.embed = SN(nn.Embedding(n_classes, feat_dim))

    def forward(self, x, y):
        h = self.features(x).sum(dim=[2, 3])          # global sum pooling
        out = self.linear(h).squeeze(1)
        out = out + (self.embed(y) * h).sum(dim=1)    # projection conditioning
        return out                                    # logit: real vs fake


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class SmallCNN(nn.Module):
    """Downstream diagnostic classifier (the 'indirect' evaluation model)."""

    def __init__(self, n_classes=2):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(256, n_classes))

    def embed(self, x):
        return self.features(x).mean(dim=[2, 3])

    def forward(self, x):
        return self.head(self.features(x))
