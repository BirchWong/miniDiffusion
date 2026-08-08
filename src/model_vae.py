import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.shortcut = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        self.norm1 = nn.GroupNorm(16, out_dim)
        self.norm2 = nn.GroupNorm(16, out_dim)

    def forward(self, x):
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.relu(h + self.shortcut(x))

class DownSample(nn.Module):
    """ kernel_size=4 stride=2 """
    def __init__(self, in_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, in_dim, 4, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)

class UpSample(nn.Module):
    """ kernel_size=4 stride=2 """
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, x):
        return self.up(x)

class VAE(nn.Module):
    def __init__(self, latent_dim=4, base_channels=64):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # 下采样 8×
        self.encoder = nn.Sequential(
            ResBlock(3, base_channels),
            ResBlock(base_channels * 1, base_channels* 1),

            DownSample(base_channels),     # 256->128
            ResBlock(base_channels * 1, base_channels * 2),
            ResBlock(base_channels * 2, base_channels * 2),
            DownSample(base_channels*2),     # 128->64
            ResBlock(base_channels * 2, base_channels * 4),
            ResBlock(base_channels * 4, base_channels * 4),
            DownSample(base_channels*4),  # 64->32
            ResBlock(base_channels * 4, base_channels * 8),
            ResBlock(base_channels * 8, base_channels * 8),
        )

        # latent space 4 * 32 * 32
        self.mu = nn.Conv2d(base_channels * 8, latent_dim, 1)
        self.logvar = nn.Conv2d(base_channels * 8, latent_dim, 1)

        self.decoder = nn.Sequential(
            ResBlock(latent_dim, base_channels * 8),
            ResBlock(base_channels * 8, base_channels * 8),

            UpSample(),     # 64
            ResBlock(base_channels * 8, base_channels * 4),
            ResBlock(base_channels * 4, base_channels * 4),
            UpSample(),     # 128
            ResBlock(base_channels * 4, base_channels * 2),
            ResBlock(base_channels * 2, base_channels * 2),
            UpSample(),     # 256
            ResBlock(base_channels * 2, base_channels * 1),
            ResBlock(base_channels * 1, base_channels * 1),

            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, inference_mode=False):
        mu, logvar = self.encode(x)
        if inference_mode:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
