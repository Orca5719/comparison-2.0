#!/usr/bin/env python3
"""Load pretrained flower GAN generator. Bypasses torchgan entirely."""

import torch
from torch import nn


class FlowerGANGenerator(nn.Module):
    """DCGAN generator reconstructed from pretrained weights (enc=128 → out=128×128×3)."""

    def __init__(self, encoding_dims=128, step_channels=64, out_channels=3):
        super().__init__()
        # Architecture: 4x4 → 8 → 16 → 32 → 64 → 128
        # channels: encoding_dims → 16*step → 8*step → 4*step → 2*step → step → out_channels
        d0 = step_channels * 16  # 1024
        self.model = nn.Sequential(
            nn.Sequential(
                nn.ConvTranspose2d(encoding_dims, d0, 4, 1, 0, bias=False),
                nn.BatchNorm2d(d0),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(d0, d0 // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d0 // 2),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(d0 // 2, d0 // 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d0 // 4),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(d0 // 4, d0 // 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d0 // 8),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(d0 // 8, d0 // 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d0 // 16),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(d0 // 16, out_channels, 4, 2, 1, bias=True),
                nn.Tanh(),
            ),
        )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.model(z)


def load_flower_gan(model_path='./gan4.model', device='cpu'):
    """Load pretrained Flower GAN generator.

    Returns netG in eval mode on the specified device.
    """
    netG = FlowerGANGenerator()
    ckpt = torch.load(model_path, weights_only=False, map_location=device)
    netG.load_state_dict(ckpt['generator'])
    netG.to(device)
    netG.eval()
    return netG


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    netG = load_flower_gan(device=device)
    z = torch.randn(1, 128).to(device)
    with torch.no_grad():
        img = netG(z)
    print(f'GAN OK — output shape: {img.shape}, range: [{img.min().item():.3f}, {img.max().item():.3f}]')
