"""TnT Artifact Generator for BagNet + CIFAR-10.

Random search in GAN latent space: sample N latent vectors, generate flower
patches, blend onto CIFAR-10 images, pick the one with highest ASR as UAP.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import cv2
import os
import sys
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'PatchGuard'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'PatchGuard', 'nets'))

from gan import load_flower_gan
from utils import bg_remove_threshold
import nets.bagnet

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
CIFAR_STD = torch.tensor([0.2023, 0.1994, 0.2010])
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def unnormalize_cifar(x):
    """x: (B,3,H,W) normalized with CIFAR stats -> [0,1]"""
    mean = CIFAR_MEAN.to(x.device)[None, :, None, None]
    std = CIFAR_STD.to(x.device)[None, :, None, None]
    return x * std + mean


def normalize_cifar(x):
    """x: (B,3,H,W) in [0,1] -> normalized with CIFAR stats"""
    mean = CIFAR_MEAN.to(x.device)[None, :, None, None]
    std = CIFAR_STD.to(x.device)[None, :, None, None]
    return (x - mean) / std


def blend_cifar(images, flower, patch_size, corner=192):
    """
    images: (B,3,192,192) normalized with CIFAR stats
    flower: (1,3,ps,ps) in [-1, 1] (GAN output, resized)
    Returns: (B,3,192,192) normalized with CIFAR stats
    """
    B = images.shape[0]
    ps = patch_size

    images_01 = unnormalize_cifar(images)
    flower_01 = flower * 0.5 + 0.5

    flower_np = flower_01.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    flower_np = np.uint8(np.clip(flower_np * 255, 0, 255))
    mask = bg_remove_threshold(flower_np, MODEL='default')

    mask_t = torch.from_numpy(mask).float().to(DEVICE)
    mask_t = mask_t.unsqueeze(0).repeat(3, 1, 1)

    x0 = corner - ps
    y0 = corner - ps

    for j in range(B):
        patch_region = images_01[j, :, x0:x0 + ps, y0:y0 + ps]
        images_01[j, :, x0:x0 + ps, y0:y0 + ps] = (
            patch_region * (1 - mask_t) + flower_01.squeeze(0) * mask_t
        )

    images_01 = torch.clamp(images_01, 0, 1)
    return normalize_cifar(images_01)


def evaluate_asr(model, dataloader, flower, patch_size):
    """Compute attack success rate of flower patch on model."""
    model.eval()
    total, success = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            adv_images = blend_cifar(images, flower, patch_size)
            outputs = model(adv_images)
            if outputs.dim() == 4:
                outputs = outputs.mean(dim=(1, 2))
            preds = outputs.argmax(dim=1)
            success += (preds != labels).sum().item()
            total += labels.size(0)
    return success / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, required=True, choices=[16, 32])
    parser.add_argument('--n_search', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_eval', type=int, default=500)
    parser.add_argument('--data_dir', default='../PatchGuard/data/cifar')
    parser.add_argument('--output_dir', default='artifacts')
    parser.add_argument('--gan_path', default='./gan4.model')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(192),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    val_dataset = datasets.CIFAR10(root=args.data_dir, train=False,
                                   download=True, transform=transform)
    if args.n_eval < len(val_dataset):
        val_dataset = Subset(val_dataset, range(args.n_eval))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = nets.bagnet.bagnet17(pretrained=True, aggregation='mean')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model = nn.DataParallel(model)
    ckpt = torch.load('../PatchGuard/checkpoints/bagnet17_192_cifar.pth',
                      weights_only=False, map_location=DEVICE)
    model.load_state_dict(ckpt['net'])
    model = model.to(DEVICE)
    model.eval()

    netG = load_flower_gan(args.gan_path, device=DEVICE)

    best_asr = 0.0
    best_z = None
    best_flower = None

    print(f'Searching {args.n_search} latent vectors for ps={args.patch_size}...')
    for i in tqdm(range(args.n_search)):
        z = torch.randn(1, 128).to(DEVICE)
        with torch.no_grad():
            flower_128 = netG(z)
            flower = F.interpolate(flower_128, size=(args.patch_size, args.patch_size),
                                   mode='bilinear', align_corners=False)

        asr = evaluate_asr(model, val_loader, flower, args.patch_size)
        if asr > best_asr:
            best_asr = asr
            best_z = z.clone()
            best_flower = flower.clone()
            tqdm.write(f'  [{i}] New best ASR: {best_asr:.4f}')

    out_path = os.path.join(args.output_dir, f'bagnet17-cifar-ps{args.patch_size}.pt')
    torch.save(best_flower.cpu(), out_path)
    torch.save(best_z.cpu(), out_path.replace('.pt', '_z.pt'))
    print(f'Artifact saved to {out_path}')
    print(f'Best ASR: {best_asr:.4f}')


if __name__ == '__main__':
    main()
