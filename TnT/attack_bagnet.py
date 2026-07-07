"""Generate TnT adversarial examples for the full CIFAR-10 val set."""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'PatchGuard'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'PatchGuard', 'nets'))

from gen_artifact import blend_cifar

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, required=True, choices=[16, 32])
    parser.add_argument('--artifact', type=str, required=True)
    parser.add_argument('--data_dir', default='../PatchGuard/data/cifar')
    parser.add_argument('--output_dir', default='../shared_data/tnt_adv')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    flower = torch.load(args.artifact, map_location=DEVICE, weights_only=False)
    if flower.dim() == 3:
        flower = flower.unsqueeze(0)
    print(f'Loaded artifact: {args.artifact}, shape={flower.shape}')

    transform = transforms.Compose([
        transforms.Resize(192),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    val_dataset = datasets.CIFAR10(root=args.data_dir, train=False,
                                   download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    all_adv, all_labels = [], []

    print(f'Generating TnT adv examples for ps={args.patch_size}...')
    for images, labels in tqdm(val_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        with torch.no_grad():
            adv_images = blend_cifar(images, flower, args.patch_size)
        all_adv.append(adv_images.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    adv_all = np.concatenate(all_adv, axis=0).astype(np.float32)
    labels_all = np.concatenate(all_labels, axis=0)

    adv_path = os.path.join(args.output_dir, f'ps{args.patch_size}_adv.npy')
    lbl_path = os.path.join(args.output_dir, f'ps{args.patch_size}_labels.npy')
    np.save(adv_path, adv_all)
    np.save(lbl_path, labels_all)
    print(f'Saved: {adv_path} ({adv_all.shape})')
    print(f'Saved: {lbl_path} ({labels_all.shape})')


if __name__ == '__main__':
    main()
