"""Generate PGD patch adversarial examples for CIFAR-10 + BagNet17."""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nets'))
from PatchAttacker import PatchAttacker
import nets.bagnet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, required=True, choices=[16, 32])
    parser.add_argument('--data_dir', default='../data/cifar')
    parser.add_argument('--output_dir', default='../../shared_data/pgd_adv')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_images', type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(192),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    val_dataset = datasets.CIFAR10(root=args.data_dir, train=False,
                                   download=True, transform=transform)
    from torch.utils.data import Subset
    val_dataset = Subset(val_dataset, range(args.max_images))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = nets.bagnet.bagnet17(pretrained=True, aggregation='mean')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model = nn.DataParallel(model)
    ckpt = torch.load('../checkpoints/bagnet17_192_cifar.pth',
                      weights_only=False, map_location=DEVICE)
    model.load_state_dict(ckpt['net'])
    model.eval()

    attacker = PatchAttacker(
        model=model,
        mean=CIFAR_MEAN,
        std=CIFAR_STD,
        image_size=192,
        patch_size=args.patch_size,
        steps=args.steps,
        step_size=0.05,
        random_start=True,
    )

    all_adv, all_labels = [], []

    print(f'Generating PGD adv examples for ps={args.patch_size} ({args.steps} steps)...')
    for images, labels in tqdm(val_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        adv_images, _ = attacker.perturb(images, labels)
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
