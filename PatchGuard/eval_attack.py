"""Unified evaluation: test PatchGuard defenses against pre-generated attacks."""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import os
import sys
import argparse
from math import ceil
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'PatchGuard'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nets'))
from utils.defense_utils import (
    provable_masking, masking_defense,
    provable_clipping, clipping_defense,
    pg2_detection, pg2_detection_provable,
)
import nets.bagnet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RF_STRIDE = 8


def compute_window_size(patch_size, rf_size):
    return ceil((patch_size + rf_size - 1) / RF_STRIDE)


def evaluate_masking(model, dataloader, patch_size, rf_size, thres=0.0):
    """Run PatchGuard masking defense (--m)."""
    model.eval()
    ws = [compute_window_size(patch_size, rf_size)] * 2
    result_list, clean_corr = [], 0

    for images, labels in tqdm(dataloader, desc='Masking'):
        images = images.to(DEVICE)
        labels_np = labels.numpy()
        with torch.no_grad():
            output = model(images).cpu().numpy()
        for i in range(len(labels_np)):
            r = provable_masking(output[i], labels_np[i], thres=thres, window_shape=ws)
            result_list.append(r)
            clean_corr += int(masking_defense(output[i], thres=thres, window_shape=ws) == labels_np[i])

    cases, cnt = np.unique(result_list, return_counts=True)
    prv = cnt[-1] / len(result_list) if len(cnt) == 3 else 0.0
    return prv, clean_corr / len(result_list)


def evaluate_cbn(model, dataloader, patch_size, rf_size):
    """Run CBN baseline defense (--cbn)."""
    model.eval()
    ws = [compute_window_size(patch_size, rf_size)] * 2
    result_list, clean_corr = [], 0

    for images, labels in tqdm(dataloader, desc='CBN'):
        images = images.to(DEVICE)
        labels_np = labels.numpy()
        with torch.no_grad():
            output = model(images).cpu().numpy()
        for i in range(len(labels_np)):
            r = provable_clipping(output[i], labels_np[i], window_shape=ws)
            result_list.append(r)
            clean_corr += int(clipping_defense(output[i]) == labels_np[i])

    cases, cnt = np.unique(result_list, return_counts=True)
    prv = cnt[-1] / len(result_list) if len(cnt) == 3 else 0.0
    return prv, clean_corr / len(result_list)


def evaluate_pg2(model, dataloader, patch_size, rf_size, tau=0.0):
    """Run PatchGuard++ detection (--det)."""
    model.eval()
    ws = [compute_window_size(patch_size, rf_size)] * 2
    result_list, clean_corr = [], 0

    for images, labels in tqdm(dataloader, desc='PG++'):
        images = images.to(DEVICE)
        labels_np = labels.numpy()
        with torch.no_grad():
            output = model(images).cpu().numpy()
        for i in range(len(labels_np)):
            r = pg2_detection_provable(output[i], labels_np[i], tau=tau, window_shape=ws)
            result_list.append(r)
            clean_corr += int(pg2_detection(output[i], tau=tau, window_shape=ws) == labels_np[i])

    cases, cnt = np.unique(result_list, return_counts=True)
    prv = cnt[-1] / len(result_list) if len(cnt) == 3 else 0.0
    return prv, clean_corr / len(result_list)


def evaluate_no_defense(model, dataloader):
    """Evaluate accuracy / ASR without defense."""
    model.eval()
    correct, total = 0, 0
    for images, labels in tqdm(dataloader, desc='NoDef'):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        with torch.no_grad():
            output = model(images)
            if output.dim() == 4:
                output = output.mean(dim=(1, 2))
        correct += (output.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


def load_adv_dataset(adv_dir, tag, patch_size, max_images=500):
    adv_path = os.path.join(adv_dir, f'ps{patch_size}_adv.npy')
    lbl_path = os.path.join(adv_dir, f'ps{patch_size}_labels.npy')
    adv = np.load(adv_path)[:max_images]
    labels = np.load(lbl_path)[:max_images]
    print(f'Loaded {tag} ps={patch_size}: {adv.shape}')
    ds = TensorDataset(torch.from_numpy(adv), torch.from_numpy(labels.astype(np.int64)))
    return DataLoader(ds, batch_size=8, shuffle=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adv_dir', type=str, default='../shared_data')
    parser.add_argument('--model_dir', type=str, default='checkpoints')
    parser.add_argument('--max_images', type=int, default=500)
    args = parser.parse_args()

    # ========== Clean baseline ==========
    print('=' * 60)
    print('CLEAN BASELINE')
    print('=' * 60)

    transform = transforms.Compose([
        transforms.Resize(192), transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    clean_ds = datasets.CIFAR10(root='data/cifar', train=False, download=False, transform=transform)
    clean_ds = Subset(clean_ds, range(args.max_images))
    clean_loader = DataLoader(clean_ds, batch_size=8, shuffle=False)

    # BagNet17 for masking/CBN
    print('Loading BagNet17...')
    model17 = nets.bagnet.bagnet17(pretrained=True, aggregation='none')
    model17.fc = nn.Linear(model17.fc.in_features, 10)
    model17 = nn.DataParallel(model17)
    ckpt17 = torch.load(os.path.join(args.model_dir, 'bagnet17_192_cifar.pth'),
                        weights_only=False, map_location=DEVICE)
    model17.load_state_dict(ckpt17['net'])
    model17 = model17.to(DEVICE)

    # BagNet33 for PG++
    print('Loading BagNet33...')
    model33 = nets.bagnet.bagnet33(pretrained=True, aggregation='none')
    model33.fc = nn.Linear(model33.fc.in_features, 10)
    model33 = nn.DataParallel(model33)
    ckpt33 = torch.load(os.path.join(args.model_dir, 'bagnet33_192_cifar.pth'),
                        weights_only=False, map_location=DEVICE)
    model33.load_state_dict(ckpt33['net'])
    model33 = model33.to(DEVICE)

    # Clean accuracy (no attack, no defense)
    print('\n--- Clean (BagNet17, no defense) ---')
    clean_acc = evaluate_no_defense(model17, clean_loader)
    print(f'Clean accuracy: {clean_acc:.4f}')

    print('\n--- Clean + Masking ---')
    cl_pr, cl_cad = evaluate_masking(model17, clean_loader, patch_size=32, rf_size=17)

    print('\n--- Clean + CBN ---')
    cl_pr_c, cl_cad_c = evaluate_cbn(model17, clean_loader, patch_size=32, rf_size=17)

    print('\n--- Clean + PG++ ---')
    cl_pr_d, cl_cad_d = evaluate_pg2(model33, clean_loader, patch_size=32, rf_size=33)

    results = {
        'clean_acc': clean_acc,
        'clean_masking_pr': cl_pr, 'clean_masking_cad': cl_cad,
        'clean_cbn_pr': cl_pr_c, 'clean_cbn_cad': cl_cad_c,
        'clean_pg2_pr': cl_pr_d, 'clean_pg2_cad': cl_cad_d,
    }

    # ========== Attack evaluation ==========
    for attack_name, adv_subdir in [('PGD', 'pgd_adv'), ('TnT', 'tnt_adv')]:
        for ps in [16, 32]:
            tag = f'{attack_name} ps={ps}'
            print('\n' + '=' * 60)
            print(tag)
            print('=' * 60)

            adv_loader = load_adv_dataset(
                os.path.join(args.adv_dir, adv_subdir), attack_name, ps,
                max_images=args.max_images)

            # ASR (no defense)
            asr = 1.0 - evaluate_no_defense(model17, adv_loader)
            print(f'  ASR (no defense): {asr:.4f}')

            # Masking
            pr_m, cad_m = evaluate_masking(model17, adv_loader, patch_size=ps, rf_size=17)
            print(f'  Masking:   PR={pr_m:.4f}, CAD={cad_m:.4f}')

            # CBN
            pr_c, cad_c = evaluate_cbn(model17, adv_loader, patch_size=ps, rf_size=17)
            print(f'  CBN:       PR={pr_c:.4f}, CAD={cad_c:.4f}')

            # PG++
            pr_d, cad_d = evaluate_pg2(model33, adv_loader, patch_size=ps, rf_size=33)
            print(f'  PG++:      PR={pr_d:.4f}, CAD={cad_d:.4f}')

            prefix = f'{attack_name.lower()}_ps{ps}'
            results[f'{prefix}_asr'] = asr
            results[f'{prefix}_masking_pr'] = pr_m
            results[f'{prefix}_masking_cad'] = cad_m
            results[f'{prefix}_cbn_pr'] = pr_c
            results[f'{prefix}_cbn_cad'] = cad_c
            results[f'{prefix}_pg2_pr'] = pr_d
            results[f'{prefix}_pg2_cad'] = cad_d

    # ========== Summary ==========
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    hdr = f'{"":>25} {"Clean":>8} {"PGD16":>10} {"TnT16":>10} {"PGD32":>10} {"TnT32":>10}'
    print(hdr)

    c = lambda k: results.get(k, 0)
    rows = [
        ('Clean Accuracy', 'clean_acc', ['pgd_ps16_asr']),
        ('ASR (no defense)', None, ['pgd_ps16_asr', 'tnt_ps16_asr', 'pgd_ps32_asr', 'tnt_ps32_asr']),
    ]

    print(f'{"Clean Accuracy":>25} {c("clean_acc"):>8.4f} {"-":>10} {"-":>10} {"-":>10} {"-":>10}')
    print(f'{"ASR (no defense)":>25} {"-":>8} {c("pgd_ps16_asr"):>10.4f} {c("tnt_ps16_asr"):>10.4f} {c("pgd_ps32_asr"):>10.4f} {c("tnt_ps32_asr"):>10.4f}')

    for dname, dtag in [('MASK', 'masking'), ('CBN', 'cbn'), ('PG++', 'pg2')]:
        print(f'\n{dtag.upper()}:')
        print(f'{"  Provable Robust":>25} {c(f"clean_{dtag}_pr"):>8.4f} {c(f"pgd_ps16_{dtag}_pr"):>10.4f} {c(f"tnt_ps16_{dtag}_pr"):>10.4f} {c(f"pgd_ps32_{dtag}_pr"):>10.4f} {c(f"tnt_ps32_{dtag}_pr"):>10.4f}')
        print(f'{"  Clean+Defense":>25} {c(f"clean_{dtag}_cad"):>8.4f} {c(f"pgd_ps16_{dtag}_cad"):>10.4f} {c(f"tnt_ps16_{dtag}_cad"):>10.4f} {c(f"pgd_ps32_{dtag}_cad"):>10.4f} {c(f"tnt_ps32_{dtag}_cad"):>10.4f}')

    out_path = os.path.join(args.adv_dir, 'results.npz')
    np.savez(out_path, **results)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
