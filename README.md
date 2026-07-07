# TnT vs PatchGuard 对比实验

信息安全大作业 v2.0，验证 GAN 生成的自然外观对抗 patch（TnT）是否能绕过 PatchGuard 防御。

## 背景

PatchGuard (USENIX Security'21) 是一种针对 patch 对抗攻击的可证明鲁棒防御，通过 BagNet 的局部特征和滑动窗口 masking 检测/屏蔽恶意 patch。其威胁模型假设攻击是"空间局部的异常信号"。

TnT 使用 GAN 生成看起来自然的对抗 patch（如花卉图案），将其无缝融合到图片中来欺骗分类器。本实验验证：**自然外观的 patch 能否绕过 PatchGuard 的异常检测？**

v1.0 复现了 Erase-and-Restore 和 PatchGuard；v2.0 聚焦 PatchGuard，增加 TnT 攻击作为对比。

## 实验设计

| 维度 | 内容 |
|------|------|
| 数据集 | CIFAR-10（resize 到 192×192） |
| 目标模型 | BagNet17（Masking/CBN） / BagNet33（PG++ 检测） |
| 攻击方法 | TnT（GAN flower patch）、PGD（baseline） |
| Patch 大小 | 16×16, 32×32 |
| 防御模式 | Masking、CBN、PG++ Detection、无防御 |
| 评测指标 | Attack Success Rate、Provable Robust Accuracy、Clean Accuracy+Defense |

## 环境搭建

```bash
conda create -n adv-defense python=3.11 -y
conda activate adv-defense
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python scikit-learn scikit-image pandas joblib tqdm scipy torchgan
```

## 所需文件

从 Google Drive 下载以下文件：

| 文件 | 链接 | 放置位置 |
|------|------|---------|
| CIFAR-10 checkpoint | [Google Drive](https://drive.google.com/drive/folders/1u5RsCuZNf7ddWW0utI4OrgWGmJCUDCuT) | `PatchGuard/checkpoints/` |
| Flower GAN 预训练模型 | [Google Drive](https://drive.google.com/file/d/1etVNk5GU2Ux4uclKzSwqujGvAAzY5_pC) | `TnT/gan4.model` |
| CIFAR-10 数据集 | [官网](https://www.cs.toronto.edu/~kriz/cifar.html) | `PatchGuard/data/cifar/cifar-10-batches-py/` |

需要的 checkpoint:
- `bagnet17_192_cifar.pth` — BagNet17 for CIFAR-10
- `bagnet33_192_cifar.pth` — BagNet33 for CIFAR-10

## 项目结构

```
├── PatchGuard/
│   ├── mask_bn.py          # PatchGuard masking 防御
│   ├── det_bn.py           # PatchGuard++ 攻击检测
│   ├── mask_ds.py          # Derandomized Smoothing + Masking
│   ├── eval_attack.py      # 统一防御评测脚本 ★
│   ├── nets/               # BagNet / ResNet 模型
│   ├── utils/              # defense_utils.py (核心算法)
│   ├── checkpoints/        # 预训练权重
│   ├── data/cifar/         # CIFAR-10 数据集
│   └── misc/               # 训练/攻击脚本
│       ├── gen_pgd_adv.py  # PGD 攻击样本生成 ★
│       └── PatchAttacker.py # PGD patch 攻击器
├── TnT/
│   ├── gan.py              # Flower GAN 生成器（已修复，独立于 torchgan）
│   ├── gen_artifact.py     # TnT artifact 生成 ★
│   ├── attack_bagnet.py    # TnT 攻击样本生成 ★
│   ├── utils.py            # blend 工具函数
│   ├── artifacts/          # 生成的 TnT UAP
│   └── gan4.model          # 预训练 GAN 权重
├── shared_data/
│   ├── tnt_adv/            # TnT 对抗样本 (.npy)
│   ├── pgd_adv/            # PGD 对抗样本 (.npy)
│   └── results.npz         # 评测结果
├── docs/superpowers/       # 设计文档和实现计划
├── CHANGELOG.md            # 改动记录 + 实验结果
├── FROM_V1_LESSONS.md      # v1.0 经验总结
└── README.md               # 本文件
```

★ = v2.0 新增文件

## 实验流程

### 1. 生成 TnT Artifact

```bash
cd TnT
python gen_artifact.py --patch_size 16 --n_search 500
python gen_artifact.py --patch_size 32 --n_search 500
```

在 GAN latent space 中随机搜索 500 个向量，选 ASR 最高的花卉图案作为 UAP。

### 2. 生成 TnT 攻击样本

```bash
cd TnT
python attack_bagnet.py --patch_size 16 --artifact artifacts/bagnet17-cifar-ps16.pt
python attack_bagnet.py --patch_size 32 --artifact artifacts/bagnet17-cifar-ps32.pt
```

将 artifact 贴到全部 CIFAR-10 验证集图片上，保存为 `.npy`。

### 3. 生成 PGD 攻击样本 (Baseline)

```bash
cd PatchGuard/misc
python gen_pgd_adv.py --patch_size 16 --max_images 500
python gen_pgd_adv.py --patch_size 32 --max_images 500
```

### 4. 统一评测

```bash
cd PatchGuard
python eval_attack.py --max_images 500
```

对所有攻击×防御组合进行评测，输出 SUMMARY 表格，同时保存 `shared_data/results.npz`。

## 实验结果

### Attack Success Rate

| Attack | ps=16 | ps=32 |
|--------|:-----:|:-----:|
| PGD    | 69.2% | 98.8% |
| TnT    | 22.2% | 22.8% |

### PatchGuard Masking

| 指标 | Clean | PGD16 | TnT16 | PGD32 | TnT32 |
|------|:-----:|:-----:|:-----:|:-----:|:-----:|
| Provable Robust | 35.6% | 14.0% | 52.2% | 0.0% | 30.8% |
| Clean+Defense | 75.8% | 70.6% | 76.8% | 66.6% | 76.6% |

### PatchGuard++ Detection

| 指标 | Clean | PGD16 | TnT16 | PGD32 | TnT32 |
|------|:-----:|:-----:|:-----:|:-----:|:-----:|
| Provable Robust | 73.4% | 73.6% | 76.8% | 60.2% | 72.4% |

> 干净精度 (BagNet17, 无防御): 77.8% | 500 张 CIFAR-10 子集

## 关键结论

1. **TnT 攻击力远弱于 PGD**：ASR 22% vs 69-99%。自然花卉 patch 无法有效攻击 BagNet 的小感受野局部分类器
2. **PG++ 是最稳定的防御**：在所有攻击下可证明鲁棒率维持在 60-77%
3. **核心发现**：自然外观 patch 能"绕过检测"的前提是它能先"攻破模型"——但 BagNet 架构本身就使自然 patch 难以成为有效攻击。TnT 没有绕过 PatchGuard，因为它连 BagNet 的基础分类都没攻破。

## 修复记录

- `TnT/gan.py`：移除 torchgan 依赖，用 standalone FlowerGANGenerator 直接加载权重
- `TnT/utils.py`：移除未使用的 pudb import
- `torchgan/models/dcgan.py`：允许非 2 的幂尺寸

详见 [CHANGELOG.md](CHANGELOG.md)。

## 参考文献

- **[PatchGuard]** Chong Xiang, Arjun Nitin Bhagoji, Vikash Sehwag, Prateek Mittal. *PatchGuard: A Provably Robust Defense against Adversarial Patches via Small Receptive Fields and Masking*. USENIX Security Symposium, 2021. [Paper](https://www.usenix.org/conference/usenixsecurity21/presentation/xiang)
- **[TnT]** Bao Gia Doan, Minhui Xue, Shiqing Ma, Ehsan Abbasnejad, Damith C. Ranasinghe. *TnT Attacks! Universal Naturalistic Adversarial Patches Against Deep Neural Network Systems*. IEEE Transactions on Information Forensics & Security (TIFS), 2022. [arXiv:2111.09999](https://arxiv.org/abs/2111.09999)
