# CHANGELOG

## 2026-07-08 — TnT vs PatchGuard 对比实验

### 修复
- `TnT/gan.py` — 完全重写，移除 torchgan 依赖，用 standalone `FlowerGANGenerator` 直接加载预训练权重；修复缺少 `import torch`、`Adam`、`device` 的问题
- `TnT/utils.py` — 移除未使用的 `from pudb import set_trace`（pudb 未安装导致 import 失败）
- `torchgan/models/dcgan.py` — 注释掉 2 的幂检查，允许 PATCH_SIZE=96/128 等非标准尺寸

### 新增
- `CHANGELOG.md` — 本文件
- `TnT/gen_artifact.py` — GAN latent space 随机搜索，生成针对 BagNet17+CIFAR10 的 TnT UAP artifact
- `TnT/attack_bagnet.py` — 加载 artifact，blend 到 CIFAR-10 val set 生成对抗样本
- `TnT/artifacts/bagnet17-cifar-ps16.pt` — ps=16 TnT artifact (ASR=0.2280)
- `TnT/artifacts/bagnet17-cifar-ps32.pt` — ps=32 TnT artifact (ASR=0.2320)
- `PatchGuard/misc/gen_pgd_adv.py` — PGD patch 攻击样本生成（baseline 攻击）
- `PatchGuard/eval_attack.py` — 统一防御评测脚本（MASK + CBN + PG++ × Clean/PGD/TnT）
- `shared_data/tnt_adv/` — TnT 攻击样本 (10000 张 CIFAR-10)
- `shared_data/pgd_adv/` — PGD 攻击样本 (500 张 CIFAR-10)
- `shared_data/results.npz` — 评测结果数据

### 环境
- 使用 `adv-defense` conda 环境，Python 3.11，PyTorch 2.7.1+cu118
- GPU: NVIDIA GeForce RTX 4060 Laptop
- 下载了 CIFAR-10 数据集到 `PatchGuard/data/cifar/cifar-10-batches-py/`
- 下载了 checkpoint: `bagnet17_192_cifar.pth` (55MB), `bagnet33_192_cifar.pth` (63MB)
- 下载了 flower GAN 预训练模型: `gan4.model` (286MB)
- GAN 实际输出 128×128（非原始代码标注的 96×96）

### 实验结果 (CIFAR-10, BagNet17/33, 500 张测试)

#### Attack Success Rate (ASR, 越低越好)
| Attack | ps=16 | ps=32 |
|--------|:-----:|:-----:|
| PGD    | 69.2% | 98.8% |
| TnT    | 22.2% | 22.8% |

#### PatchGuard Masking (--m)
| 指标 | Clean | PGD16 | TnT16 | PGD32 | TnT32 |
|------|:-----:|:-----:|:-----:|:-----:|:-----:|
| Provable Robust | 35.6% | 14.0% | 52.2% | 0.0%  | 30.8% |
| Clean+Defense   | 75.8% | 70.6% | 76.8% | 66.6% | 76.6% |

#### CBN Baseline (--cbn)
| 指标 | Clean | PGD16 | TnT16 | PGD32 | TnT32 |
|------|:-----:|:-----:|:-----:|:-----:|:-----:|
| Provable Robust | 15.0% | 33.2% | 44.0% | 3.0%  | 14.6% |
| Clean+Defense   | 75.6% | 67.0% | 75.2% | 51.0% | 76.8% |

#### PatchGuard++ Detection (--det, BagNet33)
| 指标 | Clean | PGD16 | TnT16 | PGD32 | TnT32 |
|------|:-----:|:-----:|:-----:|:-----:|:-----:|
| Provable Robust | 73.4% | 73.6% | 76.8% | 60.2% | 72.4% |
| Clean+Defense   | 73.4% | 73.6% | 76.8% | 60.2% | 72.4% |

### 关键发现
1. **TnT 攻击力远弱于 PGD**：TnT ASR 仅 ~22%，而 PGD 达到 69-99%。自然花卉 patch 无法有效攻击 BagNet 的小感受野局部分类器
2. **PatchGuard 对弱攻击的 Provable Robust 反而更高**：TnT 下 masking PR 高于 Clean baseline，因为大部分样本仍被正确分类（case 0 少），剩余样本中可认证比例高
3. **PG++ (BagNet33) 是最稳定的防御**：在所有攻击下 PR 维持在 60-77%，且对弱攻击（TnT）的 PR 接近 Clean baseline
4. **CBN 对强攻击 (PGD32) 几乎失效**：PR 仅 3.0%，CAD 降至 51.0%
5. **核心结论**：TnT 的自然外观 patch 能绕过 PatchGuard 检测的前提是 patch 必须能攻破模型——但 BagNet 的小感受野架构本身就使自然 patch 难以成为有效攻击。即"看起来自然 ≠ 能骗过局部分类器"
