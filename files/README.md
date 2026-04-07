# Efficient Vision Transformer Models
## IEOR Data Mining — Research Project

Comparing four attention mechanisms inside Vision Transformer (ViT) architectures
across four image classification benchmarks.

---

## Models Compared

| ID | Model | Attention | Complexity |
|---|---|---|---|
| 1 | **ViT** | Standard Softmax | O(N²·d) |
| 2 | **Performer** | FAVOR+ (softmax kernel) | O(N·r·d) |
| 3 | **Performer + Cayley-STRING** | FAVOR+ + Cayley RPE | O(N·r·d) |
| 4 | **Performer + Circulant-STRING** | FAVOR+ + Circulant RPE | O(N·r·d) |

## Datasets

| Dataset | Classes | Size | Train | Val |
|---|---|---|---|---|
| MNIST | 10 | 28×28 gray | 60,000 | 10,000 |
| Fashion-MNIST | 10 | 28×28 gray | 60,000 | 10,000 |
| CIFAR-10 | 10 | 32×32 RGB | 50,000 | 10,000 |
| CIFAR-100 | 100 | 32×32 RGB | 50,000 | 10,000 |

---

## Setup

```bash
# 1. Create environment
conda create -n vit_project python=3.10
conda activate vit_project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify everything works (runs in ~1 min)
python sanity_check.py
```

---

## Running Experiments

```bash
# Full experiment suite (16 runs, ~8-12h on a single GPU)
python run_experiments.py

# Single experiment (for testing)
python run_experiments.py --attn standard --dataset cifar10 --epochs 5

# Specify GPU
python run_experiments.py --gpu 0

# Override number of random features
python run_experiments.py --num_features 256
```

Results are saved to `./results/` with:
- `{model}__{dataset}_log.csv` — per-epoch metrics
- `{model}__{dataset}_summary.json` — final stats
- `{model}__{dataset}_best.pth` — best checkpoint

---

## Generating Plots

```bash
python plot_results.py --results_dir ./results
```

Outputs to `./results/plots/`:
- `accuracy_comparison.png` — grouped bar chart
- `curves_{dataset}.png` — training curves per dataset
- `speed_accuracy_{dataset}.png` — speed vs accuracy scatter
- `accuracy_heatmap.png` — heatmap of all results

---

## Declared Hyperparameters (for report)

### Architecture
| Parameter | Value | Notes |
|---|---|---|
| Patch size | 4×4 | For both 28×28 and 32×32 images |
| Embedding dim (D) | 256 | Model width |
| Depth | 6 | Number of transformer layers |
| Attention heads | 8 | Head dim = 256/8 = 32 |
| FFN hidden dim | 512 | = 2× embedding dim |
| Pool | CLS token | Classification via [CLS] token |

### Performer-specific
| Parameter | Value | Notes |
|---|---|---|
| Random features (r) | **128** | Key hyperparameter: controls accuracy/speed tradeoff |
| Kernel type | **softmax** | FAVOR+ positive random features |
| Ortho. features | Yes | Orthogonal random matrices (lower variance) |

### Training
| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 (peak) |
| LR schedule | Cosine annealing |
| Warmup epochs | 10 |
| Weight decay | 0.05 |
| Batch size | 128 |
| Epochs | 100 |
| Gradient clip | 1.0 |
| Label smoothing | 0.1 |
| Random seed | 42 |

---

## Code Structure

```
vit_project/
├── attention.py          # All 4 attention mechanisms
│   ├── StandardAttention        — O(N²) softmax attention
│   ├── PerformerAttention       — FAVOR+ linear attention
│   ├── CayleySTRINGAttention    — FAVOR+ + Cayley RPE
│   └── CirculantSTRINGAttention — FAVOR+ + Circulant RPE
│
├── model.py              # ViT backbone + build_model() factory
├── data.py               # DataLoader factories for all datasets
├── train.py              # Training loop, evaluation, logging
├── run_experiments.py    # Orchestrates all 16 experiments
├── plot_results.py       # Result visualization
├── sanity_check.py       # Pre-flight checks
└── requirements.txt      # Dependencies
```

---

## References

1. Dosovitskiy et al. (2020). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.* ICLR 2021.
2. Choromanski et al. (2020). *Rethinking Attention with Performers.* ICLR 2021.
3. Schenck et al. (2025). *STRING: Structured Relative Positional Encoding for Performers.*
