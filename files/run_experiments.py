"""
run_experiments.py
==================
Orchestrates all 16 experiments:
    4 model types × 4 datasets = 16 runs

Usage:
    # Run all experiments
    python run_experiments.py

    # Run a single experiment for quick testing
    python run_experiments.py --attn standard --dataset cifar10 --epochs 5

    # Run on specific GPU
    python run_experiments.py --gpu 0

Results are saved to ./results/ as JSON summaries and CSV epoch logs.
Final comparison table is printed and saved to ./results/comparison_table.csv
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from model import build_model
from data  import get_dataloaders
from train import train


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter configuration
# ─────────────────────────────────────────────────────────────────────────────

# These are the declared hyperparameters for the report
TRAINING_CONFIG = {
    'epochs':          100,     # Total training epochs
    'warmup_epochs':   10,      # Linear warmup epochs
    'lr':              1e-3,    # Peak learning rate (AdamW)
    'weight_decay':    0.05,    # AdamW weight decay
    'batch_size':      128,     # Training batch size
    'grad_clip':       1.0,     # Gradient clipping norm
    'label_smoothing': 0.1,     # Cross-entropy label smoothing
    'amp':             True,    # Automatic mixed precision (GPU only)
    'seed':            42,      # Random seed
}

# Shared ViT architecture hyperparameters
MODEL_CONFIG = {
    'dim':          256,    # Embedding dimension
    'depth':        6,      # Number of transformer layers
    'heads':        8,      # Number of attention heads
    'mlp_dim':      512,    # FFN hidden dimension (= 2×dim)
    'dropout':      0.1,    # Dropout rate
    'num_features': 128,    # Random features r (Performer variants)
    'kernel':       'softmax',  # FAVOR+ kernel
    'pool':         'cls',  # Use [CLS] token for classification
}

DATASETS   = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']
ATTN_TYPES = ['standard', 'performer', 'cayley_string', 'circulant_string']

ATTN_DISPLAY_NAMES = {
    'standard':         'ViT (Standard Attention)',
    'performer':        'Performer (FAVOR+)',
    'cayley_string':    'Performer + Cayley-STRING',
    'circulant_string': 'Performer + Circulant-STRING',
}


# ─────────────────────────────────────────────────────────────────────────────
# Single experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(attn_type: str, dataset: str, config: dict,
                   save_dir: str) -> dict:
    """
    Runs one (attn_type, dataset) experiment.

    Returns the results summary dict.
    """
    set_seed(config['seed'])

    run_name = f"{attn_type}__{dataset}"
    print(f"\n{'━'*60}")
    print(f" Model : {ATTN_DISPLAY_NAMES[attn_type]}")
    print(f" Data  : {dataset.upper()}")
    print(f"{'━'*60}")

    # Check for existing results (resume support)
    summary_path = Path(save_dir) / f'{run_name}_summary.json'
    if summary_path.exists():
        print(f"  ↩  Found existing results, skipping: {summary_path}")
        with open(summary_path) as f:
            return json.load(f)

    # Build dataloaders
    train_loader, val_loader = get_dataloaders(
        dataset    = dataset,
        batch_size = config['batch_size'],
        num_workers= 4,
    )

    # Build model
    model = build_model(
        dataset   = dataset,
        attn_type = attn_type,
        **{k: MODEL_CONFIG[k] for k in MODEL_CONFIG},
    )

    # Train
    results = train(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        config       = config,
        run_name     = run_name,
        save_dir     = save_dir,
    )
    results['attn_type'] = attn_type
    results['dataset']   = dataset

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Results aggregation & display
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_table(all_results: list, save_dir: str):
    """
    Builds and prints a formatted comparison table.
    Saves CSV to save_dir/comparison_table.csv.
    """
    rows = []
    for r in all_results:
        rows.append({
            'Model':               ATTN_DISPLAY_NAMES.get(r['attn_type'], r['attn_type']),
            'Dataset':             r['dataset'].upper(),
            'Best Val Acc (%)':    f"{r['best_val_acc1']*100:.2f}",
            'Train (samples/sec)': f"{r.get('avg_train_samples_per_sec', 0):.0f}",
            'Inference (ms/img)':  f"{r.get('avg_val_ms_per_sample', 0):.3f}",
            'Parameters':          f"{r.get('num_parameters', 0):,}",
        })

    df = pd.DataFrame(rows)

    # Pivot so datasets are columns, models are rows
    pivot = df.pivot_table(
        index='Model',
        columns='Dataset',
        values='Best Val Acc (%)',
        aggfunc='first'
    )

    print("\n" + "="*80)
    print("VALIDATION ACCURACY COMPARISON (Top-1 %)")
    print("="*80)
    print(pivot.to_string())

    print("\n" + "="*80)
    print("FULL RESULTS TABLE")
    print("="*80)
    print(df.to_string(index=False))

    # Save
    df.to_csv(Path(save_dir) / 'comparison_table.csv', index=False)
    pivot.to_csv(Path(save_dir) / 'accuracy_pivot.csv')
    print(f"\n✓ Saved comparison tables to {save_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='ViT Attention Comparison Experiments')
    parser.add_argument('--attn',     type=str, default=None,
                        choices=ATTN_TYPES + [None],
                        help='Run only this attention type (default: all)')
    parser.add_argument('--dataset',  type=str, default=None,
                        choices=DATASETS + [None],
                        help='Run only this dataset (default: all)')
    parser.add_argument('--epochs',   type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--gpu',      type=int, default=0,
                        help='GPU index to use')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--num_features', type=int, default=128,
                        help='Number of random features for Performer variants')
    return parser.parse_args()


def main():
    args = parse_args()

    # Set GPU
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available — running on CPU (will be slow!)")

    os.makedirs(args.save_dir, exist_ok=True)

    # Build config, allow CLI overrides
    config = {**TRAINING_CONFIG}
    if args.epochs is not None:
        config['epochs'] = args.epochs
    config['seed'] = args.seed

    # Allow override of num_features
    if args.num_features != MODEL_CONFIG['num_features']:
        MODEL_CONFIG['num_features'] = args.num_features
        print(f"Using num_features = {args.num_features}")

    # Select which experiments to run
    attn_types = [args.attn]  if args.attn    else ATTN_TYPES
    datasets   = [args.dataset] if args.dataset else DATASETS

    experiments = [(a, d) for a in attn_types for d in datasets]
    print(f"\nRunning {len(experiments)} experiments: "
          f"{len(attn_types)} models × {len(datasets)} datasets")

    # Save experiment config for reproducibility
    exp_config = {
        'training': config,
        'model':    MODEL_CONFIG,
        'attn_types': attn_types,
        'datasets':   datasets,
        'timestamp':  time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(Path(args.save_dir) / 'experiment_config.json', 'w') as f:
        json.dump(exp_config, f, indent=2)

    # Run all experiments
    all_results = []
    total_start = time.time()

    for i, (attn_type, dataset) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}]", end='')
        results = run_experiment(attn_type, dataset, config, args.save_dir)
        all_results.append(results)

    total_time = time.time() - total_start
    print(f"\n\nAll experiments completed in {total_time/3600:.1f}h")

    # Print and save comparison
    build_comparison_table(all_results, args.save_dir)


if __name__ == '__main__':
    main()
