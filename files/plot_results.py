"""
plot_results.py
===============
Generates publication-quality plots from experiment results:
  1. Accuracy comparison bar chart (all models × datasets)
  2. Training curves (loss + accuracy per epoch)
  3. Speed vs Accuracy scatter plot
  4. Heatmap of best validation accuracies

Usage:
    python plot_results.py --results_dir ./results
"""

import argparse
import json
import csv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


ATTN_DISPLAY_NAMES = {
    'standard':         'ViT\n(Softmax Attn)',
    'performer':        'Performer\n(FAVOR+)',
    'cayley_string':    'Performer +\nCayley-STRING',
    'circulant_string': 'Performer +\nCirculant-STRING',
}

COLORS = {
    'standard':         '#2196F3',   # Blue
    'performer':        '#FF9800',   # Orange
    'cayley_string':    '#4CAF50',   # Green
    'circulant_string': '#9C27B0',   # Purple
}

DATASET_ORDER = ['MNIST', 'FASHION_MNIST', 'CIFAR10', 'CIFAR100']


def load_all_summaries(results_dir: str) -> pd.DataFrame:
    """Load all *_summary.json files into a DataFrame."""
    rows = []
    for path in Path(results_dir).glob('*_summary.json'):
        with open(path) as f:
            d = json.load(f)
        rows.append(d)
    return pd.DataFrame(rows)


def load_epoch_log(results_dir: str, attn_type: str, dataset: str) -> pd.DataFrame:
    """Load per-epoch CSV log for a specific run."""
    log_path = Path(results_dir) / f'{attn_type}__{dataset}_log.csv'
    if not log_path.exists():
        return None
    return pd.read_csv(log_path)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Accuracy bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_comparison(df: pd.DataFrame, save_path: str):
    """
    Grouped bar chart: x-axis = datasets, groups = model types.
    """
    datasets   = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']
    attn_types = ['standard', 'performer', 'cayley_string', 'circulant_string']

    fig, ax = plt.subplots(figsize=(14, 6))

    n_groups = len(datasets)
    n_bars   = len(attn_types)
    width    = 0.18
    x        = np.arange(n_groups)

    for i, attn in enumerate(attn_types):
        heights = []
        for ds in datasets:
            row = df[(df['attn_type'] == attn) & (df['dataset'] == ds)]
            acc = row['best_val_acc1'].values[0] * 100 if len(row) > 0 else 0
            heights.append(acc)
        offset = (i - n_bars / 2 + 0.5) * width
        bars = ax.bar(x + offset, heights, width,
                      label=ATTN_DISPLAY_NAMES[attn].replace('\n', ' '),
                      color=COLORS[attn], alpha=0.85, edgecolor='white')
        # Add value labels on bars
        for bar, h in zip(bars, heights):
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                        f'{h:.1f}', ha='center', va='bottom',
                        fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([d.upper().replace('_', '-') for d in datasets], fontsize=12)
    ax.set_ylabel('Top-1 Validation Accuracy (%)', fontsize=12)
    ax.set_title('Validation Accuracy by Model and Dataset', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(results_dir: str, dataset: str, save_path: str):
    """
    4-panel plot: val accuracy over epochs for each model on a single dataset.
    """
    attn_types = ['standard', 'performer', 'cayley_string', 'circulant_string']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Training Curves — {dataset.upper().replace("_", "-")}',
                 fontsize=14, fontweight='bold')

    for attn in attn_types:
        log = load_epoch_log(results_dir, attn, dataset)
        if log is None:
            continue
        label = ATTN_DISPLAY_NAMES[attn].replace('\n', ' ')
        color = COLORS[attn]
        axes[0].plot(log['epoch'], log['val_acc1'] * 100,
                     label=label, color=color, linewidth=1.8)
        axes[1].plot(log['epoch'], log['val_loss'],
                     label=label, color=color, linewidth=1.8)

    for ax, ylabel in zip(axes, ['Val Accuracy (%)', 'Val Loss']):
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Speed vs Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def plot_speed_accuracy(df: pd.DataFrame, dataset: str, save_path: str):
    """
    Scatter plot: x = inference latency (ms/img), y = val accuracy.
    Each point is one model; marker size = train speed.
    """
    attn_types = ['standard', 'performer', 'cayley_string', 'circulant_string']

    fig, ax = plt.subplots(figsize=(8, 6))

    for attn in attn_types:
        row = df[(df['attn_type'] == attn) & (df['dataset'] == dataset)]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        x  = row.get('avg_val_ms_per_sample', 0)
        y  = row['best_val_acc1'] * 100
        ax.scatter(x, y, s=150, color=COLORS[attn], zorder=5,
                   label=ATTN_DISPLAY_NAMES[attn].replace('\n', ' '))
        ax.annotate(ATTN_DISPLAY_NAMES[attn].replace('\n', ' '),
                    (x, y), textcoords='offset points',
                    xytext=(6, 4), fontsize=8)

    ax.set_xlabel('Inference Latency (ms per sample)', fontsize=12)
    ax.set_ylabel('Best Val Accuracy (%)', fontsize=12)
    ax.set_title(f'Speed vs. Accuracy — {dataset.upper().replace("_","-")}',
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_heatmap(df: pd.DataFrame, save_path: str):
    """
    Heatmap: rows = model types, cols = datasets, values = best val accuracy.
    """
    attn_types = ['standard', 'performer', 'cayley_string', 'circulant_string']
    datasets   = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']

    data = np.zeros((len(attn_types), len(datasets)))
    for i, attn in enumerate(attn_types):
        for j, ds in enumerate(datasets):
            row = df[(df['attn_type'] == attn) & (df['dataset'] == ds)]
            if len(row) > 0:
                data[i, j] = row['best_val_acc1'].values[0] * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=data[data>0].min()-2)

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([d.upper().replace('_', '-') for d in datasets], fontsize=11)
    ax.set_yticks(range(len(attn_types)))
    ax.set_yticklabels([ATTN_DISPLAY_NAMES[a].replace('\n', ' ') for a in attn_types], fontsize=10)

    # Annotate cells
    for i in range(len(attn_types)):
        for j in range(len(datasets)):
            val = data[i, j]
            text_color = 'white' if val > data.mean() else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                    fontsize=11, color=text_color, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Validation Accuracy (%)')
    ax.set_title('Best Validation Accuracy Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./results')
    args = parser.parse_args()

    plots_dir = Path(args.results_dir) / 'plots'
    plots_dir.mkdir(exist_ok=True)

    df = load_all_summaries(args.results_dir)
    if df.empty:
        print("No summary files found. Run experiments first.")
        return

    # Plot 1: Accuracy comparison
    plot_accuracy_comparison(df, str(plots_dir / 'accuracy_comparison.png'))

    # Plot 2: Training curves (one per dataset)
    for ds in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']:
        plot_training_curves(args.results_dir, ds,
                             str(plots_dir / f'curves_{ds}.png'))

    # Plot 3: Speed vs accuracy (one per dataset)
    for ds in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']:
        plot_speed_accuracy(df, ds,
                            str(plots_dir / f'speed_accuracy_{ds}.png'))

    # Plot 4: Heatmap
    plot_accuracy_heatmap(df, str(plots_dir / 'accuracy_heatmap.png'))

    print(f"\n✓ All plots saved to {plots_dir}/")


if __name__ == '__main__':
    main()
