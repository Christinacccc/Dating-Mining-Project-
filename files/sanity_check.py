"""
sanity_check.py
===============
Quick test to verify all 4 attention mechanisms work correctly
before launching full training runs.

Checks:
  1. Forward pass completes without error
  2. Output shape is correct
  3. Backward pass (gradients) flows through all parameters
  4. Approximate parameter count
  5. Single-batch timing

Run this first before run_experiments.py !

Usage:
    python sanity_check.py
"""

import time
import torch
from model import build_model, ATTENTION_REGISTRY

ATTN_TYPES = ['standard', 'performer', 'cayley_string', 'circulant_string']
DATASETS   = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']

DATASET_SHAPES = {
    'mnist':         (1, 28, 28),
    'fashion_mnist': (1, 28, 28),
    'cifar10':       (3, 32, 32),
    'cifar100':      (3, 32, 32),
}
DATASET_CLASSES = {
    'mnist': 10, 'fashion_mnist': 10, 'cifar10': 10, 'cifar100': 100
}

PASS = '✓'
FAIL = '✗'


def test_model(attn_type: str, dataset: str, batch_size: int = 8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    C, H, W = DATASET_SHAPES[dataset]
    num_classes = DATASET_CLASSES[dataset]

    model = build_model(dataset=dataset, attn_type=attn_type).to(device)
    model.train()

    imgs   = torch.randn(batch_size, C, H, W, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)

    # Forward pass
    t0     = time.perf_counter()
    logits = model(imgs)
    fwd_ms = (time.perf_counter() - t0) * 1000

    # Check output shape
    assert logits.shape == (batch_size, num_classes), \
        f"Expected ({batch_size}, {num_classes}), got {logits.shape}"

    # Backward pass
    loss = logits.mean()
    loss.backward()

    # Check all parameters have gradients
    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    assert len(no_grad) == 0, f"No gradient for: {no_grad}"

    return {
        'params':  model.count_parameters(),
        'fwd_ms':  fwd_ms,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sanity Check — Device: {device}")
    print("=" * 70)
    print(f"{'Model':<25} {'Dataset':<15} {'Params':>10} {'Fwd(ms)':>10} {'Status':>8}")
    print("-" * 70)

    all_passed = True
    for attn in ATTN_TYPES:
        for ds in DATASETS:
            try:
                info = test_model(attn, ds)
                print(f"{attn:<25} {ds:<15} {info['params']:>10,} "
                      f"{info['fwd_ms']:>10.1f}  {PASS}")
            except Exception as e:
                print(f"{attn:<25} {ds:<15} {'—':>10} {'—':>10}  {FAIL} {e}")
                all_passed = False

    print("=" * 70)
    if all_passed:
        print(f"\n{PASS} All checks passed! Ready to run experiments.\n")
        print("Next step:")
        print("  python run_experiments.py --epochs 100")
        print("\nFor a quick smoke test (5 epochs):")
        print("  python run_experiments.py --epochs 5")
    else:
        print(f"\n{FAIL} Some checks failed. Fix errors before running experiments.")


if __name__ == '__main__':
    main()
