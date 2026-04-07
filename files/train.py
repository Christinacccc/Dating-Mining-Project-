"""
train.py
========
Training and evaluation loop with:
  - Cosine annealing LR schedule with warmup
  - Gradient clipping
  - Timing measurement (train speed, inference speed)
  - Structured result logging (CSV + JSON)
  - Checkpoint saving / loading
"""

import time
import json
import csv
import math
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Learning rate schedule: linear warmup + cosine decay
# ─────────────────────────────────────────────────────────────────────────────

def cosine_schedule_with_warmup(optimizer, warmup_epochs: int,
                                 total_epochs: int, steps_per_epoch: int):
    """
    Creates a schedule with:
        - Linear warmup from 0 to 1 over warmup_epochs * steps_per_epoch steps
        - Cosine decay from 1 to 0 over the remaining steps

    This is the standard schedule used in ViT training.
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps  = total_epochs  * steps_per_epoch

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / \
                   max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    """Tracks running average of a scalar (e.g. loss, accuracy)."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def accuracy(logits: torch.Tensor, targets: torch.Tensor, topk=(1,)):
    """Compute top-k accuracy for each k in topk."""
    maxk  = max(topk)
    batch = targets.size(0)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred    = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum()
        res.append((correct_k / batch).item())
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Training epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
    scaler=None,           # For AMP (mixed precision)
) -> Dict:
    """
    Runs one full training epoch.

    Returns dict with:
        loss        : mean cross-entropy loss
        acc1        : top-1 accuracy
        samples_per_sec : training throughput
        epoch_time  : total wall-clock time for epoch (seconds)
    """
    model.train()
    loss_m = AverageMeter()
    acc_m  = AverageMeter()
    total_samples = 0
    t0 = time.perf_counter()

    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), \
                       labels.to(device, non_blocking=True)
        batch_size = imgs.size(0)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()

        acc1 = accuracy(logits.detach(), labels, topk=(1,))[0]
        loss_m.update(loss.item(), batch_size)
        acc_m.update(acc1, batch_size)
        total_samples += batch_size

    epoch_time = time.perf_counter() - t0
    return {
        'loss':             loss_m.avg,
        'acc1':             acc_m.avg,
        'samples_per_sec':  total_samples / epoch_time,
        'epoch_time':       epoch_time,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Validation epoch
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict:
    """
    Evaluates model on the validation set.

    Returns dict with:
        loss            : mean cross-entropy loss
        acc1            : top-1 accuracy
        ms_per_sample   : mean inference latency (ms per sample)
    """
    model.eval()
    loss_m = AverageMeter()
    acc_m  = AverageMeter()
    total_samples = 0
    t0 = time.perf_counter()

    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), \
                       labels.to(device, non_blocking=True)
        batch_size = imgs.size(0)

        logits = model(imgs)
        loss   = criterion(logits, labels)
        acc1   = accuracy(logits, labels, topk=(1,))[0]

        loss_m.update(loss.item(), batch_size)
        acc_m.update(acc1, batch_size)
        total_samples += batch_size

    total_time   = time.perf_counter() - t0
    ms_per_sample = total_time * 1000 / total_samples

    return {
        'loss':           loss_m.avg,
        'acc1':           acc_m.avg,
        'ms_per_sample':  ms_per_sample,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader,
    val_loader,
    config: Dict,
    run_name: str,
    save_dir: str = './checkpoints',
) -> Dict:
    """
    Full training loop for one (model, dataset) experiment.

    Args:
        model        : the ViT model to train
        train_loader : training DataLoader
        val_loader   : validation DataLoader
        config       : hyperparameter dict (see train_config below)
        run_name     : unique identifier for this run (for saving)
        save_dir     : directory to save checkpoints and logs

    Returns:
        results dict with best validation accuracy and timing stats
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = model.to(device)

    os.makedirs(save_dir, exist_ok=True)
    log_path = Path(save_dir) / f'{run_name}_log.csv'

    # Optimizer: AdamW with weight decay (decoupled L2)
    # Exclude LayerNorm and bias from weight decay (standard ViT practice)
    decay_params     = [p for n, p in model.named_parameters()
                        if p.requires_grad and p.dim() >= 2]
    no_decay_params  = [p for n, p in model.named_parameters()
                        if p.requires_grad and p.dim() < 2]
    optimizer = AdamW([
        {'params': decay_params,    'weight_decay': config['weight_decay']},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=config['lr'])

    scheduler = cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs   = config['warmup_epochs'],
        total_epochs    = config['epochs'],
        steps_per_epoch = len(train_loader),
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.1))

    # Optional: automatic mixed precision (AMP) for faster GPU training
    scaler = torch.cuda.amp.GradScaler() if (
        config.get('amp', True) and device.type == 'cuda'
    ) else None

    best_acc1  = 0.0
    all_epochs = []

    # CSV log header
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_loss', 'train_acc1', 'val_loss', 'val_acc1',
            'train_samples_per_sec', 'val_ms_per_sample', 'lr'
        ])

    print(f"\n{'='*60}")
    print(f"Run: {run_name}")
    print(f"Device: {device} | Parameters: {model.count_parameters():,}")
    print(f"{'='*60}\n")

    for epoch in range(1, config['epochs'] + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, config.get('grad_clip', 1.0), scaler
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        is_best    = val_metrics['acc1'] > best_acc1
        if is_best:
            best_acc1 = val_metrics['acc1']
            torch.save({
                'epoch':      epoch,
                'state_dict': model.state_dict(),
                'best_acc1':  best_acc1,
                'config':     config,
            }, Path(save_dir) / f'{run_name}_best.pth')

        # Log epoch results
        row = {
            'epoch':               epoch,
            'train_loss':          train_metrics['loss'],
            'train_acc1':          train_metrics['acc1'],
            'val_loss':            val_metrics['loss'],
            'val_acc1':            val_metrics['acc1'],
            'train_samples_per_sec': train_metrics['samples_per_sec'],
            'val_ms_per_sample':   val_metrics['ms_per_sample'],
            'lr':                  current_lr,
        }
        all_epochs.append(row)

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([row[k] for k in [
                'epoch', 'train_loss', 'train_acc1', 'val_loss', 'val_acc1',
                'train_samples_per_sec', 'val_ms_per_sample', 'lr'
            ]])

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:>3}/{config['epochs']} | "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"Acc: {train_metrics['acc1']:.3f} | "
                f"Val Loss: {val_metrics['loss']:.4f} "
                f"Acc: {val_metrics['acc1']:.3f} "
                f"{'★ BEST' if is_best else ''}"
            )

    # Aggregate timing stats (average over last 10 epochs, skip warmup)
    recent = all_epochs[-10:] if len(all_epochs) >= 10 else all_epochs
    results = {
        'run_name':                run_name,
        'best_val_acc1':           best_acc1,
        'final_val_acc1':          all_epochs[-1]['val_acc1'],
        'avg_train_samples_per_sec': sum(r['train_samples_per_sec'] for r in recent) / len(recent),
        'avg_val_ms_per_sample':     sum(r['val_ms_per_sample']     for r in recent) / len(recent),
        'num_parameters':          model.count_parameters(),
        'log_path':                str(log_path),
    }

    # Save summary
    with open(Path(save_dir) / f'{run_name}_summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Best Val Acc: {best_acc1:.4f}")
    print(f"  Avg Train Speed: {results['avg_train_samples_per_sec']:.0f} samples/sec")
    print(f"  Avg Inference: {results['avg_val_ms_per_sample']:.2f} ms/sample\n")

    return results
