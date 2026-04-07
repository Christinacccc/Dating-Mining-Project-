"""
data.py
=======
DataLoader factories for CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST.

Applies standard augmentations:
  - Training : random crop, horizontal flip, normalization
  - Validation: center crop (or resize), normalization only

All datasets are downloaded automatically to ./data/ on first run.
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


# Per-dataset normalization statistics (mean, std per channel)
DATASET_STATS = {
    'cifar10': dict(
        mean=(0.4914, 0.4822, 0.4465),
        std =(0.2470, 0.2435, 0.2616),
    ),
    'cifar100': dict(
        mean=(0.5071, 0.4867, 0.4408),
        std =(0.2675, 0.2565, 0.2761),
    ),
    'mnist': dict(
        mean=(0.1307,),
        std =(0.3081,),
    ),
    'fashion_mnist': dict(
        mean=(0.2860,),
        std =(0.3530,),
    ),
}


def get_transforms(dataset: str, train: bool):
    """
    Returns the appropriate transform pipeline.

    Training augmentations:
      CIFAR: RandomCrop(32, padding=4) + RandomHorizontalFlip
      MNIST/FashionMNIST: RandomCrop(28, padding=4)

    All: ToTensor + Normalize
    """
    stats = DATASET_STATS[dataset]
    mean, std = stats['mean'], stats['std']

    if train:
        if dataset in ('cifar10', 'cifar100'):
            return T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])
        else:  # MNIST, Fashion-MNIST
            return T.Compose([
                T.RandomCrop(28, padding=4),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


def get_dataloaders(
    dataset: str,
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """
    Build train and validation DataLoaders for the specified dataset.

    Args:
        dataset     : 'cifar10' | 'cifar100' | 'mnist' | 'fashion_mnist'
        data_dir    : directory to cache downloaded data
        batch_size  : training and validation batch size
        num_workers : parallel data-loading workers
        pin_memory  : pin memory for faster GPU transfer

    Returns:
        (train_loader, val_loader)
    """
    train_tf = get_transforms(dataset, train=True)
    val_tf   = get_transforms(dataset, train=False)

    # Dataset class lookup
    DS = {
        'cifar10':       (torchvision.datasets.CIFAR10,       True),
        'cifar100':      (torchvision.datasets.CIFAR100,      True),
        'mnist':         (torchvision.datasets.MNIST,         True),
        'fashion_mnist': (torchvision.datasets.FashionMNIST,  True),
    }
    ds_cls, use_train_flag = DS[dataset]

    train_ds = ds_cls(root=data_dir, train=True,  download=True, transform=train_tf)
    val_ds   = ds_cls(root=data_dir, train=False, download=True, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,      # Avoids batch-norm issues with tiny last batch
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,   # Larger batch for faster validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
