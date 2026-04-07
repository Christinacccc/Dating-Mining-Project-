"""
model.py
========
Vision Transformer (ViT) backbone with pluggable attention mechanisms.

Architecture follows the original ViT paper (Dosovitskiy et al., 2020),
but with a configurable attention block so we can drop in:
  - StandardAttention
  - PerformerAttention
  - CayleySTRINGAttention
  - CirculantSTRINGAttention

Key design choices for small datasets (CIFAR/MNIST):
  - Smaller patch size (4×4) to get enough tokens
  - Fewer layers and heads than ViT-Base to avoid overfitting
  - Learnable absolute positional embedding (standard)
  - Pre-norm transformer blocks (more stable training)
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from attention import (
    StandardAttention,
    PerformerAttention,
    CayleySTRINGAttention,
    CirculantSTRINGAttention,
)


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    FFN(x) = Linear(GELU(Linear(x)))
    Hidden dim is typically 4× the embedding dim.
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Using pre-norm (LayerNorm before attention) rather than post-norm
    is more stable for training from scratch on small datasets.
    """
    def __init__(self, dim: int, attention_module: nn.Module,
                 mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = attention_module
        self.ff    = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Patch Embedding
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Splits image into non-overlapping patches and projects each to dim.

    For a 32×32 image with patch_size=4:
        num_patches = (32/4)^2 = 64 patches
        each patch is a 4×4×C = 48 (for RGB) dimensional vector

    Args:
        image_size  : height (= width) of image in pixels
        patch_size  : height (= width) of each patch
        in_channels : number of image channels (1=grayscale, 3=RGB)
        dim         : output embedding dimension
    """
    def __init__(self, image_size: int, patch_size: int,
                 in_channels: int, dim: int):
        super().__init__()
        assert image_size % patch_size == 0, \
            f"Image size {image_size} must be divisible by patch size {patch_size}"
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        self.projection = nn.Sequential(
            # Rearrange image into sequence of flattened patches
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return self.projection(x)


# ─────────────────────────────────────────────────────────────────────────────
# Full ViT Model
# ─────────────────────────────────────────────────────────────────────────────

ATTENTION_REGISTRY = {
    'standard':          StandardAttention,
    'performer':         PerformerAttention,
    'cayley_string':     CayleySTRINGAttention,
    'circulant_string':  CirculantSTRINGAttention,
}


class ViT(nn.Module):
    """
    Vision Transformer with configurable attention mechanism.

    Args:
        image_size    : image height = width (e.g. 32 for CIFAR, 28 for MNIST)
        patch_size    : patch height = width (e.g. 4)
        num_classes   : number of output classes
        dim           : token embedding dimension (model width)
        depth         : number of transformer blocks (model depth)
        heads         : number of attention heads
        mlp_dim       : hidden dimension of FFN (typically 4*dim)
        in_channels   : 1 (grayscale) or 3 (RGB)
        dropout       : dropout rate in FFN and attention
        attn_type     : one of 'standard', 'performer', 'cayley_string',
                        'circulant_string'
        num_features  : number of random features for Performer variants
        kernel        : FAVOR+ kernel ('softmax' or 'relu')
        pool          : 'cls' (use [CLS] token) or 'mean' (global average pool)
    """
    def __init__(
        self,
        image_size: int   = 32,
        patch_size: int   = 4,
        num_classes: int  = 10,
        dim: int          = 256,
        depth: int        = 6,
        heads: int        = 8,
        mlp_dim: int      = 512,
        in_channels: int  = 3,
        dropout: float    = 0.1,
        attn_type: str    = 'standard',
        num_features: int = 128,
        kernel: str       = 'softmax',
        pool: str         = 'cls',
    ):
        super().__init__()
        assert attn_type in ATTENTION_REGISTRY, \
            f"attn_type must be one of {list(ATTENTION_REGISTRY.keys())}"
        assert pool in ('cls', 'mean')

        self.attn_type = attn_type
        self.pool = pool

        # 1. Patch embedding
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, dim)
        num_patches = self.patch_embed.num_patches
        max_seq_len = num_patches + 1   # +1 for [CLS]

        # 2. [CLS] token and positional embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)
        self.pos_drop  = nn.Dropout(dropout)

        # 3. Build attention modules based on attn_type
        def make_attention():
            attn_cls = ATTENTION_REGISTRY[attn_type]
            if attn_type == 'standard':
                return attn_cls(dim=dim, num_heads=heads, dropout=dropout)
            elif attn_type == 'performer':
                return attn_cls(dim=dim, num_heads=heads,
                                num_features=num_features, kernel=kernel,
                                dropout=dropout)
            else:  # STRING variants
                return attn_cls(dim=dim, num_heads=heads,
                                num_features=num_features,
                                max_seq_len=max_seq_len,
                                kernel=kernel, dropout=dropout)

        # 4. Stack transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                attention_module=make_attention(),
                mlp_dim=mlp_dim,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        # 5. Final norm and classification head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Standard ViT weight initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Special init for positional embedding and cls token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (B, C, H, W) → logits: (B, num_classes)
        """
        B = img.shape[0]

        # Patch embedding: (B, N, D)
        x = self.patch_embed(img)

        # Prepend [CLS] token
        cls = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x   = torch.cat([cls, x], dim=1)          # (B, N+1, D)

        # Add positional embedding
        x = x + self.pos_embed[:, :x.shape[1]]
        x = self.pos_drop(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Pooling: use [CLS] token or global average
        if self.pool == 'cls':
            x = x[:, 0]            # (B, D)
        else:
            x = x.mean(dim=1)      # (B, D)

        return self.head(x)        # (B, num_classes)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(dataset: str, attn_type: str, **kwargs) -> ViT:
    """
    Convenience factory: build a ViT configured for a given dataset.

    Handles MNIST/Fashion-MNIST (28×28 grayscale) vs
    CIFAR-10/CIFAR-100 (32×32 RGB) automatically.

    Args:
        dataset   : 'mnist', 'fashion_mnist', 'cifar10', or 'cifar100'
        attn_type : attention mechanism to use
        **kwargs  : override any ViT hyperparameters

    Returns:
        Configured ViT model
    """
    DATASET_CONFIGS = {
        'mnist': dict(
            image_size=28, patch_size=4, num_classes=10, in_channels=1
        ),
        'fashion_mnist': dict(
            image_size=28, patch_size=4, num_classes=10, in_channels=1
        ),
        'cifar10': dict(
            image_size=32, patch_size=4, num_classes=10, in_channels=3
        ),
        'cifar100': dict(
            image_size=32, patch_size=4, num_classes=100, in_channels=3
        ),
    }
    assert dataset in DATASET_CONFIGS, \
        f"dataset must be one of {list(DATASET_CONFIGS.keys())}"

    # Default shared hyperparameters (can be overridden via kwargs)
    config = dict(
        dim=256,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        attn_type=attn_type,
        num_features=128,    # r: number of random features for Performer variants
        kernel='softmax',    # FAVOR+ kernel type
        pool='cls',
    )
    config.update(DATASET_CONFIGS[dataset])
    config.update(kwargs)

    return ViT(**config)
