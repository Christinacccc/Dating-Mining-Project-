"""
attention.py
============
Implements four attention mechanisms for Vision Transformers:
  1. StandardAttention    — vanilla softmax self-attention, O(N^2)
  2. PerformerAttention   — FAVOR+ linear attention, O(N*r)
  3. CayleySTRINGAttention    — FAVOR+ + Cayley-based relative PE
  4. CirculantSTRINGAttention — FAVOR+ + Circulant-based relative PE

References
----------
- Choromanski et al. (2020): "Rethinking Attention with Performers"
- Schenck et al. (2025): "STRING: Structured Relative Positional Encoding for Performers"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─────────────────────────────────────────────────────────────────────────────
# 1. Standard Softmax Attention  (O(N^2 * d))
# ─────────────────────────────────────────────────────────────────────────────

class StandardAttention(nn.Module):
    """
    Multi-head self-attention with standard softmax kernel.
    Complexity: O(N^2 * d) time and memory.

    Args:
        dim        : token embedding dimension
        num_heads  : number of attention heads
        dropout    : attention dropout probability
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)  →  output: (B, N, D)
        """
        B, N, D = x.shape
        # Project to Q, K, V and split heads
        qkv = self.qkv(x)                                   # (B, N, 3D)
        qkv = rearrange(qkv, 'b n (three h d) -> three b h n d',
                        three=3, h=self.num_heads)           # (3, B, H, N, d)
        q, k, v = qkv.unbind(0)                             # each (B, H, N, d)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale       # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                                       # (B, H, N, d)
        out = rearrange(out, 'b h n d -> b n (h d)')        # (B, N, D)
        return self.proj(out)


# ─────────────────────────────────────────────────────────────────────────────
# 2. FAVOR+ Performer Attention  (O(N * r * d))
# ─────────────────────────────────────────────────────────────────────────────

def orthogonal_random_matrix(num_rows: int, num_cols: int,
                              device=None, dtype=None) -> torch.Tensor:
    """
    Generate a matrix of orthogonal random features.
    Blocks of square orthogonal matrices (from QR decomposition of Gaussian)
    are stacked to fill (num_rows, num_cols), then row-normalized.

    This reduces variance of the FAVOR+ estimator compared to i.i.d. Gaussians.
    See Choromanski et al. (2020) Appendix B.
    """
    num_full_blocks = num_rows // num_cols
    blocks = []
    for _ in range(num_full_blocks + 1):
        g = torch.randn(num_cols, num_cols, device=device, dtype=dtype)
        q, _ = torch.linalg.qr(g)            # (num_cols, num_cols) orthogonal
        blocks.append(q.T)
    raw = torch.cat(blocks, dim=0)[:num_rows]   # (num_rows, num_cols)

    # Scale each row so that E[||φ(x)||^2] ≈ d  (matches softmax kernel)
    norms = torch.randn(num_rows, device=device, dtype=dtype).norm()
    scaling = (num_cols ** 0.5) * torch.ones(num_rows, device=device, dtype=dtype)
    return raw * (scaling / (raw.norm(dim=1, keepdim=True) + 1e-8))


def favor_plus_map(x: torch.Tensor, projection: torch.Tensor,
                   kernel: str = 'softmax') -> torch.Tensor:
    """
    Apply FAVOR+ random feature map φ(x) to input x.

    For the softmax kernel:
        φ(x) = exp(-||x||²/2) * [exp(w_i·x)]   (positive features variant)
    This approximates exp(q·k^T) as φ(q)·φ(k)^T.

    For the relu kernel:
        φ(x) = relu(w_i·x + b_i)   (simpler, always positive)

    Args:
        x          : (B, H, N, d)  — queries or keys
        projection : (r, d)        — random projection matrix
        kernel     : 'softmax' | 'relu'

    Returns:
        φ(x) : (B, H, N, r)
    """
    # x: (B, H, N, d), projection: (r, d)
    projected = torch.einsum('bhnd,rd->bhnr', x, projection)  # (B, H, N, r)

    if kernel == 'softmax':
        # Positive random features for numerical stability
        # φ(x) = exp(w·x - ||x||²/2) / sqrt(r)
        x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True) / 2    # (B, H, N, 1)
        features = torch.exp(projected - x_norm_sq) / math.sqrt(projection.shape[0])
        # Add small epsilon to ensure strict positivity
        features = features + 1e-6
    elif kernel == 'relu':
        features = F.relu(projected) / math.sqrt(projection.shape[0])
        features = features + 1e-6
    else:
        raise ValueError(f"Unknown kernel: {kernel}. Choose 'softmax' or 'relu'.")

    return features


class PerformerAttention(nn.Module):
    """
    FAVOR+ Performer attention with O(N*r) complexity.

    The key identity that enables linear attention:
        softmax(QK^T)V  ≈  φ(Q) [φ(K)^T V]

    By computing [φ(K)^T V] first (shape r×d), then multiplying by φ(Q),
    we avoid ever materializing the N×N attention matrix.

    Args:
        dim          : token embedding dimension
        num_heads    : number of attention heads
        num_features : number of random features r (controls speed/accuracy tradeoff)
        kernel       : 'softmax' (FAVOR+) or 'relu'
        dropout      : dropout on output projection
        redraw_steps : how often to resample random projections (-1 = never)
    """
    def __init__(self, dim: int, num_heads: int = 8, num_features: int = 128,
                 kernel: str = 'softmax', dropout: float = 0.0,
                 redraw_steps: int = -1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads    = num_heads
        self.head_dim     = dim // num_heads
        self.num_features = num_features
        self.kernel       = kernel
        self.redraw_steps = redraw_steps
        self._step        = 0

        self.qkv  = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

        # Random projection matrix — registered as buffer (not a parameter)
        self.register_buffer(
            'projection',
            orthogonal_random_matrix(num_features, self.head_dim)
        )

    def _maybe_redraw(self):
        """Periodically resample random features to reduce bias."""
        if self.redraw_steps > 0:
            self._step += 1
            if self._step % self.redraw_steps == 0:
                new_proj = orthogonal_random_matrix(
                    self.num_features, self.head_dim,
                    device=self.projection.device,
                    dtype=self.projection.dtype
                )
                self.projection.copy_(new_proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) → (B, N, D)"""
        B, N, D = x.shape
        self._maybe_redraw()

        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b n (three h d) -> three b h n d',
                        three=3, h=self.num_heads)
        q, k, v = qkv.unbind(0)   # each (B, H, N, d)

        # Apply FAVOR+ feature map
        q_feat = favor_plus_map(q, self.projection, self.kernel)  # (B, H, N, r)
        k_feat = favor_plus_map(k, self.projection, self.kernel)  # (B, H, N, r)

        # Linear attention: φ(Q) · [φ(K)^T · V]
        # Step 1: kv = φ(K)^T · V  →  (B, H, r, d)
        kv = torch.einsum('bhnr,bhnd->bhrd', k_feat, v)
        # Step 2: out = φ(Q) · kv  →  (B, H, N, d)
        out = torch.einsum('bhnr,bhrd->bhnd', q_feat, kv)

        # Normalize (denominator = φ(Q) · [φ(K)^T · 1])
        k_sum = k_feat.sum(dim=2)                                  # (B, H, r)
        denom = torch.einsum('bhnr,bhr->bhn', q_feat, k_sum)      # (B, H, N)
        out   = out / (denom.unsqueeze(-1) + 1e-6)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.drop(self.proj(out))


# ─────────────────────────────────────────────────────────────────────────────
# 3. STRING: Structured Relative Positional Encoding for Performers
# ─────────────────────────────────────────────────────────────────────────────

def cayley_transform(A: torch.Tensor) -> torch.Tensor:
    """
    Cayley transform: maps skew-symmetric matrix A → orthogonal matrix R.
        R = (I - A)(I + A)^{-1}

    Properties:
    - R is orthogonal (R R^T = I), so it represents a rotation
    - Maps relative positions to the unit sphere geometry
    - Numerically stable for small-to-moderate d

    Args:
        A: (..., d, d) skew-symmetric matrix (A = -A^T)
    Returns:
        R: (..., d, d) orthogonal matrix
    """
    d = A.shape[-1]
    I = torch.eye(d, device=A.device, dtype=A.dtype).expand_as(A)
    return torch.linalg.solve(I + A, I - A)


class CayleySTRINGAttention(nn.Module):
    """
    Performer attention with Cayley-STRING relative positional encoding.

    Core idea (Schenck et al., 2025):
    Instead of absolute positional embeddings added to tokens, we encode
    the *relative position* between token i and token j by rotating the
    query/key feature maps using a position-dependent orthogonal matrix R(i-j).

    The Cayley parameterization ensures R is orthogonal (a rotation), which:
    - Preserves the inner product structure needed by FAVOR+
    - Allows efficient O(N) computation (no pairwise N×N matrix)

    Implementation:
    - Learn a skew-symmetric parameter matrix A per head
    - Derive R_Δ = Cayley(Δ · A) for each relative position offset Δ
    - Rotate feature maps: φ_i → R_{i} φ_i before computing linear attention
    - This encodes geometry while keeping O(N·r) complexity

    Args:
        dim          : token embedding dimension
        num_heads    : number of attention heads
        num_features : random features r
        max_seq_len  : maximum sequence length (number of patches + 1)
        kernel       : FAVOR+ kernel type
        dropout      : output dropout
    """
    def __init__(self, dim: int, num_heads: int = 8, num_features: int = 128,
                 max_seq_len: int = 65, kernel: str = 'softmax',
                 dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads    = num_heads
        self.head_dim     = dim // num_heads
        self.num_features = num_features
        self.kernel       = kernel
        self.max_seq_len  = max_seq_len

        self.qkv  = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

        # Random projection (shared across heads, as in standard Performer)
        self.register_buffer(
            'projection',
            orthogonal_random_matrix(num_features, self.head_dim)
        )

        # Learnable skew-symmetric parameters for Cayley transform
        # One per head, upper-triangle only (d*(d-1)/2 free parameters)
        d = self.head_dim
        self.skew_param = nn.Parameter(
            torch.zeros(num_heads, d, d)
        )

    def _get_skew_symmetric(self) -> torch.Tensor:
        """Enforce skew-symmetry: A = (P - P^T) / 2"""
        P = self.skew_param
        return (P - P.transpose(-1, -2)) / 2.0   # (H, d, d)

    def _build_rotation_matrices(self, seq_len: int) -> torch.Tensor:
        """
        Build rotation matrices R_Δ for each absolute position index.

        We use absolute position index i to define a rotation:
            R_i = Cayley(i * A / max_seq_len)

        This means:
        - Position 0 → identity rotation
        - Position N → full rotation R_N
        - Relative geometry is captured implicitly through how Q and K
          are rotated by their respective position matrices

        Returns: (H, seq_len, d, d) — one rotation matrix per head per position
        """
        A = self._get_skew_symmetric()            # (H, d, d)
        positions = torch.arange(seq_len, device=A.device, dtype=A.dtype)
        # Scale: shape (seq_len, H, d, d)
        scaled_A = positions.view(-1, 1, 1, 1) * A.unsqueeze(0) / self.max_seq_len
        # Cayley transform for each (position, head)
        H, d = self.num_heads, self.head_dim
        scaled_A_flat = scaled_A.view(-1, d, d)      # (seq_len*H, d, d)
        R_flat = cayley_transform(scaled_A_flat)      # (seq_len*H, d, d)
        R = R_flat.view(seq_len, H, d, d)            # (seq_len, H, d, d)
        return R.permute(1, 0, 2, 3)                  # (H, seq_len, d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) → (B, N, D)"""
        B, N, D = x.shape

        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b n (three h d) -> three b h n d',
                        three=3, h=self.num_heads)
        q, k, v = qkv.unbind(0)   # each (B, H, N, d)

        # Build rotation matrices for this sequence length
        R = self._build_rotation_matrices(N)          # (H, N, d, d)

        # Rotate queries and keys by their position-specific rotation
        # q_rot[h,n] = R[h,n] @ q[h,n]
        q_rot = torch.einsum('hnde,bhne->bhnd', R, q)
        k_rot = torch.einsum('hnde,bhne->bhnd', R, k)

        # FAVOR+ on rotated features
        q_feat = favor_plus_map(q_rot, self.projection, self.kernel)
        k_feat = favor_plus_map(k_rot, self.projection, self.kernel)

        # Linear attention
        kv    = torch.einsum('bhnr,bhnd->bhrd', k_feat, v)
        out   = torch.einsum('bhnr,bhrd->bhnd', q_feat, kv)
        k_sum = k_feat.sum(dim=2)
        denom = torch.einsum('bhnr,bhr->bhn', q_feat, k_sum)
        out   = out / (denom.unsqueeze(-1) + 1e-6)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.drop(self.proj(out))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Circulant-STRING Attention
# ─────────────────────────────────────────────────────────────────────────────

def build_circulant_matrix(c: torch.Tensor) -> torch.Tensor:
    """
    Build a circulant matrix from its first column vector c.

    A circulant matrix C has the property that each row is a cyclic
    right-shift of the previous row:
        C[i,j] = c[(j-i) mod d]

    Key property: C = F^{-1} diag(F c) F  where F is the DFT matrix.
    This means circulant matrices are diagonalized by the FFT — making
    matrix-vector products O(d log d) instead of O(d^2).

    For STRING, circulant matrices serve as efficient positional encodings
    that have a natural translation-equivariant structure.

    Args:
        c: (..., d) — first column of the circulant matrix
    Returns:
        C: (..., d, d) — the full circulant matrix
    """
    d = c.shape[-1]
    # Stack cyclic shifts
    idx = torch.arange(d, device=c.device)
    shift_idx = (idx.unsqueeze(0) - idx.unsqueeze(1)) % d   # (d, d)
    return c[..., shift_idx]   # (..., d, d)


class CirculantSTRINGAttention(nn.Module):
    """
    Performer attention with Circulant-STRING relative positional encoding.

    Uses circulant matrices as position encodings instead of Cayley-derived
    orthogonal matrices. Advantages over Cayley-STRING:
    - More parameter efficient: only d parameters per head (vs d*(d-1)/2)
    - Faster matrix-vector products via FFT: O(d log d) vs O(d^2)
    - Strong inductive bias for translation-equivariant (image) tasks

    The circulant matrix encodes relative positions in a structured way:
    each entry C[i,j] depends only on (i-j) mod d, making the encoding
    naturally periodic and shift-equivariant.

    Implementation:
    - Learn a vector c of length d per head (first column of circulant matrix)
    - Apply softplus to ensure positive entries (optional, for stability)
    - Build circulant matrix C from c
    - For each position i, the rotation is C^i (matrix power)
    - In practice, we use a continuous relaxation: position-scaled C

    Args:
        dim          : token embedding dimension
        num_heads    : number of attention heads
        num_features : random features r
        max_seq_len  : maximum sequence length
        kernel       : FAVOR+ kernel type
        dropout      : output dropout
    """
    def __init__(self, dim: int, num_heads: int = 8, num_features: int = 128,
                 max_seq_len: int = 65, kernel: str = 'softmax',
                 dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads    = num_heads
        self.head_dim     = dim // num_heads
        self.num_features = num_features
        self.kernel       = kernel
        self.max_seq_len  = max_seq_len

        self.qkv  = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

        self.register_buffer(
            'projection',
            orthogonal_random_matrix(num_features, self.head_dim)
        )

        # Learnable circulant first-column vector, one per head
        # Initialize near zero for stable training start
        self.circ_param = nn.Parameter(
            torch.randn(num_heads, self.head_dim) * 0.02
        )

    def _build_position_encodings(self, seq_len: int) -> torch.Tensor:
        """
        Build position-dependent encoding matrices using circulant structure.

        For each position i and head h, we compute:
            M_{h,i} = I + (i / max_seq_len) * C_h

        where C_h is the circulant matrix for head h.
        This is a first-order Taylor expansion of exp(i/L * C_h),
        which approximates a rotation while being simpler to compute.

        Returns: (H, seq_len, d, d)
        """
        d = self.head_dim
        H = self.num_heads
        # Build circulant matrices from learned parameters
        C = build_circulant_matrix(self.circ_param)    # (H, d, d)
        # Make C skew-symmetric so that M is approximately orthogonal
        C = (C - C.transpose(-1, -2)) / 2.0            # (H, d, d)

        positions = torch.arange(seq_len, device=C.device, dtype=C.dtype)
        scale     = positions / max(self.max_seq_len, 1)  # (seq_len,)

        I = torch.eye(d, device=C.device, dtype=C.dtype)  # (d, d)
        # M_i = I + scale_i * C  →  (seq_len, H, d, d)
        M = I.unsqueeze(0).unsqueeze(0) + \
            scale.view(-1, 1, 1, 1) * C.unsqueeze(0)    # (seq_len, H, d, d)

        return M.permute(1, 0, 2, 3)                     # (H, seq_len, d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) → (B, N, D)"""
        B, N, D = x.shape

        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b n (three h d) -> three b h n d',
                        three=3, h=self.num_heads)
        q, k, v = qkv.unbind(0)

        # Build circulant position encodings
        M = self._build_position_encodings(N)             # (H, N, d, d)

        # Apply position encoding to queries and keys
        q_enc = torch.einsum('hnde,bhne->bhnd', M, q)
        k_enc = torch.einsum('hnde,bhne->bhnd', M, k)

        # FAVOR+ linear attention
        q_feat = favor_plus_map(q_enc, self.projection, self.kernel)
        k_feat = favor_plus_map(k_enc, self.projection, self.kernel)

        kv    = torch.einsum('bhnr,bhnd->bhrd', k_feat, v)
        out   = torch.einsum('bhnr,bhrd->bhnd', q_feat, kv)
        k_sum = k_feat.sum(dim=2)
        denom = torch.einsum('bhnr,bhr->bhn', q_feat, k_sum)
        out   = out / (denom.unsqueeze(-1) + 1e-6)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.drop(self.proj(out))
