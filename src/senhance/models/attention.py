import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from senhance.models.magnitude_preserving import MPConv1d, SiLU


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-2, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v), v


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads

        self.qk_norm = QKNorm(dim=dim)
        self.qkv = MPConv1d(dim, dim * 3, 1)
        self.proj = nn.Sequential(
            MPConv1d(dim, 2 * dim, 1),
            SiLU(),
            MPConv1d(2 * dim, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B (K H D) L -> K B H L D", K=3, H=self.num_heads)
        q, k, v = self.qk_norm(q, k, v)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "B H L D -> B (H D) L")
        x = self.proj(x)
        return x


def precompute_freqs_cis(
    seq_len: int, dim: int, theta: float = 10000.0, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """Precompute RoPE frequency tensor for 1D sequences.

    Args:
        seq_len: Maximum sequence length
        dim: Dimension per head (must be even)
        theta: Base frequency for rotary embeddings
        device: Device to create tensor on

    Returns:
        Frequency tensor of shape (1, 1, seq_len, dim // 2, 2)
        Last dimension is [cos, sin] for each frequency
        Shape allows broadcasting with (B, H, L, D//2, 2)
    """
    assert dim % 2 == 0, "Dimension must be even for RoPE"

    # Create frequency bands: [0, 1, 2, ..., dim//2 - 1]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))

    # Create position indices: [0, 1, 2, ..., seq_len - 1]
    positions = torch.arange(seq_len, device=device)

    # Outer product to get all position-frequency pairs: (seq_len, dim // 2)
    freqs = torch.outer(positions, freqs)

    # Stack [cos, sin] pairs: (seq_len, dim // 2, 2)
    freqs_cis = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)

    # Add dimensions for broadcasting: (1, 1, seq_len, dim // 2, 2)
    return freqs_cis.unsqueeze(0).unsqueeze(0)


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        xq: Query tensor of shape (B, H, L, D)
        xk: Key tensor of shape (B, H, L, D)
        freqs_cis: Precomputed frequencies of shape (1, 1, L, D // 2, 2)

    Returns:
        Tuple of rotated (query, key) tensors with same shapes as input
    """
    # Reshape to separate even/odd dimensions: (B, H, L, D) -> (B, H, L, D//2, 2)
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # Apply rotation: [cos, -sin; sin, cos] * [x0; x1]
    # cos * x0 - sin * x1, sin * x0 + cos * x1
    xq_out = torch.stack(
        [
            freqs_cis[..., 0] * xq_[..., 0] - freqs_cis[..., 1] * xq_[..., 1],
            freqs_cis[..., 1] * xq_[..., 0] + freqs_cis[..., 0] * xq_[..., 1],
        ],
        dim=-1,
    )

    xk_out = torch.stack(
        [
            freqs_cis[..., 0] * xk_[..., 0] - freqs_cis[..., 1] * xk_[..., 1],
            freqs_cis[..., 1] * xk_[..., 0] + freqs_cis[..., 0] * xk_[..., 1],
        ],
        dim=-1,
    )

    # Flatten back to original shape
    xq_out = xq_out.flatten(-2).type_as(xq)
    xk_out = xk_out.flatten(-2).type_as(xk)

    return xq_out, xk_out
