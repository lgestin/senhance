import torch
import torch.nn as nn
import torch.nn.attention.flex_attention
from einops import rearrange

from senhance.models.unet.magnitude_preserving import MPConv1d, SiLU, normalize


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
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

        self.qkv = MPConv1d(dim, dim * 3, 1)
        self.proj = nn.Sequential(
            MPConv1d(dim, 4 * dim, 1),
            SiLU(),
            MPConv1d(4 * dim, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "B (K H D) L -> K B H L D", K=3, H=self.num_heads)
        q, k, v = normalize(qkv, dim=-1)
        x = torch.nn.attention.flex_attention.flex_attention(q, k, v)
        x = rearrange(x, "B H L D -> B (H D) L")
        x = self.proj(x)
        return x


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
