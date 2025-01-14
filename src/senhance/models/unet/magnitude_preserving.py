import math

import torch
import torch.nn as nn
import torch.nn.attention.flex_attention
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.parametrizations import weight_norm


def normalize(x, dim: int = None, eps: float = 1e-8):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(
        x, dim=dim, keepdim=True, dtype=torch.float32
    )
    norm = torch.add(eps, norm, alpha=math.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / math.sqrt((1 - t) ** 2 + t**2)


def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = math.sqrt((Na + Nb) / ((1 - t) ** 2 + t**2))
    wa = C / math.sqrt(Na) * (1 - t)
    wb = C / math.sqrt(Nb) * t
    return torch.cat([wa * a, wb * b], dim=dim)


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(
            start=0, end=half, dtype=torch.float32, device=timesteps.device
        )
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return math.sqrt(2) * embedding


class SiLU(nn.SiLU):
    def forward(self, input):
        return super().forward(input) / 0.596


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        weight_norm(self)


class MPConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation: int = 1,
        stride: int = 1,
        groups: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size)
        )
        self.dilation = dilation
        self.stride = stride
        self.groups = groups
        self.padding = padding

    def forward(self, x, gain: float = 1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))  # forced weight normalization
        w = normalize(w)  # traditional weight normalization
        w = w * (gain / math.sqrt(w[0].numel()))  # magnitude-preserving scaling
        w = w.to(x.dtype)
        return F.conv1d(
            x,
            w,
            dilation=self.dilation,
            stride=self.stride,
            groups=self.groups,
            padding=self.padding,
        )


class MPConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation: int = 1,
        stride: int = 1,
        groups: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, kernel_size)
        )
        self.dilation = dilation
        self.stride = stride
        self.groups = groups
        self.padding = padding

    def forward(self, x, gain: float = 1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))  # forced weight normalization
        w = normalize(w)  # traditional weight normalization
        w = w * (gain / math.sqrt(w[1].numel()))  # magnitude-preserving scaling
        w = w.to(x.dtype)
        return F.conv_transpose1d(
            x,
            w,
            dilation=self.dilation,
            stride=self.stride,
            groups=self.groups,
            padding=self.padding,
        )


class PixelNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return normalize(x, dim=1)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = MPConv1d(dim, 3 * dim, 1)
        self.proj = MPConv1d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "B (K H D) L -> K B H L D", K=3, H=self.num_heads)
        q, k, v = normalize(qkv, dim=-1)
        x = torch.nn.attention.flex_attention.flex_attention(q, k, v)
        x = rearrange(x, "B H L D -> B (H D) L")
        x = self.proj(x)
        return x
