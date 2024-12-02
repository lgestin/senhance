from abc import abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from denoiser.models.unet.attention import SelfAttention
from denoiser.models.unet.magnitude_preserving import (
    MPConv1d,
    PixelNorm,
    SiLU,
    mp_cat,
    mp_sum,
    timestep_embedding,
)


class TimestepAwareModule(nn.Module):
    """Any module where forward() takes timestep embeddings as a second argument."""

    @abstractmethod
    def forward(self, x_t, emb):
        """Apply the module to `x_t` given `t_emb` timestep embeddings."""


class TimestepAwareSequential(nn.Sequential, TimestepAwareModule):
    """A sequential module that passes timestep embeddings to the children that support it as an
    extra input."""

    def forward(self, x_t, emb):
        for layer in self:
            if isinstance(layer, TimestepAwareModule):
                x_t = layer(x_t, emb)
            else:
                x_t = layer(x_t)
        return x_t


class Downsample(nn.Module):
    def __init__(self, dim: int, rate: int):
        super().__init__()
        self.down = MPConv1d(dim, dim, kernel_size=2 * rate, stride=rate, padding=1)

    def forward(self, x: torch.Tensor):
        return self.down(x)


class EncoderBlock(TimestepAwareModule):
    def __init__(self, dim: int, downsample: int = None, attn: bool = False):
        super().__init__()
        self.emb = MPConv1d(4 * dim, dim, 1)
        self.emb_gain = nn.Parameter(torch.zeros([]))

        if downsample:
            downsample = Downsample(dim, rate=downsample)

        self.downsample = downsample
        self.pixel_norm = PixelNorm()

        self.conv1 = nn.Sequential(
            SiLU(),
            MPConv1d(dim, dim, 3, dilation=3, padding=3),
            SiLU(),
            MPConv1d(dim, dim, 3, padding=1),
        )

        self.conv2 = nn.Sequential(
            SiLU(),
            MPConv1d(dim, dim, 3, padding=1),
        )

        if attn:
            attn = SelfAttention(dim)
        self.attn = attn

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        emb = self.emb_gain * self.emb(emb)
        if self.downsample:
            x = self.downsample(x)
        x = self.pixel_norm(x)
        x_skip = x

        x = self.conv1(x)
        x = x * (emb + 1)
        x = self.conv2(x)
        x = mp_sum(x, x_skip)
        if self.attn:
            x = mp_sum(x, self.attn(x))
        return x


class Upsample(nn.Module):
    def __init__(self, dim: int, rate: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(dim, dim, 2 * rate, stride=rate, padding=1)

    def forward(self, x: torch.Tensor):
        return self.up(x)


class DecoderBlock(TimestepAwareModule):
    def __init__(self, dim: int, upsample: int = None, attn: bool = False):
        super().__init__()
        self.emb = MPConv1d(4 * dim, dim, 1)
        self.emb_gain = nn.Parameter(torch.zeros([]))

        if upsample:
            upsample = Upsample(dim, rate=upsample)
        self.upsample = upsample

        self.conv1 = nn.Sequential(
            SiLU(),
            MPConv1d(2 * dim, dim, 3, dilation=3, padding=3),
            SiLU(),
            MPConv1d(dim, dim, 3, padding=1),
        )

        self.conv2 = nn.Sequential(
            SiLU(),
            MPConv1d(dim, dim, 3, padding=1),
        )

        self.skip_conv = MPConv1d(2 * dim, dim, 1)

        if attn:
            attn = SelfAttention(dim)
        self.attn = attn

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        emb = self.emb_gain * self.emb(emb)

        if self.upsample:
            x = self.upsample(x)
        x_skip = self.skip_conv(x)

        x = self.conv1(x)
        x = x * (emb + 1)
        x = self.conv2(x)
        x = mp_sum(x, x_skip)

        if self.attn:
            x = mp_sum(x, self.attn(x))
        return x


@dataclass
class UNET1dDims:
    in_dim: int
    dim: int
    t_dim: int
    c_dim: int


class UNET1d(nn.Module):
    def __init__(self, dims: UNET1dDims):
        super().__init__()
        self.dims = dims
        in_dim, dim = dims.in_dim, dims.dim
        emb_dim = 4 * dim

        self.t_emb = MPConv1d(dims.t_dim, emb_dim, 1)
        self.c_emb = MPConv1d(dims.c_dim, emb_dim, 1)
        self.silu_emb = SiLU()

        encoder = nn.ModuleList(
            [
                TimestepAwareSequential(MPConv1d(in_dim + 1, dim, 3, padding=1)),
                EncoderBlock(dim),
                EncoderBlock(dim),
                EncoderBlock(dim),
                EncoderBlock(dim, downsample=2),
                EncoderBlock(dim),
                EncoderBlock(dim),
                EncoderBlock(dim),
                EncoderBlock(dim, downsample=2),
                EncoderBlock(dim, attn=True),
                EncoderBlock(dim, attn=True),
                EncoderBlock(dim, attn=True),
                EncoderBlock(dim, downsample=2),
                EncoderBlock(dim, attn=True),
                EncoderBlock(dim, attn=True),
                EncoderBlock(dim, attn=True),
            ],
        )
        decoder = nn.ModuleList(
            [
                TimestepAwareSequential(
                    DecoderBlock(dim, attn=True),
                    DecoderBlock(dim),
                ),
                DecoderBlock(dim, attn=True),
                DecoderBlock(dim, attn=True),
                DecoderBlock(dim, attn=True),
                TimestepAwareSequential(
                    DecoderBlock(dim, attn=True),
                    DecoderBlock(dim, upsample=2),
                ),
                DecoderBlock(dim, attn=True),
                DecoderBlock(dim, attn=True),
                DecoderBlock(dim, attn=True),
                TimestepAwareSequential(
                    DecoderBlock(dim, attn=True),
                    DecoderBlock(dim, upsample=2),
                ),
                DecoderBlock(dim),
                DecoderBlock(dim),
                DecoderBlock(dim),
                TimestepAwareSequential(
                    DecoderBlock(dim),
                    DecoderBlock(dim, upsample=2),
                ),
                DecoderBlock(dim),
                DecoderBlock(dim),
                DecoderBlock(dim),
                TimestepAwareSequential(
                    DecoderBlock(dim),
                    DecoderBlock(dim, upsample=2),
                ),
                DecoderBlock(dim),
                DecoderBlock(dim),
                DecoderBlock(dim),
                TimestepAwareSequential(
                    DecoderBlock(dim),
                    MPConv1d(dim, in_dim, 3),
                ),
            ],
        )
        self.encoder = encoder
        self.decoder = decoder
        self.out_gain = nn.Parameter(torch.zeros([]))

    def forward(self, x_t: torch.FloatTensor, x_cond, timestep):
        c_emb = self.c_emb(x_cond)
        t_emb = timestep_embedding(timestep, self.dims.t_dim)
        t_emb = self.t_emb(t_emb.unsqueeze(-1)).expand_as(c_emb)
        emb = self.silu_emb(mp_sum(t_emb, c_emb))

        x_skip = x_t
        x = torch.cat([x_t, torch.ones_like(x_t[:, :1])], dim=1)

        embs = [emb]
        skips = []
        for layer in self.encoder:
            if hasattr(layer, "downsample") and layer.downsample is not None:
                emb = F.avg_pool1d(emb, 2)
                embs.append(emb)
            x = layer(x, emb)
            skips += [x]

        import ipdb

        ipdb.set_trace()
        emb = embs.pop()
        skips.pop()
        for layer, skip in zip(self.decoder, reversed(skips), strict=True):
            x = mp_cat(x, skip)
            if hasattr(layer, "upsample") and layer.upsample is not None:
                emb = embs.pop()
            x = layer(x, emb)

        x = self.out_gain * x
        x = x + x_skip
        return x
