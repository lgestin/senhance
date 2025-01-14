from abc import abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from senhance.models.unet.magnitude_preserving import (
    MPConv1d,
    MPConvTranspose1d,
    PixelNorm,
    SelfAttention,
    SiLU,
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


class Block(TimestepAwareModule):
    def __init__(self, dim: int, attn: bool = False):
        super().__init__()
        self.pixnorm = PixelNorm()
        self.emb = MPConv1d(4 * dim, dim, 1)
        self.emb_gain = nn.Parameter(torch.zeros([]))
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
        emb = self.emb(emb, gain=self.emb_gain)

        x = self.pixnorm(x)
        x_skip = x
        x = self.conv1(x)
        x = x * (emb + 1)
        x = self.conv2(x)
        x = mp_sum(x, x_skip, t=0.3)
        if self.attn:
            x = mp_sum(x, self.attn(x), t=0.3)
        return x


class Downsample(nn.Module):
    def __init__(self, dim: int, rate: int):
        super().__init__()
        self.down = MPConv1d(
            dim, dim, kernel_size=2 * rate, stride=rate, padding=1
        )

    def forward(self, x: torch.Tensor):
        return self.down(x)


class EncoderBlock(Block):
    def __init__(self, dim: int, attn: bool = False, downsample: int = None):
        super().__init__(dim=dim, attn=attn)
        if downsample:
            downsample = Downsample(dim, rate=downsample)
        self.downsample = downsample

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        if self.downsample:
            x = self.downsample(x)
        return super().forward(x, emb)


class Upsample(nn.Module):
    def __init__(self, dim: int, rate: int):
        super().__init__()
        self.up = MPConvTranspose1d(dim, dim, 2 * rate, stride=rate, padding=1)

    def forward(self, x: torch.Tensor):
        return self.up(x)


class DecoderBlock(Block):
    def __init__(self, dim: int, attn: bool = False, upsample: int = None):
        super().__init__(dim=dim, attn=attn)
        if upsample:
            upsample = Upsample(dim, rate=upsample)
        self.upsample = upsample

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        if self.upsample:
            x = self.upsample(x)
        return super().forward(x, emb)


@dataclass
class UNET1dDims:
    in_dim: int
    dim: int
    t_dim: int


class UNET1d(nn.Module):
    def __init__(self, dims: UNET1dDims):
        super().__init__()
        self.dims = dims
        in_dim, dim = dims.in_dim, dims.dim
        emb_dim = 4 * dim

        self.t_emb = nn.Sequential(
            weight_norm(nn.Conv1d(dims.t_dim, emb_dim, 1)),
            SiLU(),
        )
        encoder = nn.ModuleList(
            [
                TimestepAwareSequential(
                    weight_norm(nn.Conv1d(in_dim, dim, 3, padding=1))
                ),
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
        bridger = TimestepAwareSequential(Block(dim, attn=True), Block(dim))
        decoder = nn.ModuleList(
            [
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
                DecoderBlock(dim),
            ],
        )
        self.encoder = encoder
        self.bridger = bridger
        self.decoder = decoder
        self.out_conv = weight_norm(nn.Conv1d(dim, in_dim, 3, padding=1))
        self.out_gain = nn.Parameter(torch.zeros([]))

    def forward(self, x_t: torch.FloatTensor, timestep):
        t_emb = timestep_embedding(timestep, self.dims.t_dim)
        emb = self.t_emb(t_emb.unsqueeze(-1))

        x_skip = x_t
        x = x_t

        skips = []
        for i, layer in enumerate(self.encoder):
            x = layer(x, emb)
            skips += [x]
        x = self.bridger(x, emb)
        for layer, skip in zip(self.decoder, reversed(skips), strict=True):
            x = mp_sum(x, skip)
            x = layer(x, emb)

        x = self.out_conv(x)
        x = mp_sum(x, x_skip)
        return x
