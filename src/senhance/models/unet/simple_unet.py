from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from senhance.models.unet.magnitude_preserving import timestep_embedding
from senhance.models.unet.unet import (
    Block,
    Downsample,
    SiLU,
    TimestepAwareSequential,
    Upsample,
    mp_sum,
)


class WNConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        weight_norm(self)


class EncoderBlock(Block):
    def __init__(self, dim: int, downsample: int = None):
        super().__init__(dim=dim)
        if downsample:
            downsample = Downsample(dim, rate=downsample)
        self.downsample = downsample

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        if self.downsample:
            x = self.downsample(x)
        return super().forward(x, emb)


class DecoderBlock(Block):
    def __init__(self, dim: int, upsample: int = None):
        super().__init__(dim=dim)
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

        self.t_emb = nn.Sequential(WNConv1d(dims.t_dim, emb_dim, 1), SiLU())
        encoder = nn.ModuleList(
            [
                TimestepAwareSequential(WNConv1d(in_dim, dim, 3, padding=1)),
                EncoderBlock(dim),
                EncoderBlock(dim),
                EncoderBlock(dim),
                EncoderBlock(dim, downsample=2),
                EncoderBlock(dim),
                EncoderBlock(dim),
                EncoderBlock(dim),
                EncoderBlock(dim, downsample=2),
                EncoderBlock(dim),
                EncoderBlock(dim),
                EncoderBlock(dim),
                EncoderBlock(dim, downsample=2),
                EncoderBlock(dim),
                EncoderBlock(dim),
                EncoderBlock(dim),
            ],
        )
        bridger = TimestepAwareSequential(Block(dim), Block(dim))
        decoder = nn.ModuleList(
            [
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
        self.out_conv = WNConv1d(dim, in_dim, 3, padding=1)
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
