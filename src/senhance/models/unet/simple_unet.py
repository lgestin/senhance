from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from senhance.models.unet.magnitude_preserving import timestep_embedding
from senhance.models.unet.unet import TimestepAwareModule, TimestepAwareSequential


class Block(TimestepAwareModule):
    def __init__(self, dim: int):
        super().__init__()
        self.emb = nn.Conv1d(4 * dim, dim, 1)
        self.conv1 = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(dim, dim, 3, dilation=3, padding=3),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 3, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(dim, dim, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        emb = self.emb(emb)

        x_skip = x
        x = self.conv1(x)
        x = x * (emb + 1)
        x = self.conv2(x)
        return x + x_skip


class Downsample(nn.Module):
    def __init__(self, dim: int, rate: int):
        super().__init__()
        self.down = nn.Conv1d(dim, dim, kernel_size=2 * rate, stride=rate, padding=1)

    def forward(self, x: torch.Tensor):
        return self.down(x)


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


class Upsample(nn.Module):
    def __init__(self, dim: int, rate: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(dim, dim, 2 * rate, stride=rate, padding=1)

    def forward(self, x: torch.Tensor):
        return self.up(x)


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

        self.t_emb = nn.Sequential(nn.Conv1d(dims.t_dim, emb_dim, 1), nn.SiLU())

        encoder = nn.ModuleList(
            [
                TimestepAwareSequential(nn.Conv1d(in_dim, dim, 3, padding=1)),
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
                TimestepAwareSequential(
                    DecoderBlock(dim),
                    nn.Conv1d(dim, in_dim, 3, padding=1),
                ),
            ],
        )
        self.encoder = encoder
        self.bridger = bridger
        self.decoder = decoder

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
            x = x + skip
            x = layer(x, emb)

        x = x + x_skip
        return x
