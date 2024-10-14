from abc import abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from denoiser.models.unet.magnitude_preserving import SiLU
from denoiser.models.unet.magnitude_preserving import timestep_embedding
from denoiser.models.unet.attention import SelfAttention


class TimestepAwareModule(nn.Module):
    """Any module where forward() takes timestep embeddings as a second argument."""

    @abstractmethod
    def forward(self, x_t, t_emb):
        """Apply the module to `x_t` given `t_emb` timestep embeddings."""


class TimestepAwareSequential(nn.Sequential, TimestepAwareModule):
    """A sequential module that passes timestep embeddings to the children that support it as an
    extra input."""

    def forward(self, x_t, t_emb):
        for layer in self:
            if isinstance(layer, TimestepAwareModule):
                x_t = layer(x_t, t_emb)
            else:
                x_t = layer(x_t)
        return x_t


class ResnetBlock1d(TimestepAwareModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
        attn: bool = False,
    ):
        super().__init__()
        self.convs = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            SiLU(inplace=True),
            weight_norm(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    dilation=dilation,
                    padding=dilation,
                )
            ),
            nn.GroupNorm(32, out_channels),
            SiLU(inplace=True),
            weight_norm(
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                ),
            ),
        )
        if attn:
            attn = SelfAttention(out_channels)
        self.attn = attn

    def forward(self, x_t: torch.FloatTensor, t_emb: torch.FloatTensor):
        x_t = x_t + self.convs(x_t + t_emb)
        if self.attn:
            x_t = x_t + self.attn(x_t)
        return x_t


@dataclass
class UNET1dDims:
    in_dim: int
    dim: int
    out_dim: int
    n_layers: int


class UNET1d(nn.Module):
    def __init__(self, dims: UNET1dDims):
        super().__init__()
        self.dims = dims

        t_emb_dim = 4 * dims.dim
        self.t_emb = nn.Sequential(
            weight_norm(nn.Linear(dims.dim, t_emb_dim)),
            SiLU(inplace=True),
            weight_norm(nn.Linear(t_emb_dim, dims.dim)),
        )

        self.in_conv = weight_norm(nn.Conv1d(dims.in_dim, dims.dim, 1))
        encoder = [
            ResnetBlock1d(
                in_channels=dims.dim,
                out_channels=dims.dim,
                dilation=3**i,
                attn=j >= dims.n_layers - 1,
            )
            for i in range(2)
            for j in range(dims.n_layers)
        ]
        self.encoder = nn.ModuleList(encoder)

        decoder = [
            ResnetBlock1d(
                in_channels=dims.dim,
                out_channels=dims.dim,
                dilation=3 ** (2 - i),
                attn=j < 1,
            )
            for i in range(2)
            for j in range(dims.n_layers)
        ]
        self.decoder = nn.ModuleList(decoder)
        self.out_conv = nn.Sequential(
            SiLU(inplace=True),
            weight_norm(nn.Conv1d(dims.dim, dims.out_dim, 1)),
        )
        self.cond_conv = nn.Sequential(
            weight_norm(nn.Conv1d(dims.in_dim, dims.dim, 1)),
            SiLU(inplace=True),
            weight_norm(nn.Conv1d(dims.dim, dims.dim, 3, padding=1)),
            SiLU(inplace=True),
            weight_norm(nn.Conv1d(dims.dim, 2 * dims.n_layers * dims.dim, 1)),
        )

    def forward(
        self,
        x_t: torch.FloatTensor,
        x_cond: torch.LongTensor,
        timestep: torch.FloatTensor,
    ):
        t_emb = timestep_embedding(timestep, self.dims.dim)
        t_emb = t_emb.unsqueeze(-1).repeat(1, 1, x_t.size(-1))
        t_emb = self.t_emb(t_emb.transpose(1, 2)).transpose(1, 2)

        x_cond = self.cond_conv(x_cond).chunk(2 * self.dims.n_layers, dim=1)
        x_t = self.in_conv(x_t)

        skips = []
        enc_conds = x_cond[: self.dims.n_layers]
        dec_conds = x_cond[self.dims.n_layers :]
        for layer, enc_cond, dec_cond in zip(self.encoder, enc_conds, dec_conds):
            x = (x_t + enc_cond) / 2
            x_t = layer(x, t_emb=t_emb)
            skips += [x_t + dec_cond]

        x_t = self.decoder[0](x, t_emb=t_emb)
        for layer, skip in zip(self.decoder[1:], reversed(skips[:-1])):
            x_t = layer((x_t + skip) / 2, t_emb=t_emb)

        x_t = self.out_conv(x_t)
        return x_t
