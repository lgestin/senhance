import torch
import torch.nn as nn

from senhance.models.unet.unet import TimestepAwareModule, TimestepAwareSequential


class Downsample(nn.Module):
    def __init__(self, dim: int, rate: int):
        super().__init__()
        self.down = nn.Conv1d(dim, dim, kernel_size=2 * rate, stride=rate, padding=1)

    def forward(self, x: torch.Tensor):
        return self.down(x)


class EncoderBlock(TimestepAwareModule):
    def __init__(self, dim: int, downsample: int = None):
        super().__init__()
        self.emb = nn.Conv1d(4 * dim, dim, 1)

        if downsample:
            downsample = Downsample(dim, rate=downsample)
        self.downsample = downsample

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
        if self.downsample:
            x = self.downsample(x)
        x_skip = x

        x = self.conv1(x)
        x = x * (emb + 1)
        x = self.conv2(x)
        return x + x_skip


class Upsample(nn.Module):
    def __init__(self, dim: int, rate: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(dim, dim, 2 * rate, stride=rate, padding=1)

    def forward(self, x: torch.Tensor):
        return self.up(x)


class DecoderBlock(TimestepAwareModule):
    def __init__(self, dim: int, upsample: int = None, attn: bool = False):
        super().__init__()
        self.emb = nn.Conv1d(4 * dim, dim, 1)
        self.emb_gain = nn.Parameter(torch.zeros([]))

        if upsample:
            upsample = Upsample(dim, rate=upsample)
        self.upsample = upsample

        self.conv1 = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(2 * dim, dim, 3, dilation=3, padding=3),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 3, padding=1),
        )

        self.conv2 = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(dim, dim, 3, padding=1),
        )

        self.skip_conv = nn.Conv1d(2 * dim, dim, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        emb = self.emb(emb)

        if self.upsample:
            x = self.upsample(x)
        x_skip = self.skip_conv(x)

        x = self.conv1(x)
        x = x * (emb + 1)
        x = self.conv2(x)
        return x + x_skip
