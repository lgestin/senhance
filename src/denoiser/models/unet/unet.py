import torch
import torch.nn as nn


class ResnetBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        self.convs = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                dilation=dilation,
                padding=dilation,
            ),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, x_t: torch.FloatTensor):
        x_t = x_t + self.convs(x_t)
        return x_t


class UNET1d(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.y_emb = None
        self.t_emb = nn.Linear(1, hidden_channels)
        self.in_conv = nn.Conv1d(in_channels, hidden_channels, 1)
        encoder = [
            ResnetBlock1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                dilation=1,
            ),
            ResnetBlock1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                dilation=3,
            ),
            # ResnetBlock1d(
            #     in_channels=hidden_channels,
            #     out_channels=hidden_channels,
            #     dilation=9,
            # ),
            ResnetBlock1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                dilation=1,
            ),
            ResnetBlock1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                dilation=3,
            ),
            # ResnetBlock1d(
            #     in_channels=hidden_channels,
            #     out_channels=hidden_channels,
            #     dilation=9,
            # ),
        ]
        self.encoder = nn.ModuleList(encoder)

        decoder = [
            # ResnetBlock1d(
            #     in_channels=hidden_channels,
            #     out_channels=hidden_channels,
            #     dilation=9,
            # ),
            ResnetBlock1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                dilation=3,
            ),
            ResnetBlock1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                dilation=1,
            ),
            # ResnetBlock1d(
            #     in_channels=hidden_channels,
            #     out_channels=hidden_channels,
            #     dilation=9,
            # ),
            ResnetBlock1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                dilation=3,
            ),
            ResnetBlock1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                dilation=1,
            ),
        ]
        self.decoder = nn.ModuleList(decoder)
        self.out_conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(hidden_channels, out_channels, 1),
        )
        self.cond_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, 8 * hidden_channels, 1),
        )

    def forward(
        self,
        t: torch.FloatTensor,
        x_t: torch.FloatTensor,
        x_cond=torch.LongTensor,
    ):
        x_t = self.in_conv(x_t)
        skips = []
        t = self.t_emb(t.squeeze(2)).unsqueeze(-1).repeat(1, 1, x_t.size(-1))
        x_cond = self.cond_conv(x_cond).chunk(8, dim=1)
        for layer, enc_cond, dec_cond in zip(self.encoder, x_cond[:4], x_cond[4:]):
            x = (x_t + t + enc_cond) / 3
            x_t = layer(x)
            skips += [x_t + dec_cond]

        x = x_t + t
        x_t = self.decoder[0](x)
        for layer, skip in zip(self.decoder[1:], reversed(skips[:-1])):
            x = x_t + t
            x_t = layer((x + skip) / 2)

        x_t = self.out_conv(x_t)
        return x_t
