import torch
import torch.nn as nn


class Codec(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        raise NotImplementedError

    def encode(self, x: torch.Tensor):
        raise NotImplementedError

    def decode(self, x: torch.Tensor):
        raise NotImplementedError
