import torch
import torch.nn as nn


class Codec(nn.Module):
    def __init__(self, dim: int, sample_rate: int, resolution_hz: int):
        super().__init__()
        self.dim = dim
        self.sample_rate = sample_rate
        self.resolution_hz = resolution_hz

    def forward(self, x: torch.Tensor):
        raise NotImplementedError

    def encode(self, x: torch.Tensor):
        raise NotImplementedError

    def decode(self, x: torch.Tensor):
        raise NotImplementedError

    def reconstruct(self, x: torch.Tensor):
        raise NotImplementedError

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        return self
