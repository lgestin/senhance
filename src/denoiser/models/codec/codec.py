import torch
import torch.nn as nn


class Codec(nn.Module):
    def __init__(self, sample_rate: int):
        super().__init__()
        self.sample_rate = sample_rate

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
