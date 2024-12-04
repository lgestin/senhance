import torch
from dataclasses import dataclass, asdict

from concurrent.futures import ThreadPoolExecutor
from senhance.models.unet.unet import UNET1dDims


@dataclass
class Checkpoint:
    codec: str
    step: int
    best_loss: float
    dims: UNET1dDims
    model: dict[str, torch.Tensor]
    opt: dict[str, torch.Tensor]

    def __post_init__(self):
        self.executor = ThreadPoolExecutor(1)

    def save(self, path: str):
        def save():
            torch.save(asdict(self), path)

        self.executor.submit(save)
        return

    @classmethod
    def load(cls, path: str, map_location: str | torch.device = "cpu"):
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        checkpoint = cls(**checkpoint)
        return checkpoint
