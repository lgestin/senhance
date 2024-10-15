import torch
from dataclasses import dataclass, fields

from concurrent.futures import ThreadPoolExecutor


@dataclass
class Checkpoint:
    codec: str
    step: int
    best_loss: float
    model: dict[str, torch.Tensor]
    opt: dict[str, torch.Tensor]

    def __post_init__(self):
        self.executor = ThreadPoolExecutor(1)

    def save(self, path: str):
        def save():
            torch.save({k.name: getattr(self, k.name) for k in fields(self)}, path)

        self.executor.submit(save)
        return

    @classmethod
    def load(cls, path: str, map_location: str | torch.device = "cpu"):
        checkpoint = torch.load(path, map_location=map_location)
        checkpoint = cls(**checkpoint)
        return cls
