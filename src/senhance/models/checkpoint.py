from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from senhance.models.unet.unet import UNET1dDims


@dataclass
class Checkpoint:
    codec: str
    step: int
    best_loss: float
    dims: UNET1dDims
    model: nn.Module
    opt: torch.optim.Optimizer
    scaler: torch.amp.GradScaler

    def __post_init__(self):
        self.executor = ThreadPoolExecutor(1)

    def save(self, path: str):
        state_dict = asdict(self)

        model = state_dict.pop("model")
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        state_dict["model"] = {
            k: v.to(device="cpu", non_blocking=True)
            for k, v in model.state_dict().items()
        }

        state_dict["opt"] = {
            k: v.to(device="cpu", non_blocking=True)
            if torch.is_tensor(v)
            else v
            for k, v in state_dict.pop("opt").state_dict().items()
        }
        state_dict["scaler"] = state_dict.pop("scaler").state_dict()

        def save():
            torch.save(state_dict, path)

        # self.executor.submit(save)
        save()

        return

    @classmethod
    def load(cls, path: str, map_location: str | torch.device = "cpu"):
        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )
        checkpoint = cls(**checkpoint)
        return checkpoint
