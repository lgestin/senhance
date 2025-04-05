import torch


class TimestepScheduler:
    def sample(self, batch_size: int, device: torch.device | str = "cpu"): ...


class LogNormTimestepScheduler(TimestepScheduler):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def sample(self, batch_size: int, device: torch.device | str = "cpu"):
        timestep = torch.randn((batch_size,), device=device)
        timestep = timestep * self.std + self.mean
        timestep = timestep.sigmoid()
        return timestep
