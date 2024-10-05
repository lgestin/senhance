import torch


def truncated_normal(size: tuple, min_val: float, max_val: float) -> torch.Tensor:
    """
    approximation of a truncated normal distribution between min and max
    """
    normal = torch.randn(size)
    trunc_normal = torch.fmod(normal, 2)
    trunc_normal = (trunc_normal / 2 + 1) / 2
    trunc_normal = trunc_normal * (max_val - min_val)
    trunc_normal = trunc_normal + min_val
    return trunc_normal
