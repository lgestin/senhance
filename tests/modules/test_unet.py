import torch
import torch.nn as nn

from denoiser.models.unet.unet import UNET1d


def test_unet():
    batch_size = 8
    in_dim, dim, out_dim = 4, 32, 4
    unet = UNET1d(in_dim, dim, out_dim)

    x = torch.randn(batch_size, in_dim, 128)
    x_cond = torch.randn_like(x)
    timestep = torch.rand((batch_size,))
    y = unet.forward(x_t=x, x_cond=x_cond, timestep=timestep)
    assert torch.is_tensor(y)
    assert y.shape == x.shape


if __name__ == "__main__":
    test_unet()
