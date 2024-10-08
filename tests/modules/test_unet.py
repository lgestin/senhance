import torch

from denoiser.models.unet.unet import UNET1d, UNET1dDims


def test_unet():
    batch_size = 8
    in_dim, dim, out_dim = 4, 32, 4
    dims = UNET1dDims(in_dim=in_dim, dim=dim, out_dim=out_dim, n_layers=1)
    unet = UNET1d(dims)

    x = torch.randn(batch_size, in_dim, 128)
    x_cond = torch.randn_like(x)
    timestep = torch.rand((batch_size,))
    y = unet.forward(x_t=x, x_cond=x_cond, timestep=timestep)
    assert torch.is_tensor(y)
    assert y.shape == x.shape


if __name__ == "__main__":
    test_unet()
