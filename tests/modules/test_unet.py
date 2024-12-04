import torch

from senhance.models.unet.unet import UNET1d, UNET1dDims


@torch.inference_mode()
def test_unet():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    batch_size, t = 8, 64
    dims = UNET1dDims(
        in_dim=4,
        dim=32,
        t_dim=16,
        c_dim=4,
    )
    unet = UNET1d(dims).to(device)

    x = torch.randn(batch_size, dims.in_dim, t, device=device)
    x_cond = torch.randn_like(x)
    timestep = torch.rand((batch_size,), device=device)
    y = unet.forward(x_t=x, x_cond=x_cond, timestep=timestep)
    assert torch.is_tensor(y)
    assert y.shape == x.shape


if __name__ == "__main__":
    test_unet()
