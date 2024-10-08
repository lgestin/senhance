import torch
import torch.nn as nn

from denoiser.models.unet.unet import UNET1d
from denoiser.models.cfm.cfm import ConditionalFlowMatcher


def test_cfm():
    batch_size = 8
    dim = 32
    module = UNET1d(dim, dim, dim, n_layers=1)
    cfm = ConditionalFlowMatcher(module=module)

    x = torch.randn(batch_size, dim, 128)
    x_cond = torch.randn_like(x)
    t = torch.full((batch_size,), 0.5)
    v_t, u_t = cfm.forward(x_1=x, x_cond=x_cond, timestep=t)

    assert torch.is_tensor(u_t) and torch.is_tensor(v_t)
    assert u_t.shape == x.shape and v_t.shape == x.shape


if __name__ == "__main__":
    test_cfm()
