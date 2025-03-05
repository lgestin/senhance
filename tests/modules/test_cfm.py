import torch

from senhance.models.cfm.cfm import ConditionalFlowMatcher
from senhance.models.unet.simple_unet import UNET1d, UNET1dDims


@torch.inference_mode()
def test_cfm():
    batch_size = 8
    dim = 32
    dims = UNET1dDims(in_dim=dim, dim=dim, t_dim=dim)
    module = UNET1d(dims)
    cfm = ConditionalFlowMatcher(module=module)

    x_0 = torch.randn(batch_size, dim, 128)
    x_1 = torch.randn_like(x_0)
    t = torch.full((batch_size,), 0.5)
    u_t, path_sample = cfm.forward(x_0=x_0, x_1=x_1, timestep=t)

    assert torch.is_tensor(u_t)
    assert torch.is_tensor(path_sample.x_t)
    assert torch.is_tensor(path_sample.dx_t)
    assert u_t.shape == path_sample.dx_t.shape
    assert u_t.shape == x_0.shape
    assert path_sample.x_t.shape == x_0.shape


if __name__ == "__main__":
    test_cfm()
