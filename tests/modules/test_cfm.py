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
    v_t, u_t = cfm.forward(x_0=x_0, x_1=x_1, timestep=t)

    assert torch.is_tensor(u_t) and torch.is_tensor(v_t)
    assert u_t.shape == v_t.shape
    assert u_t.shape == x_0.shape
    assert v_t.shape == x_0.shape


if __name__ == "__main__":
    test_cfm()
