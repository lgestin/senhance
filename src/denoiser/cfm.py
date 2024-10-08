import torch
import torch.nn as nn


class ConditionalFlowMatcher(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        sigma_0: float = 1.0,  # std N(0, I)
        sigma_1: float = 1e-7,  # std target
    ):
        super().__init__()
        self.module = module
        self.register_buffer("sigma_1", torch.as_tensor(sigma_1))
        self.register_buffer("sigma_0", torch.as_tensor(sigma_0))

    def sigma_t(self, t):
        sigma_t = self.sigma_0 - (self.sigma_0 - self.sigma_1) * t
        return sigma_t

    def mu_t(self, t: torch.FloatTensor, x: torch.Tensor):
        return t * x

    def u_t(self, t: torch.FloatTensor, x: torch.FloatTensor, x_1: torch.FloatTensor):
        u_t = (x_1 - (self.sigma_0 - self.sigma_1) * x) / self.sigma(t)
        return u_t

    def phi_t(self, t: torch.FloatTensor, x: torch.FloatTensor, x_1: torch.FloatTensor):
        phi_t = self.sigma_t(t) * x + t * x_1
        return phi_t

    def forward(
        self,
        t: torch.FloatTensor,
        x_1: torch.FloatTensor,
        x_cond=torch.LongTensor,
    ):
        x_0 = torch.randn_like(x_1) * self.sigma_0
        x_t = self.sigma_t(t) * x_0 + t * x_1
        v_t = self.module(t=t, x_t=x_t, x_cond=x_cond)

        u_t = x_1 - (self.sigma_0 - self.sigma_1) * x_0
        return v_t, u_t

    @torch.inference_mode
    def sample(
        self,
        x_0: torch.FloatTensor,
        x_cond: torch.FloatTensor,
        n_steps: int,
    ):
        dt = 1 / n_steps
        ts = torch.linspace(0, 1, n_steps, device=x_0.device)
        ts = ts[:, None].repeat(1, x_0.size(0))
        while ts.ndim < x_0.ndim + 1:
            ts = ts.unsqueeze(-1)

        x_ts = []
        x_t = x_0.clone()
        for t in ts:
            v_t = self.module(t=t, x_t=x_t, x_cond=x_cond)
            x_t += dt * v_t
            x_ts += [x_t.cpu().detach()]
        return x_t, x_ts
