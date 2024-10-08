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

    def sigma_t(self, timestep: torch.FloatTensor):
        sigma_t = self.sigma_0 - (self.sigma_0 - self.sigma_1) * timestep
        return sigma_t

    def mu_t(self, x: torch.Tensor, timestep: torch.FloatTensor):
        return timestep * x

    def u_t(
        self,
        x: torch.FloatTensor,
        x_1: torch.FloatTensor,
        timestep: torch.FloatTensor,
    ):
        u_t = (x_1 - (self.sigma_0 - self.sigma_1) * x) / self.sigma_t(timestep)
        return u_t

    def phi_t(
        self,
        x: torch.FloatTensor,
        x_1: torch.FloatTensor,
        timestep: torch.FloatTensor,
    ):
        phi_t = self.sigma_t(timestep) * x + timestep * x_1
        return phi_t

    def forward(
        self,
        x_1: torch.FloatTensor,
        x_cond: torch.LongTensor,
        timestep: torch.FloatTensor,
    ):
        timestep = timestep.view(-1, 1, 1)
        x_0 = torch.randn_like(x_1) * self.sigma_0
        x_t = self.sigma_t(timestep) * x_0 + timestep * x_1
        v_t = self.module(x_t=x_t, x_cond=x_cond, timestep=timestep[:, 0, 0])
        u_t = x_1 - (self.sigma_0 - self.sigma_1) * x_0
        return v_t, u_t

    @torch.inference_mode()
    def sample(
        self,
        x_0: torch.FloatTensor,
        x_cond: torch.FloatTensor,
        timesteps: list[float],
    ):
        x_t = x_0
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            timestep = torch.full(x_0.shape[:1], t_curr, device=x_0.device)
            v_t = self.module(x_t=x_t, x_cond=x_cond, timestep=timestep)
            x_t = x_t + (t_prev - t_curr) * v_t
        return x_t
