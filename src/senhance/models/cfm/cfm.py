import torch
import torch.nn as nn


class ConditionalFlowMatcher(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(
        self,
        x_0: torch.FloatTensor,
        x_1: torch.FloatTensor,
        timestep: torch.FloatTensor,
    ):
        timestep = timestep.view(-1, 1, 1)
        x_t = (1 - timestep) * x_0 + timestep * x_1
        v_t = self.module(x_t=x_t, timestep=timestep[:, 0, 0])
        # u_t = (x_1 - x_t) / (1 - timestep + 1e-8)
        u_t = x_1 - x_0
        return v_t, u_t

    @torch.inference_mode()
    def sample(self, x_0: torch.FloatTensor, timesteps: list[float]):
        x_t = x_0
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            timestep = torch.full(x_0.shape[:1], t_curr, device=x_0.device)
            v_t = self.module(x_t=x_t, timestep=timestep)
            x_t = x_t + (t_prev - t_curr) * v_t
        return x_t
