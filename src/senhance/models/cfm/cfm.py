import torch
import torch.nn as nn
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper


class WrappedModel(ModelWrapper):
    def forward(self, x, t):
        return self.model(x_t=x, timestep=t[None].expand(x.size(0)))


class ConditionalFlowMatcher(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.solver = ODESolver(velocity_model=WrappedModel(module))
        self.module = module
        self.path = AffineProbPath(scheduler=CondOTScheduler())

    @property
    def device(self) -> torch.device:
        """Get device of the model's parameters."""
        return next(self.module.parameters()).device

    def forward(
        self,
        x_0: torch.FloatTensor,
        x_1: torch.FloatTensor,
        timestep: torch.FloatTensor,
    ):
        # timestep = timestep.view(-1, 1, 1)
        # x_t = (1 - timestep) * x_0 + timestep * x_1
        # v_t = self.module(x_t=x_t, timestep=timestep[:, 0, 0])
        # u_t = x_1 - x_0
        path_sample = self.path.sample(x_0, x_1, timestep)
        u_t = self.module(x_t=path_sample.x_t, timestep=path_sample.t)
        return u_t, path_sample

    @torch.inference_mode()
    def sample(
        self,
        x_0: torch.FloatTensor,
        timesteps: torch.FloatTensor,
        method: str = "euler",
        return_intermediates: bool = False,
    ):
        samples = self.solver.sample(
            time_grid=timesteps,
            x_init=x_0,
            method=method,
            step_size=None,
            return_intermediates=return_intermediates,
        )
        return samples

    # @torch.inference_mode()
    # def sample(self, x_0: torch.FloatTensor, timesteps: list[float]):
    #     x_t = x_0
    #     for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
    #         timestep = torch.full(x_0.shape[:1], t_curr, device=x_0.device)
    #         v_t = self.module(x_t=x_t, timestep=timestep)
    #         x_t = x_t + (t_prev - t_curr) * v_t
    #     return x_t
