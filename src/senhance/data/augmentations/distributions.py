from abc import ABC, abstractmethod

import torch


class Distribution(ABC):
    @abstractmethod
    def sample(
        self,
        size: tuple | None = None,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        raise NotImplementedError


class Uniform(Distribution):
    def __init__(self, min: float, max: float):
        super().__init__()
        self.min = min
        self.max = max

    def sample(
        self,
        size: tuple | None = None,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        if size is None:
            size = tuple()

        rand = torch.rand(size, generator=generator, device=device)
        rand = rand * (self.max - self.min) + self.min
        return rand


class Binomial(Distribution):
    def __init__(self, count: float, prob: float):
        super().__init__()
        self.count = float(count)
        self.prob = prob

    def sample(
        self,
        size: tuple | None = None,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        if size is None:
            size = tuple()

        binomial = torch.binomial(
            count=torch.as_tensor(self.count, device=device).view(size),
            prob=torch.as_tensor(self.prob, device=device).view(size),
            generator=generator,
        ).bool()
        return binomial


class Normal(Distribution):
    def __init__(self, mean: float, std: float):
        super().__init__()
        self.mean = mean
        self.std = std

    def sample(
        self,
        size: tuple | None = None,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        if size is None:
            size = tuple()
        randn = torch.randn(size, generator=generator, device=device)
        randn = randn * self.std + self.mean
        return randn


class TruncatedNormal(Distribution):
    def __init__(
        self,
        min: float,
        max: float,
        underlying_mean: float = 0.0,
        underlying_std: float = 1.0,
    ):
        super().__init__()
        self.min = min
        self.max = max
        self.mean = torch.as_tensor(underlying_mean)
        self.std = torch.as_tensor(underlying_std)

    def sample(
        self,
        size: tuple | None = None,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        if size is None:
            size = tuple()

        normal01 = torch.distributions.Normal(
            torch.tensor(0.0, device=device),
            torch.tensor(1.0, device=device),
        )

        # bounds in standard normal space
        alpha = (self.min - self.mean) / self.std
        beta = (self.max - self.mean) / self.std

        cdf_alpha = normal01.cdf(alpha)
        cdf_beta = normal01.cdf(beta)

        # uniform in [cdf_alpha, cdf_beta]
        u = torch.rand(size, device=device, generator=generator)
        # broadcasting: cdf_* (param_shape) broadcast over leading sample dims
        u = cdf_alpha + (cdf_beta - cdf_alpha) * u

        # inverse CDF -> truncated standard normal
        z = normal01.icdf(u)

        # scale & shift back
        return z * self.std + self.mean
