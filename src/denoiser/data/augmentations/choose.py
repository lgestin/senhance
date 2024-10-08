import torch

from dataclasses import dataclass

from denoiser.data.audio import Audio
from denoiser.data.augmentations.augmentations import (
    Augmentation,
    AugmentationParameters,
)


@dataclass
class ChooseParameters:
    apply: torch.BoolTensor
    params: list[AugmentationParameters | None]


class Choose(Augmentation):
    def __init__(
        self,
        *augmentations: Augmentation,
        weights: list[float] | torch.FloatTensor = None,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        n = len(augmentations)
        if weights is None:
            weights = torch.full((n,), 1 / n)
        if not torch.is_tensor(weights):
            weights = torch.as_tensor(weights)

        assert weights.ndim == 1
        assert weights.shape[0] == n
        assert weights.sum().item() == 1.0
        self.weights = weights

        self.augmentations = augmentations

    def sample_parameters(
        self,
        audio: Audio,
        generator: AugmentationParameters = None,
    ) -> list[AugmentationParameters | None]:
        apply = torch.rand(tuple(), generator=generator) <= self.p
        choice = torch.multinomial(self.weights, 1).item()

        params = [None for _ in self.augmentations]
        params[choice] = self.augmentations[choice].sample_parameters(
            audio=audio, generator=generator
        )
        return ChooseParameters(apply=apply, params=params)

    def augment(
        self,
        waveform: torch.Tensor,
        parameters: ChooseParameters,
    ) -> torch.Tensor:
        if not torch.any(parameters.apply):
            return waveform

        for augmentation, params in zip(self.augmentations, parameters.params):
            if params is None:
                continue
            x = augmentation.augment(waveform=waveform, parameters=params)
        return x
