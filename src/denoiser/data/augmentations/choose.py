from dataclasses import dataclass
from collections import defaultdict

import torch

from denoiser.data.audio import Audio
from denoiser.data.augmentations.augmentations import (
    Augmentation,
    AugmentationParameters,
    BatchAugmentationParameters,
)


@dataclass(kw_only=True)
class ChooseParameters(AugmentationParameters):
    apply: torch.BoolTensor
    choice: torch.LongTensor
    params: dict[int, AugmentationParameters]

    def batch(
        self, parameters: list["ChooseParameters"]
    ) -> dict[AugmentationParameters, BatchAugmentationParameters]:
        return BatchChooseParameters(parameters)


class BatchChooseParameters(BatchAugmentationParameters):
    def __init__(self, parameters: list[ChooseParameters]):
        self._parameters = parameters

        apply = torch.empty((len(parameters),), dtype=torch.bool)
        choices = torch.empty((len(parameters),), dtype=torch.long)
        choices_params = defaultdict(list)
        for i, params in enumerate(parameters):
            apply[i] = params.apply
            choices[i] = params.choice
            choice = params.choice.item()
            choices_params[choice].append(params.params[choice])

        choices_params = {
            choice: BatchAugmentationParameters(params)
            for choice, params in choices_params.items()
        }
        self.apply = apply
        self.choices = choices
        self.params = choices_params


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
        generator: torch.Generator = None,
    ) -> ChooseParameters:
        apply = torch.rand(tuple(), generator=generator) <= self.p
        choice = torch.multinomial(self.weights, 1, generator=generator).squeeze(0)

        params = [
            AugmentationParameters(apply=torch.as_tensor(False))
            for _ in self.augmentations
        ]
        params[choice] = self.augmentations[choice.item()].sample_parameters(
            audio=audio, generator=generator
        )
        params[choice].apply = params[choice].apply & apply
        return ChooseParameters(apply=apply, choice=choice, params=params)

    def augment(
        self,
        waveform: torch.Tensor,
        parameters: ChooseParameters | BatchChooseParameters,
    ) -> torch.Tensor:
        if isinstance(parameters, AugmentationParameters):
            parameters = BatchChooseParameters([parameters])

        if not torch.any(parameters.apply):
            return waveform

        augmented = waveform.clone()
        for choice, params in parameters.params.items():
            choice_apply = parameters.choices == choice
            # params = [params for c in parameters.choices]
            augmented[choice_apply] = self.augmentations[choice].augment(
                augmented[choice_apply], parameters=params
            )
        return augmented
