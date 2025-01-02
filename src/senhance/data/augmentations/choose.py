import math
from collections import defaultdict
from dataclasses import dataclass

import torch

from senhance.data.audio import Audio
from senhance.data.augmentations.augmentations import (
    Augmentation,
    AugmentationParameters,
    BatchAugmentationParameters,
)


@dataclass(kw_only=True)
class ChooseParameters(AugmentationParameters):
    apply: torch.BoolTensor
    choice: torch.LongTensor
    params: dict[int, AugmentationParameters]

    def collate(
        self, parameters: list["ChooseParameters"]
    ) -> dict[AugmentationParameters, BatchAugmentationParameters]:
        return BatchChooseParameters(parameters)


class BatchChooseParameters(BatchAugmentationParameters):
    def collate_fields(self):
        parameters = self._parameters

        apply = torch.empty((len(parameters),), dtype=torch.bool)
        choices = torch.empty((len(parameters),), dtype=torch.long)
        choices_params = defaultdict(list)
        for i, params in enumerate(parameters):
            apply[i] = params.apply
            choices[i] = params.choice
            choice = params.choice.item()
            choices_params[choice].append(params.params)

        choices_params = {
            choice: params[0].collate(params)
            for choice, params in choices_params.items()
        }
        self.apply = apply
        self.choice = choices
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
            weights /= weights.sum()
        weights = weights.cumsum(0)

        assert weights.ndim == 1
        assert weights.shape[0] == n
        assert math.isclose(weights[-1].item(), 1.0, abs_tol=1e-6)
        self.weights = weights

        self.augmentations = augmentations

    def __getitem__(self, idx: int):
        return self.augmentations[idx]

    def __len__(self):
        return len(self.augmentations)

    def sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator = None,
    ) -> ChooseParameters:
        apply = torch.rand(tuple(), generator=generator) <= self.p
        choice = torch.searchsorted(
            self.weights, torch.rand(tuple(), generator=generator)
        )
        chosen = self.augmentations[choice.item()]
        chosen_params = chosen.sample_parameters(
            audio=audio, generator=generator
        )
        apply &= chosen_params.apply
        chosen_params.apply &= apply
        params = ChooseParameters(
            apply=apply,
            choice=choice,
            params=chosen_params,
        )
        return params

    def augment(
        self,
        waveform: torch.Tensor,
        parameters: ChooseParameters | BatchChooseParameters,
    ) -> torch.Tensor:
        if isinstance(parameters, AugmentationParameters):
            parameters = BatchChooseParameters([parameters])

        if not torch.any(parameters.apply):
            return waveform

        parameters.choice.to(waveform.device, non_blocking=True)
        for choice, params in parameters.params.items():
            apply = parameters.choice == choice
            waveform[apply] = self.augmentations[choice].augment(
                waveform[apply], parameters=params
            )
        return waveform
