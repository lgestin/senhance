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
    choice: torch.LongTensor
    params: dict[int, AugmentationParameters]

    def collate(
        self, parameters: list["ChooseParameters"]
    ) -> "BatchChooseParameters":
        return BatchChooseParameters(parameters)


class BatchChooseParameters(BatchAugmentationParameters):
    def collate_fields(self):
        parameters: list[ChooseParameters] = self._parameters

        apply = [param is not None for param in parameters]
        apply = torch.as_tensor(apply)

        parameters = [param for param in parameters if param]
        choices = [param.choice for param in parameters]
        choices = torch.as_tensor(choices, dtype=torch.long)
        choices_params = defaultdict(list)
        for i, params in enumerate(parameters):
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
        name: str = "choose",
        p: float = 1.0,
    ):
        super().__init__(name=name, p=p)
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

    def _sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator = None,
    ) -> ChooseParameters:
        choice = torch.searchsorted(
            self.weights, torch.rand(tuple(), generator=generator)
        )
        chosen = self.augmentations[choice.item()]
        chosen_params = chosen.sample_parameters(
            audio=audio, generator=generator
        )
        return ChooseParameters(choice=choice, params=chosen_params)

    def _augment(
        self,
        waveform: torch.Tensor,
        parameters: ChooseParameters | BatchChooseParameters,
    ) -> torch.Tensor:
        if isinstance(parameters, AugmentationParameters):
            parameters = BatchChooseParameters([parameters])

        if parameters is None or (not torch.any(parameters.apply)):
            return waveform

        parameters.choice.to(waveform.device, non_blocking=True)
        augmented = waveform[parameters.apply]
        for choice, params in parameters.params.items():
            apply = parameters.choice == choice
            augmented[apply] = self.augmentations[choice].augment(
                augmented[apply], parameters=params
            )
        waveform[parameters.apply] = augmented
        return waveform
