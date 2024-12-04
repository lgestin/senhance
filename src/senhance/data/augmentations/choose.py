from dataclasses import dataclass
from collections import defaultdict

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
            choices_params[choice].append(params.params[choice])

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

        assert weights.ndim == 1
        assert weights.shape[0] == n
        assert weights.sum().item() == 1.0
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
            apply = parameters.choice == choice
            apply = apply.to(augmented.device)
            augmented[apply] = self.augmentations[choice].augment(
                augmented[apply], parameters=params
            )
        return augmented
