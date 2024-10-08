import torch
from dataclasses import dataclass

from denoiser.data.audio import Audio
from denoiser.data.augmentations.augmentations import (
    Augmentation,
    AugmentationParameters,
)


@dataclass
class ChainParameters:
    apply: torch.BoolTensor
    params: list[AugmentationParameters]


class Chain(Augmentation):
    def __init__(self, *augmentations: Augmentation, p: float = 1.0):
        super().__init__(p=p)
        self.augmentations = augmentations

    def sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator = None,
    ) -> ChainParameters:
        apply = torch.rand(tuple(), generator=generator) <= self.p

        augment_parameters = []
        for augmentation in self.augmentations:
            parameters = augmentation.sample_parameters(
                audio=audio, generator=generator
            )
            augment_parameters.append(parameters)

        return ChainParameters(apply=apply, params=augment_parameters)

    def augment(
        self,
        waveform: torch.Tensor,
        parameters: ChainParameters,
    ) -> torch.Tensor:
        if not torch.any(parameters.apply):
            return waveform

        for augmentation, params in zip(
            self.augmentations, parameters.params, strict=True
        ):
            waveform = augmentation.augment(waveform=waveform, parameters=params)
        return waveform
