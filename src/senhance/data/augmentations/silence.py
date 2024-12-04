from dataclasses import dataclass

import torch

from senhance.data.audio import Audio
from senhance.data.augmentations.augmentations import (
    Augmentation,
    AugmentationParameters,
    BatchAugmentationParameters,
)


@dataclass(kw_only=True)
class SilenceParameters(AugmentationParameters):
    apply: torch.BoolTensor


class Silence(Augmentation):
    def __init__(self, p: float = 1.0):
        super().__init__(p=p)

    def sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator = None,
    ) -> SilenceParameters:
        apply = torch.rand(tuple(), generator=generator) <= self.p
        return SilenceParameters(apply=apply)

    @torch.inference_mode()
    def augment(
        self,
        waveform: torch.FloatTensor,
        parameters: SilenceParameters | BatchAugmentationParameters,
    ) -> torch.FloatTensor:
        if isinstance(parameters, AugmentationParameters):
            parameters = parameters.collate([parameters])

        if not torch.any(parameters.apply):
            return waveform

        apply = parameters.apply
        augmented = waveform.clone()
        augmented[apply] = 0.0 * augmented[apply]
        return augmented
