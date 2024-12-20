from dataclasses import dataclass

import torch
import torchaudio.functional as F

from senhance.data.audio import Audio
from senhance.data.augmentations.augmentations import (
    Augmentation,
    AugmentationParameters,
    BatchAugmentationParameters,
)


@dataclass(kw_only=True)
class DitherParameters(AugmentationParameters):
    apply: torch.BoolTensor
    density_function: str


class Dither(Augmentation):
    def __init__(self, density_function: str = "TPDF", p: float = 1.0):
        super().__init__(p=p)
        self.density_function = density_function

    def sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator = None,
    ) -> DitherParameters:
        apply = torch.rand(tuple(), generator=generator) <= self.p
        density_function = self.density_function
        return DitherParameters(apply=apply, density_function=density_function)

    @torch.inference_mode()
    def augment(
        self,
        waveform: torch.FloatTensor,
        parameters: DitherParameters | BatchAugmentationParameters,
    ) -> torch.FloatTensor:
        if isinstance(parameters, AugmentationParameters):
            parameters = parameters.collate([parameters])

        if not torch.any(parameters.apply):
            return waveform

        apply = parameters.apply
        density_function = parameters.density_function[0]

        waveform[apply] = F.dither(
            waveform=waveform[apply],
            density_function=density_function,
        )
        return waveform
