from dataclasses import dataclass

import torch

from denoiser.data.audio import Audio
from denoiser.data.augmentations.augmentations import (
    Augmentation,
    AugmentationParameters,
    BatchAugmentationParameters,
)


@dataclass(kw_only=True)
class GaussianNoiseParameters(AugmentationParameters):
    apply: torch.BoolTensor
    amplitude: torch.FloatTensor


class GaussianNoise(Augmentation):
    def __init__(
        self,
        min_amplitude: float,
        max_amplitude: float,
        p: float = 1.0,
    ):
        super().__init__(p=p)

        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator = None,
    ) -> GaussianNoiseParameters:
        apply = torch.rand(tuple(), generator=generator) <= self.p
        amplitude = (
            torch.rand(tuple(), generator=generator)
            * (self.max_amplitude - self.min_amplitude)
            + self.min_amplitude
        )
        return GaussianNoiseParameters(apply=apply, amplitude=amplitude)

    @torch.inference_mode()
    def augment(
        self,
        waveform: torch.FloatTensor,
        parameters: GaussianNoiseParameters | BatchAugmentationParameters,
    ) -> torch.FloatTensor:
        if isinstance(parameters, AugmentationParameters):
            parameters = parameters.batch([parameters])

        if not torch.any(parameters.apply):
            return waveform

        apply = parameters.apply
        augmented = waveform.clone()
        augmented[apply] = parameters.amplitude[apply].view(-1, 1, 1) * augmented[apply]
        return augmented
