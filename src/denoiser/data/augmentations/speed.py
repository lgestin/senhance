from dataclasses import dataclass

import torch
import torchaudio.functional as F

from denoiser.data.audio import Audio
from denoiser.data.augmentations.augmentations import (
    Augmentation,
    AugmentationParameters,
    BatchAugmentationParameters,
)


@dataclass(kw_only=True)
class SpeedParameters(AugmentationParameters):
    apply: torch.BoolTensor
    factor: torch.FloatTensor
    sample_rate: torch.FloatTensor


class Speed(Augmentation):
    def __init__(
        self,
        min_factor: float,
        max_factor: float,
        p: float = 1.0,
    ):
        super().__init__(p=p)

        self.min_factor = min_factor
        self.max_factor = max_factor

    def sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator = None,
    ) -> SpeedParameters:
        apply = torch.rand(tuple(), generator=generator) <= self.p
        factor = (
            torch.rand(tuple(), generator=generator)
            * (self.max_factor - self.min_factor)
            + self.min_factor
        )
        sample_rate = audio.sample_rate
        return SpeedParameters(apply=apply, factor=factor, sample_rate=sample_rate)

    @torch.inference_mode()
    def augment(
        self,
        waveform: torch.FloatTensor,
        parameters: SpeedParameters | BatchAugmentationParameters,
    ) -> torch.FloatTensor:
        if isinstance(parameters, AugmentationParameters):
            parameters = parameters.collate([parameters])

        if not torch.any(parameters.apply):
            return waveform

        assert torch.all(parameters.apply)
        sample_rate = parameters.sample_rate.unique().item()
        assert parameters.factor.unique().shape[0] == 1
        factor = parameters.factor.unique().item()

        augmented = waveform.clone()
        augmented = F.speed(waveform=augmented, orig_freq=sample_rate, factor=factor)[0]
        return augmented
