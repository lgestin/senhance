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
class OverdriveParameters(AugmentationParameters):
    apply: torch.BoolTensor
    gain: torch.FloatTensor
    colour: torch.FloatTensor


class Overdrive(Augmentation):
    def __init__(
        self,
        min_gain: float,
        max_gain: float,
        min_colour: float,
        max_colour: float,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        assert 0 <= min_gain <= max_gain <= 100
        assert 0 <= min_colour <= max_colour <= 100
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.min_colour = min_colour
        self.max_colour = max_colour

    def sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator = None,
    ) -> OverdriveParameters:
        apply = torch.rand(tuple(), generator=generator) <= self.p
        gain = (
            torch.rand(tuple(), generator=generator) * (self.max_gain - self.min_gain)
            + self.min_gain
        )
        colour = (
            torch.rand(tuple(), generator=generator)
            * (self.max_colour - self.min_colour)
            + self.min_colour
        )
        return OverdriveParameters(apply=apply, gain=gain, colour=colour)

    @torch.inference_mode()
    def augment(
        self,
        waveform: torch.FloatTensor,
        parameters: OverdriveParameters | BatchAugmentationParameters,
    ) -> torch.FloatTensor:
        if isinstance(parameters, AugmentationParameters):
            parameters = parameters.collate([parameters])

        if not torch.any(parameters.apply):
            return waveform

        apply = parameters.apply
        augmented = []
        for wav, gain, colour in zip(
            waveform[apply],
            parameters.gain[apply],
            parameters.colour[apply],
        ):
            aug = F.overdrive(waveform=wav, gain=gain, colour=colour)
            augmented.append(aug)
        augmented = torch.stack(augmented)
        return augmented
