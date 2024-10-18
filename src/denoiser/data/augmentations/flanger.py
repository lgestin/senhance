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
class FlangerParameters(AugmentationParameters):
    apply: torch.BoolTensor
    sample_rate: torch.FloatTensor


class Flanger(Augmentation):
    def __init__(
        self,
        delay: float = 0.0,
        depth: float = 2.0,
        regen: float = 0.0,
        width: float = 71.0,
        speed: float = 0.5,
        phase: float = 25.0,
        modulation: str = "sinusoidal",
        interpolation: str = "linear",
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.delay = delay
        self.depth = depth
        self.regen = regen
        self.width = width
        self.speed = speed
        self.phase = phase
        self.modulation = modulation
        self.interpolation = interpolation

    def sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator = None,
    ) -> FlangerParameters:
        apply = torch.rand(tuple(), generator=generator) <= self.p
        sample_rate = audio.sample_rate
        return FlangerParameters(apply=apply, sample_rate=sample_rate)

    @torch.inference_mode()
    def augment(
        self,
        waveform: torch.FloatTensor,
        parameters: FlangerParameters | BatchAugmentationParameters,
    ) -> torch.FloatTensor:
        if isinstance(parameters, AugmentationParameters):
            parameters = parameters.collate([parameters])

        if not torch.any(parameters.apply):
            return waveform

        device = waveform.device
        apply = parameters.apply
        augmented = waveform.clone()
        augmented[apply] = F.flanger(
            waveform=augmented[apply].cpu(),
            sample_rate=parameters.sample_rate[0],
            delay=self.delay,
            depth=self.depth,
            regen=self.regen,
            width=self.width,
            speed=self.speed,
            phase=self.phase,
            modulation=self.modulation,
            interpolation=self.interpolation,
        ).to(device)
        return augmented
