import math
from dataclasses import dataclass

import torch

from senhance.data.audio import Audio
from senhance.data.augmentations.augmentations import (
    Augmentation,
    AugmentationParameters,
    BatchAugmentationParameters,
)


@dataclass(kw_only=True)
class RandomNoiseParameters(AugmentationParameters):
    noise: torch.FloatTensor
    amplitude: torch.FloatTensor
    f_decay: torch.FloatTensor
    sample_rate: int


class RandomNoise(Augmentation):
    def __init__(
        self,
        min_amplitude: float,
        max_amplitude: float,
        min_f_decay: float = -6,
        max_f_decay: float = 6,
        name: str = "random_noise",
        p: float = 1.0,
    ):
        super().__init__(name=name, p=p)

        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude
        self.min_f_decay = min_f_decay
        self.max_f_decay = max_f_decay

    def _sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator = None,
    ) -> RandomNoiseParameters:
        amplitude = (
            torch.rand(tuple(), generator=generator)
            * (self.max_amplitude - self.min_amplitude)
            + self.min_amplitude
        )
        f_decay = (
            torch.rand(tuple(), generator=generator)
            * (self.max_f_decay - self.min_f_decay)
            + self.min_f_decay
        )
        sample_rate = audio.sample_rate

        f = torch.fft.rfftfreq(64, sample_rate)
        f[0] = 1.0

        decay = torch.ones(64 // 2 + 1, dtype=torch.cfloat)
        decay = torch.sqrt(1 / f**f_decay)
        decay_ir = torch.fft.irfft(decay)
        noise = torch.randn(
            audio.waveform.shape,
            device=audio.waveform.device,
            dtype=audio.waveform.dtype,
            generator=generator,
        )
        noise = torch.nn.functional.conv1d(
            noise, decay_ir[None, None], padding=32
        )[..., : audio.waveform.shape[-1]]
        noise /= torch.sqrt(torch.mean(noise.pow(2)))

        return RandomNoiseParameters(
            amplitude=amplitude,
            noise=noise,
            f_decay=f_decay,
            sample_rate=sample_rate,
        )

    @torch.inference_mode()
    def _augment(
        self,
        waveform: torch.FloatTensor,
        parameters: RandomNoiseParameters | BatchAugmentationParameters,
    ) -> torch.FloatTensor:
        if isinstance(parameters, AugmentationParameters):
            parameters = parameters.collate([parameters])

        if parameters is None or (not torch.any(parameters.apply)):
            return waveform

        apply = parameters.apply
        noise = parameters.noise.to(waveform.device, non_blocking=True)
        amplitude = parameters.amplitude.view(-1, 1, 1).to(
            waveform.device, non_blocking=True
        )
        waveform[apply] += amplitude * noise
        return waveform
