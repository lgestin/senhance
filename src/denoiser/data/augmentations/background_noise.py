import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from denoiser.data.audio import Audio, AudioInfo
from denoiser.data.augmentations.augmentations import (
    Augmentation,
    BatchAugmentationParameters,
)
from denoiser.data.utils import truncated_normal


@dataclass
class BackgroundNoiseParameters:
    apply: torch.BoolTensor
    noise: torch.FloatTensor
    snr: torch.FloatTensor
    clean_loudness: torch.FloatTensor
    noise_loudness: torch.FloatTensor


class BackgroundNoise(Augmentation):
    def __init__(
        self,
        noise_index_path: str,
        min_snr: float,
        max_snr: float,
        min_duration_s: float = 0.0,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.data_folder = Path(noise_index_path).parent
        with open(noise_index_path, "r") as f:
            noise_index = json.load(f)
        noise_index = filter(
            lambda index: index["duration_s"] > min_duration_s, noise_index
        )
        self.index = list(noise_index)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def load_noise(self, index: dict):
        audioinfo = AudioInfo(**index)
        audioinfo.filepath = (self.data_folder / audioinfo.filepath).as_posix()
        noise = Audio.from_audioinfo(audioinfo)
        return noise

    def sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator = None,
    ) -> BackgroundNoiseParameters:
        apply = torch.rand(tuple(), generator=generator) <= self.p

        if apply:
            i = torch.randint(0, len(self.index), size=(1,), generator=generator).item()
            index = self.index[i]
            noise = self.load_noise(index)
            noise = noise.random_excerpt(
                duration_s=audio.duration_s,
                generator=generator,
            )
            noise = noise.mono().resample(audio.sample_rate)
        else:
            zeros = torch.zeros_like(audio.waveform)
            noise = Audio(waveform=zeros, sample_rate=audio.sample_rate)
            noise._loudness = -70.0

        snr = truncated_normal(tuple(), min_val=self.min_snr, max_val=self.max_snr)
        clean_loudness = torch.as_tensor(audio.loudness).to(device=audio.device)
        noise_loudness = torch.as_tensor(noise.loudness).to(device=noise.device)

        return BackgroundNoiseParameters(
            apply=apply,
            noise=noise.waveform,
            snr=snr,
            clean_loudness=clean_loudness,
            noise_loudness=noise_loudness,
        )

    def augment(
        self,
        waveform: torch.FloatTensor,
        parameters: BatchAugmentationParameters,
    ) -> torch.FloatTensor:

        if not torch.any(parameters.apply):
            return waveform

        clean_loudness = parameters.clean_loudness[parameters.apply]
        noise_loudness = parameters.noise_loudness[parameters.apply]
        snr = parameters.snr[parameters.apply]

        gain = clean_loudness - noise_loudness - snr
        gain = torch.exp(math.log(10) / 20 * gain)
        gain = gain.view(-1, 1, 1)

        augmented = waveform.clone()
        augmented[parameters.apply] = waveform[parameters.apply] + (
            gain * parameters.noise[parameters.apply]
        )
        return augmented