import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from senhance.data.audio import Audio, AudioInfo
from senhance.data.augmentations.augmentations import (
    Augmentation,
    AugmentationParameters,
    BatchAugmentationParameters,
)
from senhance.data.source import ArrowAudioSource
from senhance.data.utils import truncated_normal


@dataclass(kw_only=True)
class BackgroundNoiseParameters(AugmentationParameters):
    apply: torch.BoolTensor
    noise_filepath: str
    noise: torch.FloatTensor
    snr: torch.FloatTensor
    clean_loudness: torch.FloatTensor
    noise_loudness: torch.FloatTensor


class BackgroundNoise(Augmentation):
    def __init__(
        self,
        noise_source: ArrowAudioSource,
        min_snr: float,
        max_snr: float,
        min_duration_s: float = 0.0,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.data_folder = noise_source.arrow_file.parent
        self.noise_source = noise_source

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
            i = torch.randint(
                0, len(self.noise_source), size=(1,), generator=generator
            ).item()
            noise = self.noise_source[i]
            noise = noise.random_excerpt(
                duration_s=audio.duration_s,
                generator=generator,
            )
            noise = noise.mono().resample(audio.sample_rate)
            noise._waveform[..., : audio.waveform.shape[-1]]
            if noise.waveform.shape[-1] < audio.waveform.shape[-1]:
                pad = audio.waveform.shape[-1] - noise.waveform.shape[-1]
                noise._waveform = F.pad(noise._waveform, (0, pad))
        else:
            zeros = torch.zeros_like(audio.waveform)
            noise = Audio(
                filepath="",
                waveform=zeros,
                sample_rate=audio.sample_rate,
            )
            noise._loudness = -70.0

        snr = truncated_normal(
            tuple(), min_val=self.min_snr, max_val=self.max_snr
        )
        clean_loudness = torch.as_tensor(audio.loudness).to(device=audio.device)
        noise_loudness = torch.as_tensor(noise.loudness).to(device=noise.device)

        return BackgroundNoiseParameters(
            apply=apply,
            noise_filepath=noise.filepath,
            noise=noise.waveform,
            snr=snr,
            clean_loudness=clean_loudness,
            noise_loudness=noise_loudness,
        )

    @torch.inference_mode()
    def augment(
        self,
        waveform: torch.FloatTensor,
        parameters: BackgroundNoiseParameters | BatchAugmentationParameters,
    ) -> torch.FloatTensor:
        if isinstance(parameters, AugmentationParameters):
            parameters = parameters.collate([parameters])

        if not torch.any(parameters.apply):
            return waveform

        device = waveform.device
        noise = parameters.noise.to(device, non_blocking=True)
        apply = parameters.apply.to(device, non_blocking=True)

        clean_loudness = parameters.clean_loudness[parameters.apply]
        noise_loudness = parameters.noise_loudness[parameters.apply]
        snr = parameters.snr[parameters.apply]

        gain = clean_loudness - noise_loudness - snr
        gain = torch.exp(math.log(10) / 20 * gain)
        gain = gain.view(-1, 1, 1)
        gain = gain.to(device, non_blocking=True)

        waveform[apply] = waveform[apply] + (gain * noise[apply])
        return waveform
