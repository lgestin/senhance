import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from senhance.data.audio import Audio
from senhance.data.augmentations.augmentations import (
    Augmentation,
    AugmentationParameters,
    BatchAugmentationParameters,
)
from senhance.data.augmentations.distributions import Distribution
from senhance.data.source import ArrowAudioSource


@dataclass(kw_only=True)
class BackgroundNoiseParameters(AugmentationParameters):
    noise_filepath: str
    noise: torch.FloatTensor
    snr: torch.FloatTensor
    clean_loudness: torch.FloatTensor
    noise_loudness: torch.FloatTensor
    gain: torch.FloatTensor


class BackgroundNoise(Augmentation):
    def __init__(
        self,
        noise_source: ArrowAudioSource,
        snr_distribution: Distribution,
        min_duration_s: float = 0.0,
        name: str = "background_noise",
        p: float = 1.0,
    ):
        super().__init__(name=name, p=p)
        self.data_folder = noise_source.arrow_file.parent
        self.noise_source = noise_source
        self.snr_distribution = snr_distribution

    def _sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator = None,
    ) -> BackgroundNoiseParameters:
        i = int(
            torch.randint(
                0, len(self.noise_source), size=(1,), generator=generator, device="cpu"
            )
        )
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

        snr = self.snr_distribution.sample(generator=generator)
        clean_loudness = torch.as_tensor(audio.loudness).to(device=audio.device)
        noise_loudness = torch.as_tensor(noise.loudness).to(device=noise.device)
        noise_loudness = noise_loudness.clamp(min=-43)

        gain = clean_loudness - noise_loudness - snr
        gain = torch.exp(math.log(10) / 20 * gain)

        return BackgroundNoiseParameters(
            noise_filepath=noise.filepath,
            noise=noise.waveform,
            snr=snr,
            clean_loudness=clean_loudness,
            noise_loudness=noise_loudness,
            gain=gain,
        )

    @torch.inference_mode()
    def _augment(
        self,
        waveform: torch.FloatTensor,
        parameters: BackgroundNoiseParameters | BatchAugmentationParameters,
    ) -> torch.FloatTensor:
        if isinstance(parameters, AugmentationParameters):
            parameters = parameters.collate([parameters])

        if parameters is None or (not torch.any(parameters.apply)):
            return waveform

        device = waveform.device
        noise = parameters.noise.to(device, non_blocking=True)
        apply = parameters.apply.to(device, non_blocking=True)

        clean_loudness = parameters.clean_loudness
        noise_loudness = parameters.noise_loudness
        snr = parameters.snr
        gain = parameters.gain

        # gain = clean_loudness - noise_loudness - snr
        # gain = torch.exp(math.log(10) / 20 * gain)
        gain = gain.view(-1, 1, 1)

        # Clamp gain to prevent extreme amplification (max 10x = 20dB boost)
        gain = gain.clamp(max=10.0)

        gain = gain.to(device)
        noise = gain * noise

        # Clamp final noise to prevent extreme values
        noise = noise.clamp(-10.0, 10.0)

        # Avoid in-place operation for multi-stream training
        result = waveform.clone()
        result[apply] = waveform[apply] + noise
        return result
