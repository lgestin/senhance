from dataclasses import dataclass

import torch

from senhance.data.audio import Audio
from senhance.data.augmentations.augmentations import (
    Augmentation,
    AugmentationParameters,
    BatchAugmentationParameters,
)
from senhance.data.augmentations.distributions import Distribution


@dataclass(kw_only=True)
class RandomNoiseParameters(AugmentationParameters):
    noise: torch.FloatTensor
    decay_ir_filter: torch.FloatTensor
    amplitude: torch.FloatTensor
    f_decay: torch.FloatTensor
    sample_rate: int


class RandomNoise(Augmentation):
    def __init__(
        self,
        amplitude_distribution: Distribution,
        f_decay_distribution: Distribution,
        name: str = "random_noise",
        p: float = 1.0,
    ):
        super().__init__(name=name, p=p)
        self.amplitude_distribution = amplitude_distribution
        self.f_decay_distribution = f_decay_distribution

    def _sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator = None,
    ) -> RandomNoiseParameters:
        amplitude = self.amplitude_distribution.sample(generator=generator)
        f_decay = self.f_decay_distribution.sample(generator=generator)
        sample_rate = audio.sample_rate

        # Use longer filter for better frequency resolution
        filter_length = 512
        f = torch.fft.rfftfreq(filter_length, d=1.0 / sample_rate)
        # Avoid division by zero at DC by clamping to minimum frequency
        f = f.clamp(min=1.0)

        # Create frequency-dependent decay filter
        decay = torch.sqrt(1 / f ** f_decay.to(f.device))
        decay_ir = torch.fft.irfft(decay)
        # Normalize filter to preserve noise power
        decay_ir = decay_ir / torch.sqrt((decay_ir**2).sum())
        noise = torch.randn(
            audio.waveform.shape,
            device=audio.waveform.device,
            dtype=audio.waveform.dtype,
            generator=generator,
        )
        # Apply filter per-channel using groups=channels
        channels = audio.waveform.shape[0]
        decay_ir_filter = decay_ir[None, None].expand(channels, 1, -1)

        return RandomNoiseParameters(
            amplitude=amplitude,
            noise=noise,
            decay_ir_filter=decay_ir_filter,
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
        noise = parameters.noise.to(waveform.device)
        decay_ir_filter = parameters.decay_ir_filter.to(waveform.device)
        amplitude = parameters.amplitude.view(-1, 1, 1).to(waveform.device)

        # Apply frequency-dependent filtering to noise
        # noise: (batch, channels, samples), decay_ir_filter: (batch, channels, 1, filter_len)
        batch_size, channels, samples = noise.shape
        padding = decay_ir_filter.shape[-1] // 2

        # Reshape for grouped convolution: (batch, batch*channels, samples)
        noise_grouped = noise.view(1, batch_size * channels, samples)
        # Reshape filter: (batch*channels, 1, filter_len)
        filter_grouped = decay_ir_filter.view(batch_size * channels, 1, -1)

        # Apply convolution with groups to process each channel independently
        filtered_noise = torch.nn.functional.conv1d(
            noise_grouped, filter_grouped, padding=padding, groups=batch_size * channels
        )
        # Reshape back and trim to original length: (batch, channels, samples)
        filtered_noise = filtered_noise.view(batch_size, channels, -1)[:, :, :samples]

        # Apply amplitude and add to waveform
        waveform[apply] = waveform[apply] + amplitude * filtered_noise
        return waveform
