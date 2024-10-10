from dataclasses import dataclass

import torch
import torchaudio.functional as F

from denoiser.data.audio import Audio
from denoiser.data.augmentations.augmentations import (
    Augmentation,
    AugmentationParameters,
    BatchAugmentationParameters,
)
from denoiser.data.augmentations.chain import Chain


@dataclass(kw_only=True)
class FilterParameters(AugmentationParameters):
    apply: torch.BoolTensor
    freq_hz: torch.FloatTensor
    sample_rate: torch.FloatTensor


class Filter(Augmentation):
    def __init__(self, freqs_hz: list[float] | torch.FloatTensor, p: float = 1.0):
        super().__init__(p=p)
        if not torch.is_tensor(freqs_hz):
            freqs_hz = torch.as_tensor(freqs_hz)
        self.freqs_hz = freqs_hz
        self.weights = torch.full((len(freqs_hz),), 1 / len(freqs_hz))

    def filter_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        freq_hz: float,
    ) -> torch.Tensor:
        raise NotImplementedError

    def sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator = None,
    ) -> FilterParameters:
        apply = torch.rand(tuple(), generator=generator) <= self.p
        freq_idx = torch.multinomial(self.weights, 1).item()
        freq_hz = self.freqs_hz[freq_idx]
        sample_rate = audio.sample_rate
        return FilterParameters(
            apply=apply,
            freq_hz=freq_hz,
            sample_rate=sample_rate,
        )

    @torch.inference_mode()
    def augment(
        self,
        waveform: torch.Tensor,
        parameters: FilterParameters | BatchAugmentationParameters,
    ) -> torch.Tensor:
        if isinstance(parameters, AugmentationParameters):
            parameters = parameters.batch([parameters])

        if not torch.any(parameters.apply):
            return waveform

        apply = parameters.apply
        augmented = waveform.clone()
        for freq_hz in parameters.freq_hz[apply].unique().tolist():
            freq_mask = parameters.freq_hz == freq_hz
            freq_mask = freq_mask & apply
            if not torch.any(freq_mask):
                continue
            sample_rate = parameters.sample_rate[freq_mask].unique().item()
            augmented[freq_mask] = self.filter_waveform(
                waveform=waveform[freq_mask],
                sample_rate=sample_rate,
                freq_hz=freq_hz,
            )
        return augmented


class LowPass(Filter):
    def __init__(self, freqs_hz: list[float] | torch.FloatTensor, p: float = 1.0):
        super().__init__(freqs_hz=freqs_hz, p=p)

    def filter_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        freq_hz: float,
    ) -> torch.Tensor:
        print(sample_rate, freq_hz)
        waveform = F.lowpass_biquad(
            waveform=waveform,
            sample_rate=sample_rate,
            cutoff_freq=freq_hz,
        )
        return waveform


class HighPass(Filter):
    def __init__(self, freqs_hz: list[float] | torch.FloatTensor, p: float = 1.0):
        super().__init__(freqs_hz=freqs_hz, p=p)

    def filter_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        freq_hz: float,
    ) -> torch.Tensor:
        waveform = F.highpass_biquad(
            waveform=waveform,
            sample_rate=sample_rate,
            cutoff_freq=freq_hz,
        )
        return waveform


class BandPass(Filter):
    def __init__(self, bands_hz: list[tuple[float]], p: float = 1.0):
        super().__init__(freqs_hz=bands_hz, p=p)
        self.bands_hz = bands_hz

    def filter_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        freq_hz: tuple[float],
    ) -> torch.Tensor:
        freq_hz = 0.5 * (freq_hz[1] - freq_hz[0])
        waveform = F.bandpass_biquad(
            waveform=waveform,
            sample_rate=sample_rate,
            central_freq=freq_hz,
        )
        return waveform


class BandPassChain(Chain):
    def __init__(self, bands_hz: list[tuple[float]], p: float = 1.0):
        low_pass = LowPass([band_hz[0] for band_hz in bands_hz], p=1.0)
        high_pass = HighPass([band_hz[1] for band_hz in bands_hz], p=1.0)
        super().__init__(low_pass, high_pass, p=p)
