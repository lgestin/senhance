from dataclasses import dataclass

import torch
import torchaudio.functional as F

from senhance.data.audio import Audio
from senhance.data.augmentations.augmentations import (
    Augmentation,
    AugmentationParameters,
    BatchAugmentationParameters,
)
from senhance.data.augmentations.chain import Chain


@dataclass(kw_only=True)
class FilterParameters(AugmentationParameters):
    sample_rate: torch.FloatTensor


class Filter(Augmentation):
    def __init__(self, freq_hz: float, name: str, p: float = 1.0):
        super().__init__(name=name, p=p)
        self.freq_hz = freq_hz

    def filter_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator = None,
    ) -> FilterParameters:
        sample_rate = audio.sample_rate
        return FilterParameters(sample_rate=sample_rate)

    @torch.inference_mode()
    def _augment(
        self,
        waveform: torch.Tensor,
        parameters: FilterParameters | BatchAugmentationParameters,
    ) -> torch.Tensor:
        if isinstance(parameters, AugmentationParameters):
            parameters = parameters.collate([parameters])

        if parameters is None or (not torch.any(parameters.apply)):
            return waveform

        device = waveform.device
        sample_rate = parameters.sample_rate.unique().to(
            device, non_blocking=True
        )

        apply = parameters.apply
        waveform[apply] = self.filter_waveform(
            waveform=waveform[apply],
            sample_rate=sample_rate,
            freq_hz=self.freq_hz,
        )
        return waveform


class LowPass(Filter):
    def __init__(self, freq_hz: float, name: str = "low_pass", p: float = 1.0):
        super().__init__(freq_hz=freq_hz, name=name, p=p)

    def filter_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        freq_hz: float,
    ) -> torch.Tensor:
        waveform = F.lowpass_biquad(
            waveform=waveform,
            sample_rate=sample_rate,
            cutoff_freq=freq_hz,
        )
        return waveform


class HighPass(Filter):
    def __init__(
        self,
        freq_hz: float,
        name: str = "high_pass",
        p: float = 1.0,
    ):
        super().__init__(freq_hz=freq_hz, name=name, p=p)

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
    def __init__(
        self,
        bands_hz: tuple[float],
        name: str = "band_pass",
        p: float = 1.0,
    ):
        super().__init__(freq_hz=bands_hz, name=name, p=p)
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
    def __init__(
        self,
        band_hz: tuple[float],
        name: str = "band_pass",
        p: float = 1.0,
    ):
        low_pass = LowPass(band_hz[0], p=1.0)
        high_pass = HighPass(band_hz[1], p=1.0)
        super().__init__(low_pass, high_pass, name=name, p=p)
