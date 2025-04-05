from dataclasses import dataclass

import julius
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
        sample_rate = parameters.sample_rate.unique().to(device)

        apply = parameters.apply
        waveform[apply] = self.filter_waveform(
            waveform=waveform[apply],
            sample_rate=sample_rate,
            # freq_hz=self.freq_hz,
        )
        return waveform


class LowPass(Filter):
    def __init__(self, freq_hz: float, name: str = "low_pass", p: float = 1.0):
        super().__init__(freq_hz=freq_hz, name=name, p=p)

    def filter_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        assert self.freq_hz < sample_rate / 2
        waveform = julius.lowpass_filter(waveform, cutoff=self.freq_hz)
        return waveform


class LowPassResample(Filter):
    def __init__(
        self,
        freq_hz: float,
        name: str = "low_pass_resample",
        p: float = 1.0,
    ):
        super().__init__(freq_hz=freq_hz, name=name, p=p)

    @torch.inference_mode()
    def filter_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        assert self.freq_hz < sample_rate / 2
        s = waveform.shape[-1]
        waveform = F.resample(
            waveform=waveform,
            orig_freq=sample_rate,
            new_freq=self.freq_hz,
        )
        waveform = F.resample(
            waveform=waveform,
            orig_freq=self.freq_hz,
            new_freq=sample_rate,
        )
        return waveform[..., :s]


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
    ) -> torch.Tensor:
        assert self.freq_hz < sample_rate / 2
        waveform = julius.highpass_filter(waveform, self.freq_hz)
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
    ) -> torch.Tensor:
        waveform = julius.bandpass_filter(
            waveform,
            cutoff_low=self.bands_hz[0],
            cutoff_high=self.bands_hz[1],
        )
        return waveform


class BandPassChain(Chain):
    def __init__(
        self,
        band_hz: tuple[float],
        name: str = "band_pass",
        p: float = 1.0,
    ):
        # High pass removes frequencies BELOW band_hz[0] (the low cutoff)
        # Low pass removes frequencies ABOVE band_hz[1] (the high cutoff)
        high_pass = HighPass(band_hz[0], p=1.0)
        low_pass = LowPass(band_hz[1], p=1.0)
        super().__init__(high_pass, low_pass, name=name, p=p)
