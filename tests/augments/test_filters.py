import pytest

from senhance.data.audio import Audio
from senhance.data.augmentations.filters import (
    BandPassChain,
    HighPass,
    LowPass,
    LowPassResample,
)

from .utils import _test_augment

freqs_hz = [2000, 4000, 8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000]


@pytest.mark.parametrize("freq_hz", freqs_hz)
def test_lowpass(audio_from_file: Audio, freq_hz: int):
    augment = LowPass(freq_hz=freq_hz, p=0.5)

    if freq_hz <= audio_from_file.sample_rate // 2:
        _test_augment(augment, audio_from_file)


@pytest.mark.parametrize("freq_hz", freqs_hz)
def test_lowpass_resample(audio_from_file: Audio, freq_hz: int):
    augment = LowPassResample(freq_hz=freq_hz, p=0.5)

    if freq_hz <= audio_from_file.sample_rate // 2:
        _test_augment(augment, audio_from_file)


@pytest.mark.parametrize("freq_hz", freqs_hz)
def test_highpass(audio_from_file: Audio, freq_hz: int):
    augment = HighPass(freq_hz=freq_hz, p=0.5)

    if freq_hz <= audio_from_file.sample_rate // 2:
        _test_augment(augment=augment, audio=audio_from_file)


band_pass_freqs_hz = [(2000, 4000), (4000, 8000), (8000, 16000)]


@pytest.mark.parametrize("band_pass_freqs_hz", band_pass_freqs_hz)
def test_bandpass(audio_from_file: Audio, band_pass_freqs_hz: tuple[int, int]):
    augment = BandPassChain(band_hz=band_pass_freqs_hz, p=0.5)

    if band_pass_freqs_hz[1] <= audio_from_file.sample_rate // 2:
        _test_augment(augment=augment, audio=audio_from_file)
