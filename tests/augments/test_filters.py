import pytest

from senhance.data.audio import Audio
from senhance.data.augmentations.filters import BandPassChain, HighPass, LowPass

from . import AUDIO_TEST_FILES
from .utils import _test_augment

freqs_hz = [2000, 4000, 8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000]


@pytest.mark.parametrize("audio_file_path", AUDIO_TEST_FILES)
@pytest.mark.parametrize("freq_hz", freqs_hz)
def test_lowpass(audio_file_path, freq_hz):
    audio = Audio(audio_file_path)
    augment = LowPass(freq_hz=freq_hz, p=0.5)

    if freq_hz <= audio.sample_rate // 2:
        _test_augment(augment, audio)


@pytest.mark.parametrize("audio_file_path", AUDIO_TEST_FILES)
@pytest.mark.parametrize("freq_hz", freqs_hz)
def test_highpass(audio_file_path, freq_hz):
    audio = Audio(audio_file_path)
    augment = HighPass(freq_hz=freq_hz, p=0.5)

    if freq_hz <= audio.sample_rate // 2:
        _test_augment(augment=augment, audio=audio)


band_pass_freqs_hz = [(2000, 4000), (4000, 8000), (8000, 16000)]


@pytest.mark.parametrize("audio_file_path", AUDIO_TEST_FILES)
@pytest.mark.parametrize("band_pass_freqs_hz", band_pass_freqs_hz)
def test_bandpass(audio_file_path, band_pass_freqs_hz):
    audio = Audio(audio_file_path)
    augment = BandPassChain(band_hz=band_pass_freqs_hz, p=0.5)

    if band_pass_freqs_hz[1] <= audio.sample_rate // 2:
        _test_augment(augment=augment, audio=audio)


if __name__ == "__main__":
    test_lowpass(AUDIO_TEST_FILES[0], 4000)
    test_highpass(AUDIO_TEST_FILES[0], 8000)
    test_bandpass(AUDIO_TEST_FILES[0], (4000, 8000))
