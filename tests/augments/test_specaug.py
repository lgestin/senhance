import pytest

from senhance.data.audio import Audio
from senhance.data.augmentations.specaug import SpecAugFreq, SpecAugTime

from . import AUDIO_TEST_FILES
from .utils import _test_stft_augment


@pytest.mark.parametrize("audio_file_path", AUDIO_TEST_FILES)
def test_specaug_freq(audio_file_path):
    audio = Audio(audio_file_path)
    augment = SpecAugFreq(min_freq_perc_mask=0.1, max_freq_perc_mask=0.3)
    _test_stft_augment(augment, audio)


@pytest.mark.parametrize("audio_file_path", AUDIO_TEST_FILES)
def test_specaug_time(audio_file_path):
    audio = Audio(audio_file_path)
    augment = SpecAugTime(min_time_perc_mask=0.1, max_time_perc_mask=0.3)
    _test_stft_augment(augment, audio)
