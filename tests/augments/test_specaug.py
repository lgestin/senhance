from senhance.data.augmentations.distributions import Uniform
from senhance.data.augmentations.specaug import SpecAugFreq, SpecAugTime

from .utils import _test_stft_augment


def test_specaug_freq(audio_from_file):
    augment = SpecAugFreq(freq_perc_mask_distribution=Uniform(min=0.1, max=0.3))
    _test_stft_augment(augment, audio_from_file)


def test_specaug_time(audio_from_file):
    augment = SpecAugTime(time_perc_mask_distribution=Uniform(min=0.1, max=0.3))
    _test_stft_augment(augment, audio_from_file)
