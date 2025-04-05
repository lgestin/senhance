from senhance.data.augmentations.chain import Chain
from senhance.data.augmentations.choose import Choose
from senhance.data.augmentations.distributions import Uniform
from senhance.data.augmentations.random_noise import RandomNoise
from senhance.data.augmentations.specaug import SpecAugFreq, SpecAugTime

from .utils import _test_augment


def test_chain_with_choose_from_file(audio_from_file):
    """Test Chain augmentation with Choose using audio from files."""
    augment = Chain(
        Choose(
            RandomNoise(
                amplitude_distribution=Uniform(min=10, max=10),
                f_decay_distribution=Uniform(min=0, max=0),
            ),
            RandomNoise(
                amplitude_distribution=Uniform(min=100, max=100),
                f_decay_distribution=Uniform(min=0, max=0),
            ),
        ),
        RandomNoise(
            amplitude_distribution=Uniform(min=1, max=1),
            f_decay_distribution=Uniform(min=0, max=0),
        ),
        p=0.5,
    )
    _test_augment(augment=augment, audio=audio_from_file)


def test_chain_with_choose_random_audio(random_audio):
    """Test Chain augmentation with Choose using randomly generated audio."""
    augment = Chain(
        Choose(
            RandomNoise(
                amplitude_distribution=Uniform(min=10, max=10),
                f_decay_distribution=Uniform(min=0, max=0),
            ),
            RandomNoise(
                amplitude_distribution=Uniform(min=100, max=100),
                f_decay_distribution=Uniform(min=0, max=0),
            ),
        ),
        RandomNoise(
            amplitude_distribution=Uniform(min=1, max=1),
            f_decay_distribution=Uniform(min=0, max=0),
        ),
        p=0.5,
    )
    _test_augment(augment=augment, audio=random_audio)


def test_chain_multiple_augments_from_file(audio_from_file):
    """Test Chain with multiple different augmentations using audio from files."""
    augment = Chain(
        RandomNoise(
            amplitude_distribution=Uniform(min=0.5, max=0.6),
            f_decay_distribution=Uniform(min=0, max=0),
        ),
        SpecAugFreq(freq_perc_mask_distribution=Uniform(min=0.1, max=0.3)),
        RandomNoise(
            amplitude_distribution=Uniform(min=0.5, max=0.6),
            f_decay_distribution=Uniform(min=0, max=0),
        ),
        SpecAugTime(time_perc_mask_distribution=Uniform(min=0.1, max=0.3)),
        RandomNoise(
            amplitude_distribution=Uniform(min=0.5, max=0.6),
            f_decay_distribution=Uniform(min=0, max=0),
        ),
        p=0.5,
    )
    _test_augment(augment=augment, audio=audio_from_file)


def test_chain_multiple_augments_random_audio(random_audio):
    """Test Chain with multiple different augmentations using randomly generated audio."""
    augment = Chain(
        RandomNoise(
            amplitude_distribution=Uniform(min=0.5, max=0.6),
            f_decay_distribution=Uniform(min=0, max=0),
        ),
        SpecAugFreq(freq_perc_mask_distribution=Uniform(min=0.1, max=0.3)),
        RandomNoise(
            amplitude_distribution=Uniform(min=0.5, max=0.6),
            f_decay_distribution=Uniform(min=0, max=0),
        ),
        SpecAugTime(time_perc_mask_distribution=Uniform(min=0.1, max=0.3)),
        RandomNoise(
            amplitude_distribution=Uniform(min=0.5, max=0.6),
            f_decay_distribution=Uniform(min=0, max=0),
        ),
        p=0.5,
    )
    _test_augment(augment=augment, audio=random_audio)


def test_chain_nested_from_file(audio_from_file):
    """Test nested Chain augmentations using audio from files."""
    augment = Chain(
        Chain(
            RandomNoise(
                amplitude_distribution=Uniform(min=0.5, max=0.6),
                f_decay_distribution=Uniform(min=0, max=0),
            ),
            SpecAugFreq(freq_perc_mask_distribution=Uniform(min=0.1, max=0.3)),
        ),
        RandomNoise(
            amplitude_distribution=Uniform(min=0.5, max=0.6),
            f_decay_distribution=Uniform(min=0, max=0),
        ),
        SpecAugTime(time_perc_mask_distribution=Uniform(min=0.1, max=0.3)),
        RandomNoise(
            amplitude_distribution=Uniform(min=0.5, max=0.6),
            f_decay_distribution=Uniform(min=0, max=0),
        ),
        p=0.5,
    )
    _test_augment(augment=augment, audio=audio_from_file)


def test_chain_nested_random_audio(random_audio):
    """Test nested Chain augmentations using randomly generated audio."""
    augment = Chain(
        Chain(
            RandomNoise(
                amplitude_distribution=Uniform(min=0.5, max=0.6),
                f_decay_distribution=Uniform(min=0, max=0),
            ),
            SpecAugFreq(freq_perc_mask_distribution=Uniform(min=0.1, max=0.3)),
        ),
        RandomNoise(
            amplitude_distribution=Uniform(min=0.5, max=0.6),
            f_decay_distribution=Uniform(min=0, max=0),
        ),
        SpecAugTime(time_perc_mask_distribution=Uniform(min=0.1, max=0.3)),
        RandomNoise(
            amplitude_distribution=Uniform(min=0.5, max=0.6),
            f_decay_distribution=Uniform(min=0, max=0),
        ),
        p=0.5,
    )
    _test_augment(augment=augment, audio=random_audio)
