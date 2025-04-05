from senhance.data.augmentations.distributions import Uniform
from senhance.data.augmentations.random_noise import RandomNoise

from .utils import _test_augment


def test_random_noise_from_file(audio_from_file):
    """Test RandomNoise augmentation with audio loaded from files."""
    augment = RandomNoise(
        amplitude_distribution=Uniform(min=0.1, max=0.3),
        f_decay_distribution=Uniform(min=-2, max=2),
        p=0.5,
    )

    _test_augment(augment=augment, audio=audio_from_file)


def test_random_noise_random_audio(random_audio):
    """Test RandomNoise augmentation with randomly generated audio."""
    augment = RandomNoise(
        amplitude_distribution=Uniform(min=0.1, max=0.3),
        f_decay_distribution=Uniform(min=-2, max=2),
        p=0.5,
    )

    _test_augment(augment=augment, audio=random_audio)
