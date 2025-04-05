import pytest

from senhance.data.audio import Audio
from senhance.data.augmentations.chain import Chain
from senhance.data.augmentations.choose import Choose
from senhance.data.augmentations.distributions import Uniform
from senhance.data.augmentations.random_noise import RandomNoise

from . import AUDIO_TEST_FILES
from .utils import _test_augment


@pytest.mark.parametrize("audio_file_path", AUDIO_TEST_FILES)
def test_choose(audio_file_path):
    audio = Audio(audio_file_path)
    augment = Choose(
        RandomNoise(
            amplitude_distribution=Uniform(min=10, max=10),
            f_decay_distribution=Uniform(min=0, max=0),
        ),
        Chain(
            RandomNoise(
                amplitude_distribution=Uniform(min=10, max=10),
                f_decay_distribution=Uniform(min=0, max=0),
            ),
            RandomNoise(
                amplitude_distribution=Uniform(min=100, max=100),
                f_decay_distribution=Uniform(min=0, max=0),
            ),
        ),
        weights=[0.5, 0.5],
        p=0.5,
    )

    _test_augment(augment=augment, audio=audio)


if __name__ == "__main__":
    test_choose(AUDIO_TEST_FILES[0])
