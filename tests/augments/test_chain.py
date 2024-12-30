import pytest

from senhance.data.audio import Audio
from senhance.data.augmentations.chain import Chain
from senhance.data.augmentations.choose import Choose
from senhance.data.augmentations.gaussian_noise import GaussianNoise

from . import AUDIO_TEST_FILES
from .utils import _test_augment


@pytest.mark.parametrize("audio_file_path", AUDIO_TEST_FILES)
def test_chain(audio_file_path):
    audio = Audio(audio_file_path)
    augment = Chain(
        Choose(
            GaussianNoise(min_amplitude=10, max_amplitude=10),
            GaussianNoise(min_amplitude=100, max_amplitude=100),
        ),
        GaussianNoise(min_amplitude=100, max_amplitude=100),
        p=0.5,
    )

    _test_augment(augment=augment, audio=audio)


if __name__ == "__main__":
    test_chain()
