import pytest

from senhance.data.audio import Audio
from senhance.data.augmentations.random_noise import RandomNoise

from . import AUDIO_TEST_FILES
from .utils import _test_augment


@pytest.mark.parametrize("audio_file_path", AUDIO_TEST_FILES)
def test_clipping(audio_file_path):
    audio = Audio(audio_file_path)
    augment = RandomNoise(
        min_amplitude=0.1,
        max_amplitude=0.3,
        min_f_decay=-2,
        max_f_decay=2,
        p=0.5,
    )

    _test_augment(augment=augment, audio=audio)


if __name__ == "__main__":
    test_clipping()
