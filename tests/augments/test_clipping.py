import pytest

from senhance.data.audio import Audio
from senhance.data.augmentations.clipping import Clipping
from senhance.data.augmentations.distributions import Uniform

from . import AUDIO_TEST_FILES
from .utils import _test_augment


@pytest.mark.parametrize("audio_file_path", AUDIO_TEST_FILES)
def test_clipping(audio_file_path):
    audio = Audio(audio_file_path)
    augment = Clipping(
        clip_percentile_distribution=Uniform(min=0.0, max=0.1),
        p=0.5,
    )

    _test_augment(augment=augment, audio=audio)


if __name__ == "__main__":
    test_clipping()
