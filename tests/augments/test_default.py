import pytest

from senhance.data.audio import Audio
from senhance.data.augmentations.default import get_default_augmentation

from . import AUDIO_TEST_FILES
from .utils import _test_augment


@pytest.mark.parametrize("audio_file_path", AUDIO_TEST_FILES)
def test_default(audio_file_path):
    audio = Audio(audio_file_path)
    augment = get_default_augmentation(
        noise_folder="/data/denoising/noise/",
        sample_rate=audio.sample_rate,
        split="train",
        sequence_length_s=0.5,
        p=0.5,
    )

    _test_augment(augment=augment, audio=audio)


if __name__ == "__main__":
    test_default(AUDIO_TEST_FILES[0])
