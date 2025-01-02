import pytest

from senhance.data.audio import Audio
from senhance.data.augmentations.background_noise import BackgroundNoise
from senhance.data.source import ArrowAudioSource

from . import AUDIO_TEST_FILES
from .utils import _test_augment


@pytest.mark.parametrize("audio_file_path", AUDIO_TEST_FILES)
def test_background_noise(audio_file_path):
    audio = Audio(audio_file_path)
    noise_source = ArrowAudioSource(
        arrow_file="/data/denoising/noise/records/urbansound8k/data.test.arrow"
    )
    augment = BackgroundNoise(
        noise_source=noise_source,
        min_snr=-15.0,
        max_snr=-5.0,
        p=0.5,
    )

    _test_augment(augment=augment, audio=audio)


if __name__ == "__main__":
    test_background_noise()
