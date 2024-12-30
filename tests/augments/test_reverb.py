import pytest

from senhance.data.audio import Audio
from senhance.data.augmentations.reverb import Reverb
from senhance.data.source import ArrowAudioSource

from . import AUDIO_TEST_FILES
from .utils import _test_augment


@pytest.mark.parametrize("audio_file_path", AUDIO_TEST_FILES)
def test_reverb(audio_file_path):
    audio = Audio(audio_file_path)
    ir_source = ArrowAudioSource(
        "/data/denoising/noise/irs/RoyJames/data.test.arrow"
    )
    augment = Reverb(ir_source=ir_source, p=0.5)

    _test_augment(augment=augment, audio=audio)


if __name__ == "__main__":
    test_reverb()
