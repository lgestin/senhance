from senhance.data.audio import Audio
from senhance.data.augmentations.reverb import Reverb
from senhance.data.source import ArrowAudioSource

from .utils import _test_augment


def test_reverb_with_file(audio_from_file: Audio):
    ir_source = ArrowAudioSource("/data/denoising/noise/irs/RoyJames/data.test.arrow")
    augment = Reverb(ir_source=ir_source, p=0.5)

    _test_augment(augment=augment, audio=audio_from_file)


def test_reverb_with_random(random_audio: Audio):
    ir_source = ArrowAudioSource("/data/denoising/noise/irs/RoyJames/data.test.arrow")
    augment = Reverb(ir_source=ir_source, p=0.5)

    _test_augment(augment=augment, audio=random_audio)
