from senhance.data.augmentations.augmentations import Augmentation
from senhance.data.augmentations.default import get_default_augmentation

from .utils import _test_augment


def set_all_probs_to_one(augmentation: Augmentation):
    """Recursively set all augmentation probabilities to 1.0."""
    augmentation.p = 1.0
    if hasattr(augmentation, "augmentations"):
        for aug in augmentation.augmentations:
            set_all_probs_to_one(aug)


def test_default(audio_from_file):
    default_augmentation = get_default_augmentation(
        noise_folder="/data/denoising/noise/",
        sample_rate=audio_from_file.sample_rate,
        split="train",
        sequence_length_s=0.5,
        p=0.5,
    )
    set_all_probs_to_one(default_augmentation)

    _test_augment(augment=default_augmentation, audio=audio_from_file)
