import pytest
import torch

from senhance.data.audio import Audio
from senhance.data.augmentations.default import get_default_augmentation

from . import AUDIO_TEST_FILES


@pytest.mark.parametrize("audio_file_path", AUDIO_TEST_FILES)
def test_augments(audio_file_path):
    generator = torch.Generator().manual_seed(42)
    audio = Audio(audio_file_path)
    augment = get_default_augmentation(
        "/data/denoising/noise/",
        sample_rate=audio.sample_rate,
        sequence_length_s=1,
        split="test",
        p=1.0,
    )

    excerpts = [
        audio.random_excerpt(0.5, generator=generator) for _ in range(8)
    ]

    augment_params = [
        augment.sample_parameters(excerpt, generator=generator)
        for excerpt in excerpts
    ]
    augment_params = augment_params[0].collate(augment_params)


if __name__ == "__main__":
    test_augments()
