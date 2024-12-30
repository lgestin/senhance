import torch

from senhance.data.audio import Audio
from senhance.data.augmentations.default import get_default_augmentation

test_file_path = Path(__file__).parent.parent / "assets/physicsworks.wav"


def test_augments():
    generator = torch.Generator().manual_seed(42)
    audio = Audio(test_file_path)
    augment = get_default_augmentation(1, split="test", p=1.0)

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
