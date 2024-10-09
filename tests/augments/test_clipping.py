import torch

from denoiser.data.audio import Audio
from denoiser.data.augmentations.augmentations import BatchAugmentationParameters
from denoiser.data.augmentations.clipping import Clipping


def test_clipping():
    generator = torch.Generator().manual_seed(42)
    audio = Audio("/data/denoising/speech/daps/clean/f10_script1_clean.wav")
    augment = Clipping(
        min_clip_percentile=0,
        max_clip_percentile=0.1,
        p=0.5,
    )

    excerpts = [audio.random_excerpt(0.5, generator=generator) for _ in range(8)]
    waveforms = torch.stack([excerpt.waveform for excerpt in excerpts])

    augment_params = [
        augment.sample_parameters(excerpt, generator=generator) for excerpt in excerpts
    ]
    augment_params = BatchAugmentationParameters(augment_params)

    augmented = augment.augment(waveforms, augment_params)
    assert torch.is_tensor(augmented)

    apply = augment_params.apply
    for wav, aug in zip(waveforms[apply], augmented[apply]):
        assert not torch.allclose(wav, aug)
    for wav, aug in zip(waveforms[~apply], augmented[~apply]):
        assert torch.allclose(wav, aug)


if __name__ == "__main__":
    test_clipping()
