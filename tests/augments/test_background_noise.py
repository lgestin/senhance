import torch

from denoiser.data.audio import Audio
from denoiser.data.augmentations.background_noise import (
    BackgroundNoise,
    BatchAugmentationParameters,
)


def test_background_noise():
    generator = torch.Generator().manual_seed(42)
    audio = Audio("/data/denoising/speech/daps/clean/f10_script1_clean.wav")
    augment = BackgroundNoise(
        noise_index_path="/data/denoising/noise/records/musan/noise/index.train.json",
        min_snr=-15.0,
        max_snr=-5.0,
        p=0.5,
    )

    excerpts = [audio.salient_excerpt(0.5, generator=generator) for _ in range(8)]
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
    test_background_noise()
