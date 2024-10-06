import torch
from denoiser.data.audio import Audio
from denoiser.data.augmentations import BackgroundNoise, BatchAugmentParameters


def test_background_noise():
    audio = Audio("/data/denoising/speech/daps/clean/f10_script1_clean.wav")
    augment = BackgroundNoise(
        noise_index_path="/data/denoising/noise/records/musan/noise/index.train.json",
        min_snr=-5.0,
        max_snr=25.0,
        p=1.0,
    )

    excerpts = [audio.salient_excerpt(0.5) for _ in range(8)]
    waveforms = torch.stack([excerpt.waveform for excerpt in excerpts])

    augment_params = [
        augment.sample_augment_parameters(excerpt) for excerpt in excerpts
    ]
    augment_params = BatchAugmentParameters(augment_params)

    augmented = augment.augment(waveforms, augment_params)
    assert torch.is_tensor(augmented)


if __name__ == "__main__":
    test_background_noise()
