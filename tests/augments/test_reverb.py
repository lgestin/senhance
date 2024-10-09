import torch

from denoiser.data.audio import Audio
from denoiser.data.augmentations.reverb import Reverb
from denoiser.data.augmentations.augmentations import BatchAugmentationParameters


def test_reverb():
    generator = torch.Generator().manual_seed(42)
    audio = Audio("/data/denoising/speech/daps/clean/f10_script1_clean.wav")
    augment = Reverb(
        "/data/denoising/noise/irs/RoyJames/OPENAIR/IRs/air-museum/index.json",
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
    test_reverb()
