import torch

from senhance.data.audio import Audio
from senhance.data.augmentations.gaussian_noise import GaussianNoise
from senhance.data.augmentations.chain import Chain
from senhance.data.augmentations.choose import Choose


def test_chain():
    audio = Audio("/data/denoising/speech/daps/clean/f10_script1_clean.wav")
    augment = Chain(
        Choose(
            GaussianNoise(min_amplitude=10, max_amplitude=10),
            GaussianNoise(min_amplitude=100, max_amplitude=100),
        ),
        GaussianNoise(min_amplitude=100, max_amplitude=100),
        p=0.5,
    )

    excerpts = [
        audio.salient_excerpt(0.5, generator=torch.Generator().manual_seed(i))
        for i in range(8)
    ]
    waveforms = torch.stack([excerpt.waveform for excerpt in excerpts])

    augment_params = [
        augment.sample_parameters(excerpt, generator=torch.Generator().manual_seed(i))
        for i, excerpt in enumerate(excerpts)
    ]
    augment_params = augment_params[0].collate(augment_params)

    augmented = augment.augment(waveforms, augment_params)
    assert torch.is_tensor(augmented)

    apply = augment_params.apply
    for wav, aug in zip(waveforms[apply], augmented[apply]):
        assert not torch.allclose(wav, aug)
    for wav, aug in zip(waveforms[~apply], augmented[~apply]):
        assert torch.allclose(wav, aug)

    excerpt = excerpts[0]
    augment_params = augment.sample_parameters(excerpt, generator=generator)
    augmented = augment.augment(excerpt.waveform[None], augment_params)
    assert torch.is_tensor(augmented)
    if augment_params.apply:
        assert not torch.allclose(augmented, excerpt.waveform[None])
    else:
        assert torch.allclose(augmented, excerpt.waveform[None])


if __name__ == "__main__":
    test_chain()
