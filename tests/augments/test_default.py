import torch

from denoiser.data.audio import Audio
from denoiser.data.augmentations.gaussian_noise import GaussianNoise
from denoiser.data.augmentations.default import get_default_augmentation
from denoiser.data.augmentations.choose import Choose


def test_default():
    generator = torch.Generator().manual_seed(42)
    audio = Audio("/data/denoising/speech/daps/clean/f10_script1_clean.wav")
    augment = get_default_augmentation(split="train", sequence_length_s=0.5, p=0.5)

    excerpts = [
        audio.salient_excerpt(0.5, generator=torch.Generator().manual_seed(i))
        for i in range(8)
    ]
    waveforms = torch.stack([excerpt.waveform for excerpt in excerpts])

    augment_params = [
        augment.sample_parameters(excerpt, generator=torch.Generator().manual_seed(i))
        for i, excerpt in enumerate(excerpts)
    ]
    augment_params = augment_params[0].batch(augment_params)

    augmented = augment.augment(waveforms, augment_params)
    assert torch.is_tensor(augmented)

    import ipdb

    ipdb.set_trace()
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
    test_default()
