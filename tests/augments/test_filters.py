from pathlib import Path

import torch

from senhance.data.audio import Audio
from senhance.data.augmentations.filters import BandPassChain, HighPass, LowPass

test_file_path = Path(__file__).parent.parent / "assets/physicsworks.wav"


def test_lowpass():
    generator = torch.Generator().manual_seed(42)
    audio = Audio(test_file_path)
    augment = LowPass(freqs_hz=[2000, 4000, 8000, 16000], p=0.5)

    excerpts = [
        audio.random_excerpt(0.5, generator=generator) for _ in range(8)
    ]
    waveforms = [excerpt.waveform for excerpt in excerpts]
    waveforms = [torch.from_numpy(waveform) / 32678.0 for waveform in waveforms]
    waveforms = torch.stack(waveforms)

    augment_params = [
        augment.sample_parameters(excerpt, generator=generator)
        for excerpt in excerpts
    ]
    augment_params = augment_params[0].collate(augment_params)

    augmented = augment.augment(waveforms.clone(), augment_params)
    assert torch.is_tensor(augmented)

    apply = augment_params.apply
    for wav, aug in zip(waveforms[apply], augmented[apply]):
        assert not torch.allclose(wav, aug)
    for wav, aug in zip(waveforms[~apply], augmented[~apply]):
        assert torch.allclose(wav, aug)

    excerpt, waveform = excerpts[0], waveforms[:1]
    augment_params = augment.sample_parameters(excerpt, generator=generator)
    augmented = augment.augment(waveform.clone(), augment_params)
    assert torch.is_tensor(augmented)
    if augment_params.apply:
        assert not torch.allclose(augmented, waveform)
    else:
        assert torch.allclose(augmented, waveform)


def test_highpass():
    generator = torch.Generator().manual_seed(42)
    audio = Audio(test_file_path)
    augment = HighPass(freqs_hz=[2000, 4000, 8000, 16000], p=0.5)

    excerpts = [
        audio.random_excerpt(0.5, generator=generator) for _ in range(8)
    ]
    waveforms = [excerpt.waveform for excerpt in excerpts]
    waveforms = [torch.from_numpy(waveform) / 32678.0 for waveform in waveforms]
    waveforms = torch.stack(waveforms)

    augment_params = [
        augment.sample_parameters(excerpt, generator=generator)
        for excerpt in excerpts
    ]
    augment_params = augment_params[0].collate(augment_params)

    augmented = augment.augment(waveforms.clone(), augment_params)
    assert torch.is_tensor(augmented)

    apply = augment_params.apply
    for wav, aug in zip(waveforms[apply], augmented[apply]):
        assert not torch.allclose(wav, aug)
    for wav, aug in zip(waveforms[~apply], augmented[~apply]):
        assert torch.allclose(wav, aug)

    excerpt, waveform = excerpts[0], waveforms[:1]
    augment_params = augment.sample_parameters(excerpt, generator=generator)
    augmented = augment.augment(waveform.clone(), augment_params)
    assert torch.is_tensor(augmented)
    if augment_params.apply:
        assert not torch.allclose(augmented, waveform)
    else:
        assert torch.allclose(augmented, waveform)


def test_bandpass():
    generator = torch.Generator().manual_seed(42)
    audio = Audio(test_file_path)
    augment = BandPassChain(
        bands_hz=[[2000, 4000], [4000, 8000], [8000, 16000]], p=0.5
    )

    excerpts = [
        audio.random_excerpt(0.5, generator=generator) for _ in range(8)
    ]
    waveforms = [excerpt.waveform for excerpt in excerpts]
    waveforms = [torch.from_numpy(waveform) / 32678.0 for waveform in waveforms]
    waveforms = torch.stack(waveforms)

    augment_params = [
        augment.sample_parameters(excerpt, generator=generator)
        for excerpt in excerpts
    ]
    augment_params = augment_params[0].collate(augment_params)

    augmented = augment.augment(waveforms.clone(), augment_params)
    assert torch.is_tensor(augmented)

    apply = augment_params.apply
    for wav, aug in zip(waveforms[apply], augmented[apply]):
        assert not torch.allclose(wav, aug)
    for wav, aug in zip(waveforms[~apply], augmented[~apply]):
        assert torch.allclose(wav, aug)

    excerpt, waveform = excerpts[0], waveforms[:1]
    augment_params = augment.sample_parameters(excerpt, generator=generator)
    augmented = augment.augment(waveform.clone(), augment_params)
    assert torch.is_tensor(augmented)
    if augment_params.apply:
        assert not torch.allclose(augmented, waveform)
    else:
        assert torch.allclose(augmented, waveform)


if __name__ == "__main__":
    test_lowpass()
    test_highpass()
    test_bandpass()
