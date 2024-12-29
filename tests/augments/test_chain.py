from pathlib import Path

import torch

from senhance.data.audio import Audio
from senhance.data.augmentations.chain import Chain
from senhance.data.augmentations.choose import Choose
from senhance.data.augmentations.gaussian_noise import GaussianNoise

test_file_path = Path(__file__).parent.parent / "assets/physicsworks.wav"


def test_chain():
    audio = Audio(test_file_path.as_posix())
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
    waveforms = [excerpt.waveform for excerpt in excerpts]
    waveforms = [torch.from_numpy(waveform) / 32678.0 for waveform in waveforms]
    waveforms = torch.stack(waveforms)

    augment_params = [
        augment.sample_parameters(
            excerpt, generator=torch.Generator().manual_seed(i)
        )
        for i, excerpt in enumerate(excerpts)
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
    augment_params = augment.sample_parameters(
        excerpt, generator=torch.Generator().manual_seed(0)
    )
    augmented = augment.augment(waveform.clone(), augment_params)
    assert torch.is_tensor(augmented)
    if augment_params.apply:
        assert not torch.allclose(augmented, waveform)
    else:
        assert torch.allclose(augmented, waveform)


if __name__ == "__main__":
    test_chain()
