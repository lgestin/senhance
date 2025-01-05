import torch

from senhance.data.audio import Audio
from senhance.data.augmentations.augmentations import Augmentation

from . import TEST_SEED


def _test_augment(augment: Augmentation, audio: Audio):
    generator = torch.Generator().manual_seed(TEST_SEED)
    excerpts = [
        audio.random_excerpt(0.5, generator=generator) for _ in range(8)
    ]
    waveforms = torch.stack([excerpt.waveform for excerpt in excerpts])

    batch_params = [
        augment.sample_parameters(excerpt, generator=generator)
        for excerpt in excerpts
    ]
    ref_params = next(filter(lambda p: p is not None, batch_params), None)
    if ref_params:
        collated_params = ref_params.collate(batch_params)
    else:
        collated_params = None

    augmented = augment.augment(waveforms.clone(), collated_params)
    assert torch.is_tensor(augmented)

    apply = collated_params.apply
    for wav, aug in zip(waveforms[apply], augmented[apply]):
        assert not torch.any(torch.isnan(aug))
        assert not torch.allclose(wav, aug)
    for wav, aug in zip(waveforms[~apply], augmented[~apply]):
        assert torch.allclose(wav, aug)

    excerpt, waveform = excerpts[0], waveforms[:1]
    augment_params = augment.sample_parameters(excerpt, generator=generator)
    augmented = augment.augment(waveform.clone(), augment_params)
    assert torch.is_tensor(augmented)
    if augment_params is None:
        assert torch.allclose(augmented, waveform)
    else:
        assert not torch.allclose(augmented, waveform)
