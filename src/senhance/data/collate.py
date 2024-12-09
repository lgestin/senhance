import torch

from senhance.data.dataset import Batch, Sample


def collate(samples: list[Sample]) -> Batch:
    idxs = [sample.idx for sample in samples]
    audios = [sample.audio for sample in samples]
    waveforms = torch.stack([audio.waveform for audio in audios])
    augmentation_params = samples[0].augmentation_params
    if augmentation_params is not None:
        augmentation_params = augmentation_params.collate(
            [sample.augmentation_params for sample in samples]
        )
    return Batch(
        idxs=idxs,
        audios=audios,
        waveforms=waveforms,
        augmentation_params=augmentation_params,
    )
