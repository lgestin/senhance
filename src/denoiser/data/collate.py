import torch
from denoiser.data.augmentations import BatchAugmentParameters
from denoiser.data.dataset import Sample, Batch


def collate(samples: list[Sample]) -> Batch:
    idxs = [sample.idx for sample in samples]
    audios = [sample.audio for sample in samples]
    waveforms = torch.stack([audio.waveform for audio in audios])
    augmentation_params = BatchAugmentParameters(
        [sample.augmentation_params for sample in samples]
    )
    return Batch(
        idxs=idxs,
        audios=audios,
        waveforms=waveforms,
        augmentation_params=augmentation_params,
    )
