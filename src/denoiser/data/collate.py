import torch

from denoiser.data.augmentations.augmentations import BatchAugmentationParameters
from denoiser.data.dataset import Batch, Sample


def collate(samples: list[Sample]) -> Batch:
    idxs = [sample.idx for sample in samples]
    audios = [sample.audio for sample in samples]
    waveforms = torch.stack([audio.waveform for audio in audios])
    augmentation_params = BatchAugmentationParameters(
        [sample.augmentation_params for sample in samples]
    )
    return Batch(
        idxs=idxs,
        audios=audios,
        waveforms=waveforms,
        augmentation_params=augmentation_params,
    )
