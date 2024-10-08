from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from denoiser.data.audio import Audio
from denoiser.data.augmentations.augmentations import (
    Augmentation,
    AugmentationParameters,
    BatchAugmentationParameters,
)
from denoiser.data.source import AudioSource


@dataclass
class Sample:
    idx: int
    audio: Audio
    augmentation_params: AugmentationParameters


@dataclass
class Batch:
    idxs: list[int]
    audios: list[Audio]
    waveforms: torch.FloatTensor
    augmentation_params: BatchAugmentationParameters

    def to(self, device: str | torch.device):
        for key in self.__annotations__.keys():
            value = getattr(self, key)
            if torch.is_tensor(value) or isinstance(value, BatchAugmentationParameters):
                value = value.to(device)
                setattr(self, key, value)
        return self


class AudioDataset(Dataset):
    def __init__(
        self,
        audio_source: AudioSource,
        sample_rate: int,
        augmentation: Augmentation = None,
    ):
        self.audio_source = audio_source
        self.sample_rate = sample_rate
        self.augmentation = augmentation

    def __len__(self):
        return len(self.audio_source)

    def __getitem__(self, idx: int) -> Sample:
        audio = self.audio_source[idx]
        audio = audio.resample(self.sample_rate).normalize(-24.0)

        augmentation_params = None
        if self.augmentation is not None:
            augmentation_params = self.augmentation.sample_parameters(
                audio=audio,
                generator=torch.Generator().manual_seed(idx),
            )
        return Sample(idx=idx, audio=audio, augmentation_params=augmentation_params)
