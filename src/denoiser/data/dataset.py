import time
from dataclasses import dataclass, fields

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
        for field in fields(self):
            value = getattr(self, field.name)
            if torch.is_tensor(value) or isinstance(value, BatchAugmentationParameters):
                value = value.to(device, non_blocking=True)
                setattr(self, field.name, value)
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
        t0 = time.perf_counter()
        audio = self.audio_source[idx]
        t1 = time.perf_counter()
        audio = audio.resample(self.sample_rate).normalize(-24.0)
        t2 = time.perf_counter()

        augmentation_params = None
        if self.augmentation is not None:
            augmentation_params = self.augmentation.sample_parameters(
                audio=audio,
                generator=torch.Generator().manual_seed(idx),
            )
            t3 = time.perf_counter()
        # print(f"\tAUDIO    {t1 - t0:.6f}s")
        # print(f"\tRESAMPLE {t2 - t1:.6f}s")
        # print(f"\tAUG      {t3 - t2:.6f}s")
        return Sample(idx=idx, audio=audio, augmentation_params=augmentation_params)
