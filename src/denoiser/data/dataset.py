import torch
from torch.utils.data import Dataset

from dataclasses import dataclass

from denoiser.data.audio import Audio
from denoiser.data.source import AudioSource


@dataclass
class Sample:
    audio: Audio
    idx: int


class Batch:
    audios: list[Audio]
    idxs: list[int]


class AudioDataset(Dataset):
    def __init__(self, audio_source: AudioSource, sample_rate: int):
        self.audio_source = audio_source
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.audio_source)

    def __getitem__(self, idx: int) -> Sample:
        audio = self.audio_source[idx]
        audio = audio.resample(self.sample_rate)
        return idx, audio
