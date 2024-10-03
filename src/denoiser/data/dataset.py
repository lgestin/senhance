import torch
from torch.utils.data import Dataset

from dataclasses import dataclass

from denoiser.data.utils import resample
from denoiser.data.source import AudioSource


@dataclass
class Sample:
    waveform: torch.Tensor
    sample_rate: int
    idx: int


class AudioDataset(Dataset):
    def __init__(self, audio_source: AudioSource, sample_rate: int):
        self.audio_source = audio_source
        self.sample_rate = sample_rate

    def __getitem__(self, idx: int) -> Sample:
        audio = self.audio_source[idx]
        waveform = audio.waveform
        waveform = resample(waveform, audio.sample_rate, self.sample_rate)
        item = Sample(waveform=audio.waveform, sample_rate=self.sample_rate, idx=idx)
        return item
