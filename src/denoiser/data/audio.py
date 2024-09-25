import torch
import torchaudio
from dataclasses import dataclass

from denoiser.data.utils import load_audio


@dataclass
class Audio:
    filepath: str
    start: int = 0
    end: int | None = None

    def __post_init__(self):
        self._waveform = None
        self._sample_rate = None

    @property
    def sample_rate(self):
        if self._sample_rate is None:
            self._sample_rate = torchaudio.info(self.filepath).sample_rate
        return self._sample_rate

    @property
    def waveform(self):
        if self._waveform is None:
            waveform, sr = load_audio(self.filepath, start=self.start, end=self.end)
            self._waveform = waveform
            self._sample_rate = sr
        return waveform

    @property
    def duration(self):
        waveform = self.waveform
        sample_rate = self.sample_rate
        duration = waveform.shape[-1] / sample_rate
        return duration
