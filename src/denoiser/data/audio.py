import torch
import torchaudio
import soundfile as sf
from dataclasses import dataclass
from pathlib import Path

from denoiser.data.utils import load_audio


@dataclass
class AudioInfo:
    filepath: str
    sample_rate: int
    duration_s: float
    loudness: float


class Audio:
    def __init__(
        self,
        filepath: str = None,
        waveform: torch.Tensor = None,
        sample_rate: int = None,
        start_s: float = 0,
        end_s: float = None,
    ):
        assert isinstance(filepath, (type(None), str, Path))
        if filepath is None:
            assert torch.is_tensor(waveform)
            assert isinstance(sample_rate, int)
            assert waveform is not None and sample_rate is not None

        self.filepath = filepath
        self._waveform = waveform
        self._sample_rate = sample_rate
        self.start_s = start_s
        self.end_s = end_s
        self._loudness = None

    @property
    def sample_rate(self):
        if self._sample_rate is None:
            sample_rate = sf.SoundFile(self.filepath).samplerate
            self._sample_rate = sample_rate
        return self._sample_rate

    @property
    def waveform(self):
        waveform = self._waveform
        if waveform is None:
            sample_rate = self.sample_rate
            start = int(self.start_s * sample_rate)
            end = int(self.end_s * sample_rate) if self.end_s is not None else -1
            waveform, sr = load_audio(self.filepath, start=start, end=end)
            self._waveform = waveform
            self._sample_rate = sr
        return waveform

    @property
    def info(self):
        info = AudioInfo(
            filepath=self.filepath,
            sample_rate=self.sample_rate,
            duration_s=self.duration_s,
            loudness=self.loudness,
        )
        return info

    @property
    def duration_s(self):
        waveform = self.waveform
        sample_rate = self.sample_rate
        duration_s = waveform.shape[-1] / sample_rate
        return duration_s

    @property
    def n_frames(self):
        return self.waveform.shape[-1]

    @property
    def loudness(self):
        loudness = self._loudness
        if loudness is None:
            loudness = torchaudio.functional.loudness(
                self.waveform, sample_rate=self.sample_rate
            ).item()
        return loudness

    def resample(self, sample_rate: int):
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                self.waveform, orig_freq=self.sample_rate, new_freq=sample_rate
            )
            self._waveform = waveform
            self._sample_rate = sample_rate
        return self

    @classmethod
    def from_audioinfo(cls, audioinfo: AudioInfo):
        audio = cls(filepath=audioinfo.filepath)
        audio._sample_rate = audioinfo.sample_rate
        audio._duration_s = audioinfo.duration_s
        audio._loudness = audioinfo.loudness
        return audio

    def excerpt(self, offset_s: float, duration_s: float):
        waveform = self._waveform
        if waveform is not None:
            start = int(offset_s * self.sample_rate)
            end = start + int(duration_s * self.sample_rate)
            waveform = waveform[..., start:end]
        sample_rate = self._sample_rate

        excerpt = Audio(
            filepath=self.filepath,
            waveform=waveform,
            sample_rate=sample_rate,
            start_s=offset_s,
            end_s=offset_s + duration_s,
        )
        excerpt._loudness = self.loudness
        return excerpt

    def random_excerpt(self, duration_s: float, generator: torch.Generator = None):
        assert duration_s <= self.duration_s

        offset_s = torch.rand((1,), generator=generator).item()
        offset_s = (self.duration_s - duration_s) * offset_s
        excerpt = self.excerpt(offset_s=offset_s, duration_s=duration_s)
        return excerpt
