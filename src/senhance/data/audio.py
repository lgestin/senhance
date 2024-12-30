import math
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as F

from senhance.data.utils import load_audio, resample


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
        waveform: torch.FloatTensor = None,
        sample_rate: int = None,
        start_s: float = 0,
        end_s: float = None,
    ):
        assert isinstance(filepath, (type(None), str, Path))
        assert waveform is None or torch.is_tensor(waveform)
        if filepath is None:
            assert isinstance(sample_rate, int)
            assert waveform is not None and sample_rate is not None

        self.filepath = filepath
        self._waveform = waveform
        self._sample_rate = sample_rate
        self.start_s = start_s
        self.end_s = end_s
        self._loudness = None

    def to(self, device: str | torch.device):
        self.waveform.to(device, non_blocking=True)
        return self

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
            end = -1
            if self.end_s:
                end = int(self.end_s * sample_rate)
            waveform, sr = load_audio(
                path=self.filepath,
                sample_rate=sample_rate,
                start=start,
                end=end,
            )
            waveform = torch.from_numpy(waveform) / 32678.0
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
            waveform = self.waveform
            loudness = F.loudness(waveform, sample_rate=self.sample_rate).item()
        if math.isnan(loudness):
            loudness = -70.0
        self._loudness = loudness
        return loudness

    def mono(self):
        waveform = self.waveform
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        self._waveform = waveform
        return self

    def resample(self, sample_rate: int):
        waveform = self.waveform
        waveform = (32678 * waveform).to(torch.int16).numpy()
        resampled = resample(
            waveform=waveform,
            orig_sr=self.sample_rate,
            targ_sr=sample_rate,
        )
        resampled = torch.from_numpy(resampled) / 32678.0
        self._waveform = resampled
        self._sample_rate = sample_rate
        return self

    def normalize(self, db: float):
        gain = db - self.loudness
        gain = math.exp(math.log(10) / 20 * gain)
        self._waveform = gain * self.waveform
        self._loudness = db
        return self

    @classmethod
    def from_audioinfo(cls, audioinfo: AudioInfo):
        audio = cls(filepath=audioinfo.filepath)
        audio._sample_rate = audioinfo.sample_rate
        audio._duration_s = audioinfo.duration_s
        audio._loudness = audioinfo.loudness
        return audio

    def excerpt(self, offset_s: float, duration_s: float = None):
        waveform = self._waveform
        if waveform is not None:
            start = int(offset_s * self.sample_rate)
            if duration_s is not None:
                end = start + int(duration_s * self.sample_rate)
            else:
                duration_s = self.duration_s - offset_s
                end = waveform.shape[-1]
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

    def random_excerpt(
        self,
        duration_s: float,
        generator: torch.Generator = None,
    ):
        assert duration_s <= self.duration_s

        offset_s = torch.rand(tuple(), generator=generator).item()
        offset_s = (self.duration_s - duration_s) * offset_s
        excerpt = self.excerpt(offset_s=offset_s, duration_s=duration_s)
        return excerpt

    def salient_excerpt(
        self,
        duration_s: float,
        loudness_threshold: float = -60.0,  # -40, -60
        n_tries: int = 10,
        generator: torch.Generator = None,
    ):
        assert 0.5 <= duration_s <= self.duration_s
        loudness = self.loudness
        excerpt = self.random_excerpt(
            duration_s=duration_s,
            generator=generator,
        )
        excerpt._loudness = None

        n_try = 0
        while (excerpt.loudness < loudness_threshold) and (n_try < n_tries):
            excerpt = self.random_excerpt(
                duration_s=duration_s,
                generator=generator,
            )
            excerpt._loudness = None
            n_try += 1
        excerpt._loudness = loudness
        return excerpt

    @property
    def device(self):
        return self.waveform.device
