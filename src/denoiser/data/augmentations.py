import math
import json
import torch
from dataclasses import dataclass
from pathlib import Path

from denoiser.data.audio import Audio, AudioInfo
from denoiser.data.utils import truncated_normal


class Augmentation:
    def __init__(self, p: float = 1.0):
        assert 0 <= p <= 1.0
        self.p = p

    def parameters(self, generator: torch.Generator):
        raise NotImplementedError

    def augment(self, x: Audio, generator: torch.Generator = None) -> Audio:
        raise NotImplementedError

    def __call__(self, x: Audio, apply: bool = True, generator: torch.Generator = None):
        augmented = x
        if apply:
            augmented = self.augment(augmented, generator=generator)
        return augmented


@dataclass
class BackgroundNoiseParameters:
    noise: Audio
    snr: float
    apply: bool = True


class BackgroundNoise(Augmentation):
    def __init__(
        self, noise_index_path: str, min_snr: float, max_snr: float, p: float = 1.0
    ):
        super().__init__(p=p)
        self.data_folder = Path(noise_index_path).parent
        with open(noise_index_path, "r") as f:
            noise_index = json.load(f)
        self.index = noise_index

        self.min_snr = min_snr
        self.max_snr = max_snr

    def parameters(self, generator: torch.Generator = None):
        apply = torch.rand((1,), generator=generator).item() <= self.p

        i = torch.randint(0, len(self.index), size=(1,), generator=generator).item()
        audioinfo = AudioInfo(**self.index[i])
        audioinfo.filepath = (self.data_folder / audioinfo.filepath).as_posix()
        noise = Audio.from_audioinfo(audioinfo)

        snr = truncated_normal((1,), min_val=self.min_snr, max_val=self.max_snr).item()

        return BackgroundNoiseParameters(
            noise=noise,
            snr=snr,
            apply=apply,
        )

    def augment(self, x: Audio, generator: torch.Generator = None) -> Audio:
        parameters = self.parameters(generator=generator)

        noise = parameters.noise
        noise = noise.random_excerpt(duration_s=x.duration_s, generator=generator)
        noise = noise.resample(x.sample_rate)
        assert noise.sample_rate == x.sample_rate

        if not parameters.apply:
            return x

        snr = x.loudness - noise.loudness - parameters.snr
        gain = math.exp(math.log(10) / 20 * snr)
        print(gain, noise.loudness, x.loudness)
        waveform = x.waveform + gain * noise.waveform
        augmented = Audio(waveform=waveform, sample_rate=x.sample_rate)
        return augmented, noise


class TestTransform:
    def __init__(self):
        from audiotools import ransforms as tfm

        self.transform = tfm.Compose(
            tfm.LowPass(cutoff=("uniform", 4000, 8000)),
            tfm.ClippingDistortion(),
            tfm.TimeMask(),
        )

    def transform(self, waveform: torch.Tensor, params: dict) -> torch.Tensor:
        return self.transform(waveform, **params)

    def instantiate(self, state=None, signal=None):
        return self.transform.instantiate(state=state, signal=signal)
