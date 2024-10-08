import math
import json
import torch
from dataclasses import dataclass
from pathlib import Path

from denoiser.data.audio import Audio, AudioInfo
from denoiser.data.utils import truncated_normal


@dataclass
class AugmentationParameters:
    apply: bool = True


class BatchAugmentationParameters:
    """
    each param is now a tensor to be used by Augmentation.augment
    """

    def __init__(self, parameters: list[AugmentationParameters]):
        self.collate(parameters)
        self._parameters = parameters

    def collate(self, parameters: list[AugmentationParameters]):
        assert all(isinstance(param, type(parameters[0])) for param in parameters)
        for attr in parameters[0].__annotations__.keys():
            values = [getattr(param, attr) for param in parameters]
            if torch.is_tensor(values[0]):
                batch = torch.stack(values)
            setattr(self, attr, batch)
        return self

    def __getitem__(self, idx: int):
        return self._parameters[idx]

    def to(self, device: str | torch.device):
        for key in self._parameters[0].__annotations__.keys():
            value = getattr(self, key)
            if torch.is_tensor(value):
                value = value.to(device)
                setattr(self, key, value)
        return self


class Augmentation:
    def __init__(self, name: str = None, p: float = 1.0):
        super().__init__()
        assert 0 <= p <= 1.0
        self.p = p

        if name is None:
            name = self.__class__.__name__
        self.name = name

    def sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator = None,
    ) -> AugmentationParameters:
        """
        anything not related to torch.Tensor
        getting settings and preparing audios
        """
        raise NotImplementedError

    def augment(
        self,
        waveform: torch.Tensor,
        parameters: BatchAugmentationParameters,
    ) -> torch.Tensor:
        """
        everything related to torch.Tensor
        """
        raise NotImplementedError

    def __call__(self, audio: Audio, generator: torch.Generator = None):
        parameters = self.sample_parameters(audio=audio, generator=generator)
        augmented = self.augment(audio.waveform, parameters=parameters)
        return augmented


@dataclass
class ChainParameters:
    apply: torch.BoolTensor
    params: list[AugmentationParameters]


class Chain(Augmentation):
    def __init__(self, *augmentations: Augmentation, p: float = 1.0):
        super().__init__(p=p)
        self.augmentations = augmentations

    def sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator = None,
    ) -> ChainParameters:
        apply = torch.rand(tuple(), generator=generator) <= self.p

        augment_parameters = []
        for augmentation in self.augmentations:
            parameters = augmentation.sample_parameters(
                audio=audio, generator=generator
            )
            augment_parameters.append(parameters)

        return ChainParameters(apply=apply, params=augment_parameters)

    def augment(
        self,
        waveform: torch.Tensor,
        parameters: ChainParameters,
    ) -> torch.Tensor:
        if not torch.any(parameters.apply):
            return waveform

        for augmentation, params in zip(
            self.augmentations, parameters.params, strict=True
        ):
            waveform = augmentation.augment(waveform=waveform, parameters=params)
        return waveform


@dataclass
class ChooseParameters:
    apply: torch.BoolTensor
    choice: int
    params: list[AugmentationParameters | None]


class Choose(Augmentation):
    def __init__(
        self,
        *augmentations: Augmentation,
        weights: list[float] | torch.FloatTensor = None,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        n = len(augmentations)
        if weights is None:
            weights = torch.full((n,), 1 / n)
        if not torch.is_tensor(weights):
            weights = torch.as_tensor(weights)

        assert weights.ndim == 1
        assert weights.shape[0] == n
        assert weights.sum().item() == 1.0
        self.weights = weights

        self.augmentations = augmentations

    def sample_parameters(
        self,
        audio: Audio,
        generator: AugmentationParameters = None,
    ) -> list[AugmentationParameters | None]:
        apply = torch.rand(tuple(), generator=generator) <= self.p
        choice = torch.multinomial(self.weights, 1).item()

        params = [None for _ in self.augmentations]
        params[choice] = self.augmentations[choice].sample_parameters(
            audio=audio, generator=generator
        )
        return ChooseParameters(apply=apply, choice=choice, params=params)

    def augment(
        self,
        waveform: torch.Tensor,
        parameters: ChooseParameters,
    ) -> torch.Tensor:
        if not torch.any(parameters.apply):
            return waveform

        for augmentation, params in zip(
            self.augmentations, parameters.params, strict=True
        ):
            if params is None:
                continue
            waveform = augmentation.augment(waveform=waveform, parameters=params)
        return waveform


@dataclass
class BackgroundNoiseParameters:
    apply: torch.BoolTensor
    noise: torch.FloatTensor
    snr: torch.FloatTensor
    clean_loudness: torch.FloatTensor
    noise_loudness: torch.FloatTensor


class BackgroundNoise(Augmentation):
    def __init__(
        self,
        noise_index_path: str,
        min_snr: float,
        max_snr: float,
        min_duration_s: float = 0.0,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.data_folder = Path(noise_index_path).parent
        with open(noise_index_path, "r") as f:
            noise_index = json.load(f)
        noise_index = filter(
            lambda index: index["duration_s"] > min_duration_s, noise_index
        )
        self.index = list(noise_index)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def load_noise(self, index: dict):
        audioinfo = AudioInfo(**index)
        audioinfo.filepath = (self.data_folder / audioinfo.filepath).as_posix()
        noise = Audio.from_audioinfo(audioinfo)
        return noise

    def sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator = None,
    ) -> BackgroundNoiseParameters:
        apply = torch.rand(tuple(), generator=generator) <= self.p

        if apply:
            i = torch.randint(0, len(self.index), size=(1,), generator=generator).item()
            index = self.index[i]
            noise = self.load_noise(index)
            noise = noise.random_excerpt(
                duration_s=audio.duration_s,
                generator=generator,
            )
            noise = noise.mono().resample(audio.sample_rate)
        else:
            zeros = torch.zeros_like(audio.waveform)
            noise = Audio(waveform=zeros, sample_rate=audio.sample_rate)
            noise._loudness = -70.0

        snr = truncated_normal(tuple(), min_val=self.min_snr, max_val=self.max_snr)
        clean_loudness = torch.as_tensor(audio.loudness).to(device=audio.device)
        noise_loudness = torch.as_tensor(noise.loudness).to(device=noise.device)

        return BackgroundNoiseParameters(
            apply=apply,
            noise=noise.waveform,
            snr=snr,
            clean_loudness=clean_loudness,
            noise_loudness=noise_loudness,
        )

    def augment(
        self,
        waveform: torch.FloatTensor,
        parameters: BatchAugmentationParameters,
    ) -> torch.FloatTensor:

        if not torch.any(parameters.apply):
            return waveform

        clean_loudness = parameters.clean_loudness[parameters.apply]
        noise_loudness = parameters.noise_loudness[parameters.apply]
        snr = parameters.snr[parameters.apply]

        gain = clean_loudness - noise_loudness - snr
        gain = torch.exp(math.log(10) / 20 * gain)
        gain = gain.view(-1, 1, 1)

        augmented = waveform.clone()
        augmented[parameters.apply] = waveform[parameters.apply] + (
            gain * parameters.noise[parameters.apply]
        )
        return augmented


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
