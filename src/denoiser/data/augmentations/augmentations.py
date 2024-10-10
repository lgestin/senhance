from dataclasses import dataclass, fields, field

import torch

from denoiser.data.audio import Audio


@dataclass
class AugmentationParameters:
    apply: torch.BoolTensor = field(default_factory=lambda: torch.as_tensor(False))

    def batch(
        self, parameters: list["AugmentationParameters"]
    ) -> "BatchAugmentationParameters":
        return BatchAugmentationParameters(parameters)


class BatchAugmentationParameters:
    """
    each param is now a tensor to be used by Augmentation.augment
    """

    def __init__(self, parameters: list[AugmentationParameters]):
        self._parameters = parameters

        assert all(isinstance(param, type(parameters[0])) for param in parameters)
        for field in fields(parameters[0]):
            values = [getattr(param, field.name) for param in parameters]
            if torch.is_tensor(values[0]):
                batch = torch.stack(values)
            elif isinstance(values[0], (int, float, bool)):
                batch = torch.as_tensor(values)
            elif isinstance(values[0], str):
                batch = values
            else:
                raise TypeError
            setattr(self, field.name, batch)

    def __getitem__(self, idx: int | torch.BoolTensor):
        if isinstance(idx, int):
            item = self._parameters[idx]
        elif torch.is_tensor(idx):
            assert idx.ndim == 1
            parameters = [
                param for param, apply in zip(self._parameters, idx) if apply.item()
            ]
            item = self.__class__(parameters=parameters)
        return item

    def to(self, device: str | torch.device):
        for field in fields(self._parameters[0]):
            value = getattr(self, field.name)
            if torch.is_tensor(value):
                value = value.to(device)
                setattr(self, field.name, value)
        return self

    @property
    def size(self):
        return len(self._parameters)


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


class Identity(Augmentation):
    def sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator = None,
    ) -> AugmentationParameters:
        apply = torch.rand(tuple(), generator=generator) <= self.p
        return AugmentationParameters(apply=apply)

    def augment(
        self,
        waveform: torch.Tensor,
        parameters: BatchAugmentationParameters,
    ) -> torch.Tensor:
        apply = parameters.apply
        augmented = waveform.clone()
        augmented[apply] = waveform[apply]
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
