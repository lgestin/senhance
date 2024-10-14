from dataclasses import dataclass, fields, field

import torch

from denoiser.data.audio import Audio


@dataclass
class AugmentationParameters:
    apply: torch.BoolTensor = field(default_factory=lambda: torch.as_tensor(True))

    @classmethod
    def batch(
        cls, parameters: list["AugmentationParameters"]
    ) -> "BatchAugmentationParameters":
        return BatchAugmentationParameters(parameters)


@dataclass
class BatchAugmentationParameters:
    """
    each param is now a tensor to be used by Augmentation.augment
    """

    def __init__(self, parameters: list[AugmentationParameters]):
        self._parameters = parameters
        self._validate_parameters()
        self._batch_fields()

    def _validate_parameters(self):
        are_all_params_same = all(
            isinstance(param, type(self._parameters[0])) for param in self._parameters
        )
        assert are_all_params_same, "All parameters must be of the same type"

    def _batch_fields(self):
        for field in self.fields:
            values = [getattr(param, field.name) for param in self._parameters]
            batch = self._batch_values(values)
            setattr(self, field.name, batch)

    @staticmethod
    def _batch_values(values: list):
        if torch.is_tensor(values[0]):
            batch = torch.stack(values)
        elif isinstance(values[0], (int, float, bool)):
            batch = torch.as_tensor(values)
        elif isinstance(values[0], str):
            batch = values
        else:
            raise TypeError(f"Unsupported type for batching: {type(values[0])}")
        return batch

    @property
    def fields(self):
        return fields(self._parameters[0])

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
                value.to(device)
            elif isinstance(value, BatchAugmentationParameters):
                value.to(device)
            elif isinstance(value, list):
                if isinstance(value[0], BatchAugmentationParameters):
                    for val in value:
                        val.to(device)
            elif isinstance(value, dict):
                for val in value.values():
                    if isinstance(val, BatchAugmentationParameters):
                        val.to(device)
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
