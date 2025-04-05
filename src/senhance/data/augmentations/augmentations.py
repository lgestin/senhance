from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from senhance.data.audio import Audio
from senhance.data.augmentations.config import load_augmentation_from_yaml
from senhance.data.augmentations.distributions import Binomial


@dataclass
class AugmentationParameters:
    def collate(
        self, parameters: list["AugmentationParameters"]
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
        self.collate_fields()

    def _validate_parameters(self):
        params_type = type(next(filter(lambda p: p is not None, self._parameters)))
        are_all_params_same = all(
            isinstance(param, (params_type, type(None))) for param in self._parameters
        )
        assert are_all_params_same, "All parameters must be of the same type"

    def collate_fields(self):
        apply = [param is not None for param in self._parameters]
        apply = torch.as_tensor(apply)
        setattr(self, "apply", apply)

        parameters = [param for param in self._parameters if param]
        for field in self.fields:
            values = [getattr(param, field.name) for param in parameters]
            batch = self._collate_values(values)
            setattr(self, field.name, batch)

    @staticmethod
    def _collate_values(values: list):
        if isinstance(values[0], np.ndarray):
            batch = np.stack(values)
        elif torch.is_tensor(values[0]):
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
        param = next(filter(lambda p: p is not None, self._parameters))
        return fields(param)

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

    def to(self, device: str | torch.device, non_blocking: bool = False):
        parameters = next(filter(lambda p: p is not None, self._parameters), None)
        if parameters is None:
            return self

        for field in fields(parameters):
            value = getattr(self, field.name)
            if torch.is_tensor(value):
                setattr(self, field.name, value.to(device, non_blocking=non_blocking))
            elif isinstance(value, BatchAugmentationParameters):
                value.to(device, non_blocking=non_blocking)
            elif isinstance(value, list):
                if isinstance(value[0], BatchAugmentationParameters):
                    for val in value:
                        val.to(device, non_blocking=non_blocking)
            elif isinstance(value, dict):
                for val in value.values():
                    if isinstance(val, BatchAugmentationParameters):
                        val.to(device, non_blocking=non_blocking)
        return self

    def pin_memory(self):
        """Pin memory for all tensor fields for faster GPU transfer."""
        parameters = next(filter(lambda p: p is not None, self._parameters), None)
        if parameters is None:
            return self

        for field in fields(parameters):
            value = getattr(self, field.name)
            if torch.is_tensor(value):
                setattr(self, field.name, value.pin_memory())
            elif isinstance(value, BatchAugmentationParameters):
                value.pin_memory()
            elif isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], BatchAugmentationParameters):
                    for val in value:
                        val.pin_memory()
            elif isinstance(value, dict):
                for val in value.values():
                    if isinstance(val, BatchAugmentationParameters):
                        val.pin_memory()
        return self

    @property
    def size(self):
        return len(self._parameters)


class Augmentation:
    def __init__(self, name: str | None = None, p: float = 1.0):
        super().__init__()
        assert 0 <= p <= 1.0
        self.prob = Binomial(count=1, prob=p)

        if name is None:
            name = self.__class__.__name__
        self.name = name

    @property
    def p(self) -> float:
        """Get the augmentation probability."""
        return self.prob.prob

    @p.setter
    def p(self, value: float):
        """Set the augmentation probability."""
        assert 0 <= value <= 1.0, f"Probability must be in [0, 1], got {value}"
        self.prob.prob = value

    def sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator | None = None,
    ) -> AugmentationParameters | None:
        params = None
        if self.prob.sample():
            params = self._sample_parameters(audio=audio, generator=generator)
        return params

    def _sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator | None = None,
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
        return self._augment(waveform=waveform, parameters=parameters)

    def _augment(
        self,
        waveform: torch.Tensor,
        parameters: BatchAugmentationParameters,
    ) -> torch.Tensor:
        """
        everything related to torch.Tensor
        """
        raise NotImplementedError

    def __call__(self, audio: Audio, generator: torch.Generator | None = None):
        parameters = self.sample_parameters(audio=audio, generator=generator)
        augmented = self.augment(audio.waveform, parameters=parameters)
        return augmented

    @classmethod
    def from_yaml(
        cls, path: Path | str, sequence_length_s: float, split: str = "train"
    ) -> "Augmentation":
        augmentation = load_augmentation_from_yaml(
            config_path=path, sequence_length_s=sequence_length_s, split=split
        )
        return augmentation


class STFTAugmentation(Augmentation):
    def augment(
        self,
        stft: torch.Tensor,
        parameters: BatchAugmentationParameters,
    ) -> torch.Tensor:
        return self._augment(stft=stft, parameters=parameters)

    def _augment(
        self,
        stft: torch.Tensor,
        parameters: BatchAugmentationParameters,
    ) -> torch.Tensor:
        """
        everything related to torch.Tensor
        """
        raise NotImplementedError

    def __call__(self, audio: Audio, generator: torch.Generator | None = None):
        parameters = self.sample_parameters(audio=audio, generator=generator)
        stft = Audio.stfter.stft(audio.waveform)
        length = audio.waveform.shape[-1]
        stft = self.augment(stft=stft, parameters=parameters)
        augmented = Audio.stfter.istft(stft, length=length)
        return augmented


class Identity(Augmentation):
    def sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator | None = None,
    ) -> AugmentationParameters:
        apply = torch.rand(tuple(), generator=generator) <= self.p
        return AugmentationParameters(apply=apply)

    def _augment(
        self,
        waveform: torch.Tensor,
        parameters: BatchAugmentationParameters,
    ) -> torch.Tensor:
        return waveform
