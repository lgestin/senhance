from dataclasses import dataclass, fields

import torch

from denoiser.data.audio import Audio


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
        for field in fields(parameters[0]):
            values = [getattr(param, field.name) for param in parameters]
            if torch.is_tensor(values[0]):
                batch = torch.stack(values)
            elif isinstance(values[0], (int, float, bool)):
                batch = torch.as_tensor(values)
            elif isinstance(values[0], str):
                batch = values
            else:
                batch = []
                for i in range(len(values[0])):
                    value = [val[i] for val in values]
                    batch.append(BatchAugmentationParameters(value))
            setattr(self, field.name, batch)
        return self

    def __getitem__(self, idx: int):
        return self._parameters[idx]

    def to(self, device: str | torch.device):
        for field in fields(self._parameters[0]):
            value = getattr(self, field.name)
            if torch.is_tensor(value):
                value = value.to(device)
                setattr(self, field.name, value)
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
