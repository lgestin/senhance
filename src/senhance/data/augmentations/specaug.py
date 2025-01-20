from dataclasses import dataclass

import torch

from senhance.data.audio import Audio
from senhance.data.augmentations.augmentations import (
    AugmentationParameters,
    BatchAugmentationParameters,
    STFTAugmentation,
)


@dataclass(kw_only=True)
class SpecAugParameters(AugmentationParameters):
    start: torch.FloatTensor
    perc_mask: torch.FloatTensor


class SpecAugDim(STFTAugmentation):
    def __init__(
        self,
        min_perc_mask: float,
        max_perc_mask: float,
        dim: int,
        name: str = "specaugdim",
        p: float = 1.0,
    ):
        super().__init__(name=name, p=p)

        self.min_perc_mask = min_perc_mask
        self.max_perc_mask = max_perc_mask
        self.dim = dim
        self.other_dim = 1 + dim % 2

    def _sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator = None,
    ) -> SpecAugParameters:
        perc_mask = torch.rand(tuple(), generator=generator)
        perc_mask = (
            perc_mask * (self.max_perc_mask - self.min_perc_mask)
            + self.min_perc_mask
        )
        start = torch.rand(tuple(), generator=generator)
        start *= 1 - perc_mask
        return SpecAugParameters(start=start, perc_mask=perc_mask)

    @torch.inference_mode()
    def _augment(
        self,
        stft: torch.FloatTensor,
        parameters: SpecAugParameters | BatchAugmentationParameters,
    ) -> torch.FloatTensor:
        if isinstance(parameters, AugmentationParameters):
            parameters = parameters.collate([parameters])

        if not torch.any(parameters.apply):
            return stft

        apply = parameters.apply

        n = stft.shape[self.dim]
        start, perc = parameters.start, parameters.perc_mask
        print(start, perc, n, self.dim, stft.shape)
        arange = torch.arange(n)[None].unsqueeze(self.other_dim)
        perc = (n * perc).long()[:, None, None]
        start = (n * start).long()[:, None, None]
        end = start + perc
        mask = (start <= arange) * (arange < end)
        repeats = [1, 1, 1]
        repeats[self.other_dim] = stft.shape[self.other_dim]
        mask = mask.repeat(repeats)
        stft[apply] = torch.where(
            mask, torch.zeros_like(stft[apply]), stft[apply]
        )
        return stft


class SpecAugFreq(SpecAugDim):
    def __init__(
        self,
        min_freq_perc_mask: float,
        max_freq_perc_mask: float,
        name: str = "specaug_freq",
        p: float = 1.0,
    ):
        super().__init__(
            min_perc_mask=min_freq_perc_mask,
            max_perc_mask=max_freq_perc_mask,
            dim=1,
            name=name,
            p=p,
        )


class SpecAugTime(SpecAugDim):
    def __init__(
        self,
        min_time_perc_mask: float,
        max_time_perc_mask: float,
        name: str = "specaug_time",
        p: float = 1.0,
    ):
        super().__init__(
            min_perc_mask=min_time_perc_mask,
            max_perc_mask=max_time_perc_mask,
            dim=2,
            name=name,
            p=p,
        )
