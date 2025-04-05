from dataclasses import dataclass

import torch

from senhance.data.audio import Audio
from senhance.data.augmentations.augmentations import (
    AugmentationParameters,
    BatchAugmentationParameters,
    STFTAugmentation,
)
from senhance.data.augmentations.distributions import Distribution


@dataclass(kw_only=True)
class SpecAugParameters(AugmentationParameters):
    start: torch.FloatTensor
    perc_mask: torch.FloatTensor


class SpecAugDim(STFTAugmentation):
    def __init__(
        self,
        perc_mask_distribution: Distribution,
        dim: int,
        name: str = "specaugdim",
        p: float = 1.0,
    ):
        super().__init__(name=name, p=p)

        self.perc_mask_distribution = perc_mask_distribution
        self.dim = dim
        self.other_dim = 1 + dim % 2

    def _sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator = None,
    ) -> SpecAugParameters:
        perc_mask = self.perc_mask_distribution.sample(generator=generator)
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

        device = stft.device
        n = stft.shape[self.dim]
        start = parameters.start.to(device)
        perc = parameters.perc_mask.to(device)
        arange = torch.arange(n, device=device)[None].unsqueeze(self.other_dim)
        perc = (n * perc).long()[:, None, None]
        start = (n * start).long()[:, None, None]
        end = start + perc
        mask = (start <= arange) * (arange < end)
        repeats = [1, 1, 1]
        repeats[self.other_dim] = stft.shape[self.other_dim]
        mask = mask.repeat(repeats)
        stft[apply] = torch.where(mask, torch.zeros_like(stft[apply]), stft[apply])
        return stft


class SpecAugFreq(SpecAugDim):
    def __init__(
        self,
        freq_perc_mask_distribution: Distribution,
        name: str = "specaug_freq",
        p: float = 1.0,
    ):
        super().__init__(
            perc_mask_distribution=freq_perc_mask_distribution,
            dim=1,
            name=name,
            p=p,
        )


class SpecAugTime(SpecAugDim):
    def __init__(
        self,
        time_perc_mask_distribution: Distribution,
        name: str = "specaug_time",
        p: float = 1.0,
    ):
        super().__init__(
            perc_mask_distribution=time_perc_mask_distribution,
            dim=2,
            name=name,
            p=p,
        )
