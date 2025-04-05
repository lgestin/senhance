from dataclasses import dataclass

import torch

from senhance.data.audio import Audio
from senhance.data.augmentations.augmentations import (
    Augmentation,
    AugmentationParameters,
    BatchAugmentationParameters,
)
from senhance.data.augmentations.distributions import Distribution


@dataclass(kw_only=True)
class ClippingParameters(AugmentationParameters):
    clip_percentile: torch.FloatTensor


class Clipping(Augmentation):
    def __init__(
        self,
        clip_percentile_distribution: Distribution,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.clip_percentile_distribution = clip_percentile_distribution

    def _sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator | None = None,
    ) -> ClippingParameters:
        clip_percentile = self.clip_percentile_distribution.sample(generator=generator)
        return ClippingParameters(clip_percentile=clip_percentile)

    @torch.inference_mode()
    def augment(
        self,
        waveform: torch.FloatTensor,
        parameters: ClippingParameters | BatchAugmentationParameters | None,
    ) -> torch.FloatTensor:
        if parameters is None:
            return waveform
        if isinstance(parameters, AugmentationParameters):
            parameters = parameters.collate([parameters])
        if not torch.any(parameters.apply):
            return waveform

        apply = parameters.apply
        sign = torch.sign(waveform[apply])
        clip_percentile = parameters.clip_percentile
        quantile = [
            torch.quantile(wav.abs(), 1 - perc.item(), dim=-1, keepdim=True)
            for wav, perc in zip(waveform[apply], clip_percentile)
        ]
        quantile = torch.stack(quantile)

        waveform[apply] = sign * waveform[apply].abs().clamp(max=quantile)
        return waveform
