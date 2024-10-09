from dataclasses import dataclass

import torch

from denoiser.data.audio import Audio
from denoiser.data.augmentations.augmentations import Augmentation


@dataclass
class ClippingParameters:
    apply: torch.BoolTensor
    clip_percentile: torch.FloatTensor


class Clipping(Augmentation):
    def __init__(
        self,
        min_clip_percentile: float,
        max_clip_percentile: float,
        p: float = 1.0,
    ):
        super().__init__(p=p)

        self.min_clip_percentile = min_clip_percentile
        self.max_clip_percentile = max_clip_percentile

    def sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator = None,
    ) -> ClippingParameters:
        apply = torch.rand(tuple(), generator=generator) <= self.p
        clip_percentile = (
            torch.rand(tuple(), generator=generator)
            * (self.max_clip_percentile - self.min_clip_percentile)
            + self.min_clip_percentile
        )
        return ClippingParameters(apply=apply, clip_percentile=clip_percentile)

    @torch.inference_mode()
    def augment(
        self,
        waveform: torch.FloatTensor,
        parameters: ClippingParameters,
    ) -> torch.FloatTensor:

        if not torch.any(parameters.apply):
            return waveform

        apply = parameters.apply
        sign = torch.sign(waveform[apply])
        clip_percentile = parameters.clip_percentile[apply]
        quantile = [
            torch.quantile(wav.abs(), 1 - perc.item(), dim=-1, keepdim=True)
            for wav, perc in zip(waveform[apply], clip_percentile)
        ]
        quantile = torch.stack(quantile)

        augmented = waveform.clone()
        augmented[apply] = sign * waveform[apply].abs().clamp(max=quantile)
        return augmented
