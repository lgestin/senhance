from dataclasses import dataclass

import torch

from senhance.data.audio import Audio
from senhance.data.augmentations.augmentations import (
    Augmentation,
    AugmentationParameters,
    BatchAugmentationParameters,
)


@dataclass(kw_only=True)
class ChainParameters(AugmentationParameters):
    apply: torch.BoolTensor
    params: list[AugmentationParameters]

    def collate(
        self, parameters: list["ChainParameters"]
    ) -> dict[AugmentationParameters, BatchAugmentationParameters]:
        return BatchChainParameters(parameters)


class BatchChainParameters(BatchAugmentationParameters):
    def collate_fields(self):
        apply = torch.stack([param.apply for param in self._parameters])
        chain_parameters = []
        for i in range(len(self._parameters[0].params)):
            parameters_ = [params.params[i] for params in self._parameters]
            chain_parameters_ = parameters_[0].collate(parameters_)
            # apply = chain_parameters_.apply & chain_apply
            chain_parameters.append(chain_parameters_)

        self.apply = apply
        self.params = chain_parameters


class Chain(Augmentation):
    def __init__(self, *augmentations: Augmentation, p: float = 1.0):
        super().__init__(p=p)
        self.augmentations = augmentations

    def __getitem__(self, idx: int):
        return self.augmentations[idx]

    def __len__(self):
        return len(self.augmentations)

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
        parameters: ChainParameters | BatchChainParameters,
    ) -> torch.Tensor:
        if isinstance(parameters, ChainParameters):
            parameters = BatchChainParameters([parameters])

        if not torch.any(parameters.apply):
            return waveform

        chain_apply = parameters.apply
        for augmentation, params in zip(
            self.augmentations, parameters.params, strict=True
        ):
            apply = chain_apply & params.apply
            if not torch.any(apply):
                continue
            waveform[apply] = augmentation.augment(
                waveform=waveform[apply],
                parameters=params[apply],
            )
        return waveform
