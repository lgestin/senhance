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
    params: list[AugmentationParameters]

    def collate(
        self, parameters: list["ChainParameters"]
    ) -> dict[AugmentationParameters, BatchAugmentationParameters]:
        return BatchChainParameters(parameters)


class BatchChainParameters(BatchAugmentationParameters):
    def collate_fields(self):
        parameters: list[ChainParameters | None] = self._parameters

        apply = [p is not None for p in parameters]
        apply = torch.as_tensor(apply)
        self.apply = apply

        parameters = [p for p in parameters if p]

        if not parameters:
            self.params = None
            return

        chain_parameters = []
        n_augmentations = len(parameters[0].params)
        for i in range(n_augmentations):
            # collate individual augmentations params
            augment_params = [params.params[i] for params in parameters]
            ref_param = next((p for p in augment_params if p is not None), None)
            if ref_param:
                augment_params = ref_param.collate(augment_params)
            else:
                augment_params = None
            chain_parameters.append(augment_params)

        self.params = chain_parameters


class Chain(Augmentation):
    def __init__(
        self,
        *augmentations: Augmentation,
        name: str = "chain",
        p: float = 1.0,
    ):
        super().__init__(name=name, p=p)
        self.augmentations = augmentations

    def __getitem__(self, idx: int):
        return self.augmentations[idx]

    def __len__(self):
        return len(self.augmentations)

    def _sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator = None,
    ) -> ChainParameters:
        augment_parameters = []
        for augmentation in self.augmentations:
            parameters = augmentation.sample_parameters(
                audio=audio, generator=generator
            )
            augment_parameters.append(parameters)
        return ChainParameters(params=augment_parameters)

    def _augment(
        self,
        waveform: torch.Tensor,
        parameters: ChainParameters | BatchChainParameters,
    ) -> torch.Tensor:
        if isinstance(parameters, ChainParameters):
            parameters = BatchChainParameters([parameters])

        if parameters is None or (not torch.any(parameters.apply)):
            return waveform

        augmented = waveform[parameters.apply]
        for augment_i, params_i in zip(
            self.augmentations, parameters.params, strict=True
        ):
            augmented = augment_i.augment(
                waveform=augmented,
                parameters=params_i,
            )
        waveform[parameters.apply] = augmented
        return waveform
