from dataclasses import dataclass

import torch

from senhance.data.audio import Audio
from senhance.data.augmentations.augmentations import (
    Augmentation,
    AugmentationParameters,
    BatchAugmentationParameters,
    STFTAugmentation,
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
        stft, length = None, augmented.shape[-1]
        original_shape = augmented.shape
        batch_size, n_channels = augmented.shape[0], augmented.shape[1]

        for i, (augment_i, params_i) in enumerate(
            zip(self.augmentations, parameters.params, strict=True)
        ):
            if isinstance(augment_i, STFTAugmentation):
                if stft is None:
                    # Flatten batch and channels for STFT: [batch, channels, samples] -> [batch*channels, samples]
                    augmented_flat = augmented.reshape(batch_size * n_channels, -1)
                    stft = Audio.stfter.stft(augmented_flat[:, None, :])
                    length = augmented.shape[-1]

                # Expand parameters to match flattened batch*channels dimension
                if params_i is not None and n_channels > 1:
                    params_i = self._expand_stft_parameters(params_i, n_channels)

                stft = augment_i.augment(stft, parameters=params_i)
            else:
                if stft is not None:
                    # Restore original shape after ISTFT
                    augmented_flat = Audio.stfter.istft(stft, length=length)
                    augmented = augmented_flat.reshape(batch_size, n_channels, -1)
                    stft = None
                augmented = augment_i.augment(augmented, parameters=params_i)
        if stft is not None:
            augmented_flat = Audio.stfter.istft(stft, length=length)
            augmented = augmented_flat.reshape(batch_size, n_channels, -1)

        # Avoid in-place operation to prevent stride issues with multi-stream training
        result = waveform.clone()
        result[parameters.apply] = augmented
        return result

    def _expand_stft_parameters(
        self, params: BatchAugmentationParameters, n_channels: int
    ) -> BatchAugmentationParameters:
        """Expand parameters to handle flattened batch*channels dimension."""
        # Expand apply mask: [batch] -> [batch*channels]
        params.apply = params.apply.repeat_interleave(n_channels)

        # Expand each parameter field
        for field in params.fields:
            value = getattr(params, field.name)
            if torch.is_tensor(value) and value.dim() > 0:
                # Repeat each batch element n_channels times
                expanded = value.repeat_interleave(n_channels, dim=0)
                setattr(params, field.name, expanded)

        return params
