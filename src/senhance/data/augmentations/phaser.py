from dataclasses import dataclass

import torch
import torchaudio.functional as F

from senhance.data.audio import Audio
from senhance.data.augmentations.augmentations import (
    Augmentation,
    AugmentationParameters,
    BatchAugmentationParameters,
)


@dataclass(kw_only=True)
class PhaserParameters(AugmentationParameters):
    apply: torch.BoolTensor
    sample_rate: torch.FloatTensor
    gain_in: torch.FloatTensor
    gain_out: torch.FloatTensor
    delay_ms: torch.FloatTensor
    decay: torch.FloatTensor
    mod_speed: torch.FloatTensor
    sinusoidal: torch.BoolTensor


class Phaser(Augmentation):
    def __init__(
        self,
        min_gain_in: float,
        max_gain_in: float,
        min_gain_out: float,
        max_gain_out: float,
        min_delay_ms: float,
        max_delay_ms: float,
        min_decay: float,
        max_decay: float,
        min_mod_speed: float,
        max_mod_speed: float,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        assert 0 <= min_gain_in <= max_gain_in <= 1
        assert 0 <= min_gain_out <= max_gain_out <= 1e9
        assert 0 <= min_delay_ms <= max_delay_ms <= 5
        assert 0 <= min_decay <= max_decay <= 0.99
        assert 0.1 <= min_mod_speed <= max_mod_speed <= 2
        self.min_gain_in = min_gain_in
        self.max_gain_in = max_gain_in
        self.min_gain_out = min_gain_out
        self.max_gain_out = max_gain_out
        self.min_delay_ms = min_delay_ms
        self.max_delay_ms = max_delay_ms
        self.min_decay = min_decay
        self.max_decay = max_decay
        self.min_mod_speed = min_mod_speed
        self.max_mod_speed = max_mod_speed

    def sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator = None,
    ) -> PhaserParameters:
        apply = torch.rand(tuple(), generator=generator) <= self.p
        sample_rate = audio.sample_rate
        gain_in = (
            torch.rand(tuple(), generator=generator)
            * (self.max_gain_in - self.min_gain_in)
            + self.min_gain_in
        )
        gain_out = (
            torch.rand(tuple(), generator=generator)
            * (self.max_gain_out - self.min_gain_out)
            + self.min_gain_out
        )
        gain_out = (
            torch.rand(tuple(), generator=generator)
            * (self.max_gain_out - self.min_gain_out)
            + self.min_gain_out
        )
        delay_ms = (
            torch.rand(tuple(), generator=generator)
            * (self.max_delay_ms - self.min_delay_ms)
            + self.min_delay_ms
        )
        decay = (
            torch.rand(tuple(), generator=generator) * (self.max_decay - self.min_decay)
            + self.min_decay
        )
        mod_speed = (
            torch.rand(tuple(), generator=generator)
            * (self.max_mod_speed - self.min_mod_speed)
            + self.min_mod_speed
        )
        sinusoidal = torch.rand(tuple(), generator=generator) <= 0.5
        return PhaserParameters(
            apply=apply,
            sample_rate=sample_rate,
            gain_in=gain_in,
            gain_out=gain_out,
            delay_ms=delay_ms,
            decay=decay,
            mod_speed=mod_speed,
            sinusoidal=sinusoidal,
        )

    @torch.inference_mode()
    def augment(
        self,
        waveform: torch.FloatTensor,
        parameters: PhaserParameters | BatchAugmentationParameters,
    ) -> torch.FloatTensor:
        if isinstance(parameters, AugmentationParameters):
            parameters = parameters.collate([parameters])

        if not torch.any(parameters.apply):
            return waveform

        apply = parameters.apply
        sample_rate = parameters.sample_rate.unique().item()
        augmented = waveform.clone()
        augmented = []
        for wav, gain_in, gain_out, delay_ms, decay, mod_speed, sinusoidal in zip(
            waveform[apply],
            parameters.gain_in[apply],
            parameters.gain_out[apply],
            parameters.delay_ms[apply],
            parameters.decay[apply],
            parameters.mod_speed[apply],
            parameters.sinusoidal[apply],
        ):
            aug = F.phaser(
                waveform=wav,
                sample_rate=sample_rate,
                gain_in=gain_in,
                gain_out=gain_out,
                delay_ms=delay_ms,
                decay=decay,
                mod_speed=mod_speed,
                sinusoidal=sinusoidal,
            )
            augmented.append(aug)
        augmented = torch.stack(augmented)
        return augmented
