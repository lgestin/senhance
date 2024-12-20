from dataclasses import dataclass

import torch
import torchaudio.functional as F

from senhance.data.audio import Audio
from senhance.data.augmentations.augmentations import (
    Augmentation,
    AugmentationParameters,
    BatchAugmentationParameters,
)
from senhance.data.source import ArrowAudioSource


@dataclass(kw_only=True)
class ReverbParameters(AugmentationParameters):
    apply: torch.BoolTensor
    ir_filepath: str
    ir: torch.FloatTensor
    # drr: torch.FloatTensor


class Reverb(Augmentation):
    def __init__(
        self,
        ir_source: ArrowAudioSource,
        # min_drr: float,
        # max_drr: float,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.ir_source = ir_source
        self.data_folder = ir_source.arrow_file.parent

        # self.min_drr = min_drr
        # self.max_drr = max_drr

    def sample_parameters(
        self,
        audio: Audio,
        generator: torch.Generator = None,
    ) -> ReverbParameters:
        apply = torch.rand(tuple(), generator=generator) <= self.p
        # drr = (
        #     torch.rand(tuple(), generator=generator) * (self.max_drr - self.min_drr)
        #     + self.min_drr
        # )

        if apply:
            i = torch.randint(
                0, len(self.ir_source), size=(1,), generator=generator
            ).item()
            ir = self.ir_source[i]
            ir_filepath = ir.filepath
            ir = ir.mono().resample(audio.sample_rate)
            offset_s = ir.waveform.argmax() / ir.sample_rate
            ir = ir.excerpt(offset_s=offset_s)
            if audio.waveform.shape[-1] > ir.waveform.shape[-1]:
                ir = torch.nn.functional.pad(
                    ir.waveform,
                    (0, audio.waveform.shape[-1] - ir.waveform.shape[-1]),
                )
            else:
                ir = ir.waveform[..., : audio.waveform.shape[-1]]
        else:
            ir = torch.zeros_like(audio.waveform)
            ir[..., 0] = 1.0
            ir_filepath = ""
        return ReverbParameters(
            apply=apply,
            ir_filepath=ir_filepath,
            ir=ir,
            # drr=drr,
        )

    @torch.inference_mode()
    def augment(
        self,
        waveform: torch.FloatTensor,
        parameters: ReverbParameters | BatchAugmentationParameters,
    ) -> torch.FloatTensor:
        if isinstance(parameters, AugmentationParameters):
            parameters = parameters.collate([parameters])

        if not torch.any(parameters.apply):
            return waveform

        device = waveform.device
        ir = parameters.ir.to(device, non_blocking=True)
        apply = parameters.apply.to(device, non_blocking=True)

        ir = ir / torch.linalg.vector_norm(ir, ord=2)
        reverb = F.fftconvolve(waveform[apply], ir[apply])
        waveform[apply] = reverb[..., : waveform.shape[-1]]
        return waveform
