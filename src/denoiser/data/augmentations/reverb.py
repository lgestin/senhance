import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio.functional as F

from denoiser.data.audio import Audio, AudioInfo
from denoiser.data.augmentations.augmentations import (
    Augmentation,
    AugmentationParameters,
    BatchAugmentationParameters,
)


@dataclass(kw_only=True)
class ReverbParameters(AugmentationParameters):
    apply: torch.BoolTensor
    ir_filepath: str
    ir: torch.FloatTensor
    # drr: torch.FloatTensor


class Reverb(Augmentation):
    def __init__(
        self,
        ir_index_path: str,
        # min_drr: float,
        # max_drr: float,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.data_folder = Path(ir_index_path).parent
        with open(ir_index_path, "r") as f:
            ir_index = json.load(f)
        self.index = ir_index

        # self.min_drr = min_drr
        # self.max_drr = max_drr

    def load_ir(self, index: dict):
        audioinfo = AudioInfo(**index)
        audioinfo.filepath = (self.data_folder / audioinfo.filepath).as_posix()
        ir = Audio.from_audioinfo(audioinfo)
        return ir

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
            i = torch.randint(0, len(self.index), size=(1,), generator=generator).item()
            index = self.index[i]
            ir = self.load_ir(index)
            ir_filepath = ir.filepath
            ir = ir.mono().resample(audio.sample_rate)
            offset_s = ir.waveform.argmax() / ir.sample_rate
            ir = ir.excerpt(offset_s=offset_s)
            if audio.waveform.shape[-1] > ir.waveform.shape[-1]:
                ir = torch.nn.functional.pad(
                    ir.waveform, (0, audio.waveform.shape[-1] - ir.waveform.shape[-1])
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
            parameters = parameters.batch([parameters])

        if not torch.any(parameters.apply):
            return waveform

        device = waveform.device
        ir = parameters.ir.to(device, non_blocking=True)
        apply = parameters.apply.to(device, non_blocking=True)

        ir = ir / torch.linalg.vector_norm(ir, ord=2)
        augmented = waveform.clone()
        reverb = F.fftconvolve(waveform[apply], ir[apply])
        reverb = reverb[..., : waveform.shape[-1]]
        augmented[apply] = reverb
        return augmented
