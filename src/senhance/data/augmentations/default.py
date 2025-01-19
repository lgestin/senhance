from pathlib import Path

import torch

from senhance.data.augmentations.background_noise import BackgroundNoise
from senhance.data.augmentations.chain import Chain
from senhance.data.augmentations.choose import Choose
from senhance.data.augmentations.filters import (
    BandPassChain,
    HighPass,
    LowPass,
    LowPassResample,
)
from senhance.data.augmentations.random_noise import RandomNoise
from senhance.data.augmentations.reverb import Reverb
from senhance.data.augmentations.silence import Silence
from senhance.data.source import ArrowAudioSource


def get_default_augmentation(
    noise_folder: str,
    sample_rate: int,
    sequence_length_s: float,
    split: str,
    p: float,
):
    if not isinstance(noise_folder, Path):
        noise_folder = Path(noise_folder)

    silence = Silence(p=0.025)

    random_noise = RandomNoise(min_amplitude=0.001, max_amplitude=0.03, p=0.5)

    background_noises = []
    background_noise_paths = [
        "urbansound8k",
        "fsdnoisy18k",
        "arca23k",
        "FSD50k",
        "DNC",
        "DEMAND/48k",
        "musan/noise",
        "musan/music",
    ]
    for path in background_noise_paths:
        arrow_file = noise_folder / "records" / path / f"data.{split}.arrow"
        source = ArrowAudioSource(
            arrow_file=arrow_file,
            sequence_length_s=sequence_length_s + 0.1,
            is_speech=False,
        )
        background_noise = BackgroundNoise(
            source,
            min_snr=5,
            max_snr=25,
            name=path,
        )
        background_noises.append(background_noise)
    background_noise = Choose(
        *background_noises,
        name="background_noise",
        p=0.9,
    )

    irs = []
    ir_source_paths = ["RoyJames", "EchoThief", "MITMcDermott"]
    for path in ir_source_paths:
        arrow_file = noise_folder / "irs" / path / f"data.{split}.arrow"
        source = ArrowAudioSource(
            arrow_file=arrow_file,
            sequence_length_s=sequence_length_s + 0.1,
            is_speech=False,
        )
        ir = Reverb(source, name=path)
        irs.append(ir)
    reverb = Choose(
        *irs,
        name="irs",
        p=0.4,
    )

    low_pass_freqs_hz = torch.tensor(
        [
            0.0,
            0.3,
            0.5,
            0.6,
            0.7,
            0.8,
            0.85,
            0.9,
            0.92,
            0.94,
            0.96,
            0.98,
            1.0,
        ]
    )
    high_pass_freqs_hz = 1 - low_pass_freqs_hz
    low_pass_freqs_hz = (
        sample_rate // 4
        + (sample_rate // 2 - sample_rate // 4) * low_pass_freqs_hz
    )
    low_passes = [
        LowPassResample(freq_hz=freq_hz) for freq_hz in low_pass_freqs_hz
    ]
    low_pass = Choose(*low_passes, name="low_passes", p=1.0)

    high_pass_freqs_hz = (
        sample_rate // 4
        + (sample_rate // 2 - sample_rate // 4) * high_pass_freqs_hz
    )
    high_passes = [HighPass(freq_hz=freq_hz) for freq_hz in high_pass_freqs_hz]
    high_pass = Choose(*high_passes, name="high_passes", p=1.0)

    freqs_hz = (
        torch.linspace(sample_rate // 3, sample_rate // 2, 15).long().tolist()
    )
    bands_hz = [(bef, aft) for bef, aft in zip(freqs_hz[:-1], freqs_hz[1:])]
    band_passes = [BandPassChain(band_hz) for band_hz in bands_hz]
    band_pass = Choose(*band_passes, name="band_passes", p=1.0)

    filters = Choose(
        low_pass,
        high_pass,
        band_pass,
        weights=[0.4, 0.4, 0.2],
        name="filters",
        p=0.4,
    )
    augmentation = Chain(
        # silence,
        random_noise,
        background_noise,
        reverb,
        filters,
        p=p,
    )
    return augmentation
