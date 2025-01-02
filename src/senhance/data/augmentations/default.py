from pathlib import Path

import torch

from senhance.data.augmentations.background_noise import BackgroundNoise
from senhance.data.augmentations.chain import Chain
from senhance.data.augmentations.choose import Choose
from senhance.data.augmentations.filters import BandPassChain, HighPass, LowPass
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
    urbansound8k_path = (
        noise_folder / f"records/urbansound8k/data.{split}.arrow"
    )
    urabansound8k = ArrowAudioSource(
        arrow_file=urbansound8k_path.as_posix(),
        sequence_length_s=sequence_length_s + 0.1,
    )

    fsdnoisy18k_path = noise_folder / f"records/fsdnoisy18k/data.{split}.arrow"
    fsdnoisy18k = ArrowAudioSource(
        arrow_file=fsdnoisy18k_path.as_posix(),
        sequence_length_s=sequence_length_s + 0.1,
    )
    background_noise = Choose(
        BackgroundNoise(
            urabansound8k,
            min_snr=5.0,
            max_snr=25.0,
        ),
        BackgroundNoise(
            fsdnoisy18k,
            min_snr=5.0,
            max_snr=25.0,
        ),
        weights=[0.33, 0.67],
        p=0.8,
    )

    roy_james_path = noise_folder / f"irs/RoyJames/data.{split}.arrow"
    roy_james = ArrowAudioSource(arrow_file=roy_james_path.as_posix())
    reverb = Choose(
        Reverb(roy_james),
        weights=[1.0],
        p=0.4,
    )
    freqs_hz = torch.linspace(sample_rate // 4, sample_rate // 2, 10).tolist()

    low_passes = [LowPass(freq_hz=freq_hz) for freq_hz in freqs_hz]
    low_pass = Choose(*low_passes, p=1.0)

    high_passes = [HighPass(freq_hz=freq_hz) for freq_hz in freqs_hz]
    high_pass = Choose(*high_passes, p=1.0)

    bands_hz = [(bef, aft) for bef, aft in zip(freqs_hz[:-1], freqs_hz[1:])]
    band_passes = [BandPassChain(band_hz) for band_hz in bands_hz]
    band_pass = Choose(*band_passes, p=1.0)

    filters = Choose(
        low_pass,
        high_pass,
        band_pass,
        weights=[0.4, 0.4, 0.2],
        p=0.6,
    )
    augmentation = Chain(
        # silence,
        background_noise,
        reverb,
        filters,
        p=p,
    )
    return augmentation
