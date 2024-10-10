import torch
from denoiser.data.augmentations.chain import Chain
from denoiser.data.augmentations.choose import Choose
from denoiser.data.augmentations.background_noise import BackgroundNoise
from denoiser.data.augmentations.reverb import Reverb
from denoiser.data.augmentations.filters import LowPass, HighPass, BandPass


def get_default_augmentation(sequence_length_s: float, split: str, p: float):
    background_noise = Choose(
        BackgroundNoise(
            f"/data/denoising/noise/records/urbansound8k/index.{split}.json",
            min_snr=5.0,
            max_snr=25.0,
            min_duration_s=sequence_length_s,
        ),
        BackgroundNoise(
            f"/data/denoising/noise/records/fsdnoisy18k/index.{split}.json",
            min_snr=5.0,
            max_snr=25.0,
            min_duration_s=sequence_length_s,
        ),
        weights=[0.33, 0.67],
        p=0.8,
    )
    reverb = Choose(
        Reverb(f"/data/denoising/noise/irs/RoyJames/index.{split}.json"),
        weights=[1.0],
        p=0.4,
    )
    freqs_hz = torch.linspace(500, 23000, 100).tolist()
    bands_hz = [[before, after] for before, after in zip(freqs_hz[:-1], freqs_hz[1:])]
    filters = Choose(
        LowPass(freqs_hz=freqs_hz),
        HighPass(freqs_hz=freqs_hz),
        BandPass(bands_hz=bands_hz),
        weights=[0.4, 0.4, 0.2],
        p=0.6,
    )
    augmentation = Chain(
        background_noise,
        reverb,
        filters,
        p=p,
    )
    return augmentation
