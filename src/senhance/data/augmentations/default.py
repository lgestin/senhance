import torch
from senhance.data.augmentations.chain import Chain
from senhance.data.augmentations.choose import Choose
from senhance.data.augmentations.silence import Silence
from senhance.data.augmentations.background_noise import BackgroundNoise
from senhance.data.augmentations.reverb import Reverb
from senhance.data.augmentations.filters import LowPass, HighPass, BandPassChain


def get_default_augmentation(sequence_length_s: float, split: str, p: float):
    silence = Silence(p=0.025)
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
    freqs_hz = torch.linspace(500, 11000, 100).tolist()
    bands_hz = [[before, after] for before, after in zip(freqs_hz[:-1], freqs_hz[1:])]
    filters = Choose(
        LowPass(freqs_hz=freqs_hz),
        HighPass(freqs_hz=freqs_hz),
        BandPassChain(bands_hz=bands_hz),
        weights=[0.4, 0.4, 0.2],
        p=0.6,
    )
    augmentation = Chain(
        silence,
        background_noise,
        reverb,
        filters,
        p=p,
    )
    # augmentation = background_noise
    return augmentation
