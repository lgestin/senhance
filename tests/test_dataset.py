import torch

from denoiser.data.source import AudioSource
from denoiser.data.dataset import AudioDataset
from denoiser.data.augmentations import TestTransform

from audiotools import AudioSignal


def test_denoising_dataset():
    audio_source = AudioSource("../../data/daps/clean/index.json")
    transform = TestTransform()

    dataset = AudioDataset(audio_source=audio_source, sample_rate=16_000)

    for i in range(10):
        item = dataset[i]
        clean = item.waveform
        asig = AudioSignal(clean, item.sample_rate)
        params = transform.instantiate(item.idx, signal=asig)
        noisy = transform.transform(asig, **params)
        assert item

    clean = AudioSignal(dataset[i].waveform, dataset[i].sample_rate)
    noisy2 = transform.transform(
        clean, **transform.instantiate(dataset[i].idx, signal=clean)
    )
    assert torch.allclose(noisy.audio_data, noisy2.audio_data)


if __name__ == "__main__":
    test_denoising_dataset()
