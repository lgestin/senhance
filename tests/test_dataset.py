from denoiser.data.dataset import AudioDataset
from denoiser.data.source import AudioSource


def test_denoising_dataset():
    audio_source = AudioSource("/data/denoising/speech/daps/index.json")

    dataset = AudioDataset(audio_source=audio_source, sample_rate=16_000)

    for i in range(10):
        item = dataset[i]
        assert item


if __name__ == "__main__":
    test_denoising_dataset()
