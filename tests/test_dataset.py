from senhance.data.dataset import AudioDataset
from senhance.data.source import IndexAudioSource


def test_denoising_dataset():
    audio_source = IndexAudioSource("/data/denoising/speech/daps/index.json")

    dataset = AudioDataset(audio_source=audio_source, sample_rate=16_000)

    for i in range(10):
        item = dataset[i]
        assert item


if __name__ == "__main__":
    test_denoising_dataset()
