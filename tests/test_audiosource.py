from senhance.data.source import IndexAudioSource


def test_audiosource():
    asource = IndexAudioSource("/data/denoising/speech/daps/index.json")

    item = asource[0]
    assert item

    length = len(asource)
    assert length

    sequence_length_s = 0.5
    asource = IndexAudioSource(
        "/data/denoising/speech/daps/index.json",
        sequence_length_s=sequence_length_s,
    )

    for i in range(10):
        item = asource[i]
        assert item.waveform.shape[-1] / item.sample_rate == sequence_length_s
