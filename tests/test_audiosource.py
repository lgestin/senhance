from denoiser.data.source import AudioSource


def test_audiosource():
    asource = AudioSource("../../data/daps/clean/index.json")

    item = asource[0]
    assert item

    length = len(asource)
    assert length

    seq_len = 8192
    asource = AudioSource("../../data/daps/clean/index.json", sequence_length=seq_len)

    for i in range(10):
        item = asource[i]
        assert item.waveform.shape[-1] == seq_len
