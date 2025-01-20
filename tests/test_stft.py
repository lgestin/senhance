import pytest
import torch

from senhance.data.audio import Audio
from senhance.data.stft import STFT, MelSpectrogram

from .augments import AUDIO_TEST_FILES

n_ffts = [2**i for i in range(5, 11)]


@pytest.mark.parametrize("audio_file_path", AUDIO_TEST_FILES)
@pytest.mark.parametrize("n_fft", n_ffts)
def test_stft(audio_file_path, n_fft):
    hop_lengths = [n_fft // i for i in range(2, 8, 2)]
    audio = Audio(audio_file_path)
    stfts = [
        STFT(n_fft=n_fft, hop_length=hop_length) for hop_length in hop_lengths
    ]

    for stft in stfts:
        x_stft = stft.stft(audio.waveform)
        x_istft = stft.istft(x_stft, length=audio.waveform.shape[-1])
        assert torch.is_tensor(x_stft)
        assert torch.is_tensor(x_istft)
        assert torch.all(torch.isclose(x_istft, audio.waveform, atol=1e-6))
