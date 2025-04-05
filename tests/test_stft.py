import pytest
import torch

from senhance.data.audio import Audio
from senhance.data.stft import STFT

n_ffts = [2**i for i in range(5, 11)]
hop_length_ratios = [0.1, 0.25, 0.5]


@pytest.mark.parametrize("n_fft", n_ffts)
@pytest.mark.parametrize("hop_length_ratio", hop_length_ratios)
def test_stft_with_files(audio_from_file: Audio, n_fft: int, hop_length_ratio: float):
    hop_length = int(n_fft * hop_length_ratio)
    stft = STFT(n_fft=n_fft, hop_length=hop_length)

    x_stft = stft.stft(audio_from_file.waveform)
    x_istft = stft.istft(x_stft, length=audio_from_file.waveform.shape[-1])
    assert torch.is_tensor(x_stft)
    assert torch.is_tensor(x_istft)
    assert x_istft.shape == audio_from_file.waveform.shape
    torch.testing.assert_close(
        x_istft,
        audio_from_file.waveform,
        atol=3e-6,
        rtol=1e-5,
    )


@pytest.mark.parametrize("n_fft", n_ffts)
@pytest.mark.parametrize("hop_length_ratio", hop_length_ratios)
def test_stft_with_random(random_audio: Audio, n_fft: int, hop_length_ratio: float):
    hop_length = int(n_fft * hop_length_ratio)
    stft = STFT(n_fft=n_fft, hop_length=hop_length)

    x_stft = stft.stft(random_audio.waveform)
    x_istft = stft.istft(x_stft, length=random_audio.waveform.shape[-1])
    assert torch.is_tensor(x_stft)
    assert torch.is_tensor(x_istft)
    assert x_istft.shape == random_audio.waveform.shape
    torch.testing.assert_close(
        x_istft,
        random_audio.waveform,
        atol=3e-6,
        rtol=1e-5,
    )
