import pytest
import torch

from senhance.data.audio import Audio
from senhance.data.stft import STFT

from .augments import AUDIO_TEST_FILES, TEST_SEED

n_ffts = [2**i for i in range(5, 11)]
hop_length_ratios = [0.1, 0.25, 0.5]


@pytest.fixture(params=AUDIO_TEST_FILES)
def audio_from_file(request) -> Audio:
    return Audio(request.param)


@pytest.fixture(
    params=[
        (1, 16000, 16000),  # Mono, 1 second, 16kHz
        (2, 24000, 24000),  # Stereo, 1 second, 24kHz
        (1, 48000, 96000),  # Mono, 2 seconds, 48kHz
        (2, 22050, 44100),  # Stereo, 2 seconds, 22.05kHz
        (1, 8000, 4000),  # Mono, 0.5 seconds, 8kHz
    ]
)
def random_audio(request) -> Audio:
    channels, sample_rate, num_samples = request.param
    generator = torch.Generator().manual_seed(TEST_SEED)
    waveform = torch.randn(channels, num_samples, generator=generator)
    return Audio(waveform=waveform, sample_rate=sample_rate)


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
