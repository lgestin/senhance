"""Shared pytest fixtures for all tests."""

import pytest
import torch

from senhance.data.audio import Audio

from .augments import AUDIO_TEST_FILES, TEST_SEED


@pytest.fixture(params=AUDIO_TEST_FILES)
def audio_from_file(request) -> Audio:
    """Fixture that provides Audio objects loaded from test files."""
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
    """Fixture that provides randomly generated Audio objects with various configurations."""
    channels, sample_rate, num_samples = request.param
    generator = torch.Generator().manual_seed(TEST_SEED)
    waveform = torch.randn(channels, num_samples, generator=generator)
    return Audio(waveform=waveform, sample_rate=sample_rate)
