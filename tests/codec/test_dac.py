import pytest
import torch

from senhance.data.audio import Audio
from senhance.models.codec.dac import DescriptAudioCodec

from ..augments import AUDIO_TEST_FILES


@pytest.mark.parametrize("audio_file_path", AUDIO_TEST_FILES)
@torch.inference_mode()
def test_dac(audio_file_path):
    generator = torch.Generator().manual_seed(42)
    dac = DescriptAudioCodec("/data/models/dac/weights_24khz_8kbps_0.0.4.pth")

    audio = Audio(audio_file_path)
    audio = audio.resample(dac.sample_rate)
    audio = audio.random_excerpt(
        duration_s=64 / dac.resolution_hz,
        generator=generator,
    )

    waveform = audio.waveform[None]
    encoded = dac.encode(waveform)
    decoded = dac.decode(encoded)

    reconstructed = dac.reconstruct(waveform)

    assert torch.allclose(reconstructed, decoded)


if __name__ == "__main__":
    test_dac()
