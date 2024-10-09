import torch

from denoiser.data.audio import Audio
from denoiser.models.codec.dac import DescriptAudioCodec


@torch.inference_mode()
def test_dac():
    generator = torch.Generator().manual_seed(42)
    dac = DescriptAudioCodec("/data/models/dac/weights_24khz_8kbps_0.0.4.pth")

    audio = Audio("/data/denoising/speech/daps/clean/f10_script1_clean.wav")
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
