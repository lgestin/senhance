import torch

from denoiser.data.audio import Audio
from denoiser.models.codec.mimi import MimiCodec


@torch.inference_mode()
def test_mimi():
    generator = torch.Generator().manual_seed(42)
    mimi = MimiCodec("../../models/moshi/tokenizer-e351c8d8-checkpoint125.safetensors")

    audio = Audio("/data/denoising/speech/daps/clean/f10_script1_clean.wav")
    audio = audio.resample(mimi.sample_rate)
    audio = audio.random_excerpt(duration_s=64 / 12.5, generator=generator)

    waveform = audio.waveform[None]
    encoded = mimi.encode(waveform)
    decoded = mimi.decode(encoded)

    reconstructed = mimi.reconstruct(waveform)

    assert torch.allclose(reconstructed, decoded)


if __name__ == "__main__":
    test_mimi()
