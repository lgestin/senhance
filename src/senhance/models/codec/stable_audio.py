import torch
from diffusers import AutoencoderOobleck

from senhance.models.codec.codec import Codec


class StableAudioVAE(Codec):
    def __init__(self, device: str | torch.device = "cpu"):
        super().__init__(
            dim=64, sample_rate=44_100, resolution_hz=44_100 / 2048
        )
        self.vae = AutoencoderOobleck.from_pretrained(
            "stabilityai/stable-audio-open-1.0", subfolder="vae"
        )
        self.vae.eval()
        self.vae.to(device)

    def encode(self, x: torch.Tensor):
        x = x.repeat(1, 2, 1)
        return self.vae.encode(x).latent_dist.sample()

    def decode(self, z: torch.Tensor):
        return self.vae.decode(z).sample.mean(1, keepdim=True)

    def reconstruct(self, x: torch.Tensor):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

    def normalize(self, z):
        return z / 6.0

    def unnormalize(self, z):
        return 6.0 * z
