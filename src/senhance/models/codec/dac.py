import torch
from dac import DAC

from senhance.models.codec.codec import Codec


class DescriptAudioCodec(Codec):
    def __init__(self, path: str, device: str | torch.device = "cpu"):
        super().__init__(dim=1024, sample_rate=24_000, resolution_hz=75)
        self.dac = DAC.load(path)
        self.dac.eval()
        self.dac.to(device)

    def encode(self, x: torch.Tensor):
        return self.dac.encoder(x)

    def decode(self, z: torch.Tensor):
        quantized = self.dac.quantizer(z)[0]
        return self.dac.decode(quantized)

    def reconstruct(self, x: torch.Tensor):
        encoded = self.dac.encode(x)[0]
        decoded = self.dac.decode(encoded)
        return decoded

    def normalize(self, z):
        return z / 3.5

    def unnormalize(self, z):
        return 3.5 * z
