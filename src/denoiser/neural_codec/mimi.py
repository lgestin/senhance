import torch

from denoiser.neural_codec.codec import Codec
from moshi.models.loaders import get_mimi


class MimiCodec(Codec):
    def __init__(self, safetensors_path: str):
        super().__init__()
        mimi = get_mimi(safetensors_path)
        self.mimi = mimi

    def encode(self, x: torch.Tensor):
        return self.mimi.encode_latent(x)

    def decode(self, x: torch.Tensor):
        return self.mimi.decode_latent(x)
