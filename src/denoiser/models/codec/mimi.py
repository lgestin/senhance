import torch
from moshi.models.compression import MimiModel
from moshi.models.loaders import _quantizer_kwargs, _seanet_kwargs, _transformer_kwargs
from moshi.modules import SEANetDecoder, SEANetEncoder, transformer
from moshi.quantization import SplitResidualVectorQuantizer
from safetensors.torch import load_model

from denoiser.models.codec.codec import Codec


class MimiCodec(Codec):
    def __init__(self, safetensors_path: str, device: str | torch.device = "cpu"):
        super().__init__(dim=512, sample_rate=24_000, resolution=12.5)
        encoder = SEANetEncoder(**_seanet_kwargs)
        decoder = SEANetDecoder(**_seanet_kwargs)
        encoder_transformer = transformer.ProjectedTransformer(
            device=device, **_transformer_kwargs
        )
        decoder_transformer = transformer.ProjectedTransformer(
            device=device, **_transformer_kwargs
        )
        quantizer = SplitResidualVectorQuantizer(**_quantizer_kwargs)

        mimi = MimiModel(
            encoder,
            decoder,
            quantizer,
            channels=1,
            sample_rate=24_000,
            frame_rate=12.5,
            encoder_frame_rate=24_000 / encoder.hop_length,
            causal=True,
            resample_method="conv",
            encoder_transformer=encoder_transformer,
            decoder_transformer=decoder_transformer,
        ).to(device)
        mimi.eval()

        load_model(mimi, safetensors_path)
        self.mimi = mimi

    def encode(self, x: torch.Tensor):
        return self.mimi.encode_to_latent(x, quantize=False)

    def decode(self, x: torch.Tensor):
        codes = self.mimi.quantizer.encode(x)
        decoded = self.mimi.decode(codes)
        return decoded

    def reconstruct(self, x: torch.Tensor):
        encoded = self.mimi.encode(x)
        decoded = self.mimi.decode(encoded)
        return decoded
