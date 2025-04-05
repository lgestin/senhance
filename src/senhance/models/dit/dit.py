import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from senhance.models.attention import QKNorm, apply_rope, precompute_freqs_cis
from senhance.models.magnitude_preserving import SiLU


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by MLP."""

    def __init__(self, t_dim: int, dim: int):
        super().__init__()
        self.t_dim = t_dim

        # Precompute sinusoidal frequencies as a buffer
        half_dim = t_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            / (half_dim - 1)
            * torch.arange(half_dim, dtype=torch.float32)
        )
        self.register_buffer("freqs", freqs)

        self.mlp = nn.Sequential(
            nn.Linear(t_dim, dim),
            SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, timestep: torch.FloatTensor) -> torch.Tensor:
        """Embed timesteps.

        Args:
            timestep: Timestep values of shape (B,) in range [0, 1]

        Returns:
            Timestep embeddings of shape (B, dim)
        """
        # Sinusoidal embedding using precomputed frequencies
        # Cast freqs to the same dtype as timestep for compatibility
        freqs = self.freqs.to(timestep.dtype)
        emb = timestep[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # MLP projection
        return self.mlp(emb)


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 1.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        # Attention
        self.qk_norm = QKNorm(dim=self.head_dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj_attn = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)

        # MLP
        self.proj = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim)),
            SiLU(),
            nn.Linear(int(mlp_ratio * dim), dim),
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, L, dim) - includes timestep token
            freqs_cis: RoPE frequencies for attention

        Returns:
            Output tensor of shape (B, L, dim)
        """
        # Attention
        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.n_heads)
        q, k, v = self.qk_norm(q, k, v)
        q, k = apply_rope(q, k, freqs_cis)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = rearrange(attn, "B H L D -> B L (H D)")
        attn = self.proj_attn(attn)
        x = x + attn

        # MLP
        x = x + self.proj(self.norm2(x))

        return x


@dataclass
class DiTDims:
    in_dim: int
    dim: int
    n_layers: int
    t_dim: int


class DiT(nn.Module):
    def __init__(self, dims: DiTDims, n_heads: int = 8, mlp_ratio: float = 1.5):
        super().__init__()
        self.dims = dims
        self.n_heads = n_heads

        # Timestep embedding
        self.timestep_emb = TimestepEmbedding(dims.t_dim, dims.dim)

        # Input projection
        self.input_proj = nn.Linear(dims.in_dim, dims.dim)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                DiTBlock(dims.dim, n_heads=n_heads, mlp_ratio=mlp_ratio)
                for _ in range(dims.n_layers)
            ]
        )

        # Output projection
        self.output_proj = nn.Linear(dims.dim, dims.in_dim)

        # Cache for RoPE frequencies
        self._freqs_cis: torch.Tensor | None = None
        self._max_seq_len: int = 0

    def _get_freqs_cis(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or compute RoPE frequencies for given sequence length."""
        head_dim = self.dims.dim // self.n_heads

        # Recompute if sequence is longer than cached or device changed
        if (
            self._freqs_cis is None
            or seq_len > self._max_seq_len
            or self._freqs_cis.device != device
        ):
            with torch.no_grad():
                self._freqs_cis = precompute_freqs_cis(seq_len, head_dim, device=device)
            self._max_seq_len = seq_len

        # Clone to avoid inference mode issues when tensor was created in inference_mode
        return self._freqs_cis[:, :, :seq_len].clone()

    def forward(self, x_t: torch.Tensor, timestep: torch.FloatTensor) -> torch.Tensor:
        """Forward pass with timestep as a token.

        Args:
            x_t: Input tensor of shape (B, L, in_dim)
            timestep: Timestep values of shape (B,) in range [0, 1]

        Returns:
            Output tensor of shape (B, L, in_dim)
        """
        x_t = x_t.transpose(1, 2)
        B, L, _ = x_t.shape

        # Embed timestep as a token: (B, dim) -> (B, 1, dim)
        t_emb = self.timestep_emb(timestep).unsqueeze(1)

        # Project input to hidden dimension
        x = self.input_proj(x_t)

        # Prepend timestep token: (B, L, dim) -> (B, L+1, dim)
        x = torch.cat([t_emb, x], dim=1)

        # Get RoPE frequencies for sequence length L+1
        freqs_cis = self._get_freqs_cis(L + 1, x.device)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, freqs_cis)

        # Remove timestep token and project back: (B, L+1, dim) -> (B, L, in_dim)
        x = x[:, 1:, :]
        x = self.output_proj(x)

        x = x.transpose(1, 2)
        return x


if __name__ == "__main__":
    dims = DiTDims(in_dim=16, dim=64, n_layers=3, t_dim=256)
    dit = DiT(dims)

    with torch.inference_mode():
        x = torch.randn(2, dims.in_dim, 27)
        timestep = torch.rand(tuple())  # Random timesteps in [0, 1]

        output = dit(x, timestep)
        print(f"Input shape: {x.shape}")
        print(f"Timestep shape: {timestep.shape}")
        print(f"Output shape: {output.shape}")
        assert output.shape == x.shape, "Output shape must match input shape"
