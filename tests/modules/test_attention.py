import math

import torch

from senhance.models.unet.attention import SelfAttention
from senhance.models.unet.magnitude_preserving import normalize


def edm_attention(qkv, n_heads):
    # https://github.com/NVlabs/edm2/blob/main/training/networks_edm2.py
    b, c, t = qkv.shape[0], qkv.shape[1] // 3, qkv.shape[2]
    qkv = qkv.reshape(qkv.shape[0], n_heads, -1, 3, qkv.shape[2] * 1)
    q, k, v = normalize(qkv, dim=2).unbind(3)  # pixel norm & split
    w = torch.einsum("nhcq,nhck->nhqk", q, k / math.sqrt(q.shape[2])).softmax(dim=3)
    y = torch.einsum("nhqk,nhck->nhcq", w, v)
    y = y.reshape(b, c, t)
    return y


def attention(qkv, n_heads):
    attn = SelfAttention.attention(qkv, n_heads)
    print(attn[0, :, 0])
    return attn


def test_attention():
    b, dim, n_heads, t = 3, 16, 4, 7
    assert dim % n_heads == 0
    qkv = torch.randn(b, dim * 3, t)

    edm_attn = edm_attention(qkv, n_heads)
    attn = attention(qkv, n_heads)
    assert torch.allclose(edm_attn, attn)


if __name__ == "__main__":
    test_attention()
