# Author: Yu XU <xuyu@genomics.cn>
# Created: 2026-04-14
"""ATAC Transformer encoder (RoPE-aligned) for fusion predictor.

Outputs the same shape convention as the original CNN-based ATAC encoder: [B, 1024, L].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Minimal RoPE implementation with configurable theta.

    This is intentionally self-contained so it works across HF model versions.
    """

    def __init__(self, head_dim: int, rope_theta: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
        self.head_dim = int(head_dim)
        self.rope_theta = float(rope_theta)
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [B, L] (int64)
        freqs = torch.einsum("bl,d->bld", position_ids.to(device=device, dtype=self.inv_freq.dtype), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # [B, L, head_dim]
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        return cos, sin


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q/k.

    q/k: [B, H, L, D]
    cos/sin: [B, L, D]
    """
    cos = cos.unsqueeze(1)  # [B, 1, L, D]
    sin = sin.unsqueeze(1)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


@dataclass
class ATACTransformerEncoderConfig:
    d_low: int = 192
    n_layers: int = 6
    n_heads: int = 4
    ffn_mult: int = 4
    output_dim: int = 1024
    dropout: float = 0.1
    attn_dropout: float = 0.1


class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float, attn_dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got {d_model=} {n_heads=}")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        self.dropout = float(dropout)
        self.attn_dropout = float(attn_dropout)

        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.o = nn.Linear(d_model, d_model, bias=True)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Linear(d_model, ffn_dim, bias=True)
        self.ffn2 = nn.Linear(ffn_dim, d_model, bias=True)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        h = self.ln1(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)

        b, l, _ = q.shape
        q = q.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,L,D]
        k = k.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        # SDPA: [B,H,L,D] -> [B,H,L,D]
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, l, self.d_model)
        attn_out = self.o(attn_out)
        x = x + F.dropout(attn_out, p=self.dropout, training=self.training)

        h = self.ln2(x)
        h = self.ffn2(F.gelu(self.ffn1(h)))
        x = x + F.dropout(h, p=self.dropout, training=self.training)
        return x


class ATAC_TransformerEncoder(nn.Module):
    """Transformer encoder for ATAC signal aligned with DNA RoPE coordinates."""

    def __init__(self, base_model, *, cfg: ATACTransformerEncoderConfig | None = None):
        super().__init__()
        self.cfg = cfg or ATACTransformerEncoderConfig()

        d_low = int(self.cfg.d_low)
        n_heads = int(self.cfg.n_heads)
        if d_low % n_heads != 0:
            raise ValueError(f"ATAC encoder d_low must be divisible by n_heads, got {d_low=} {n_heads=}")
        head_dim = d_low // n_heads

        rope_theta = float(getattr(getattr(base_model, "config", None), "rope_theta", 10000.0))
        self.rope = RotaryEmbedding(head_dim=head_dim, rope_theta=rope_theta)

        self.in_proj = nn.Linear(1, d_low, bias=True)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_low,
                    n_heads=n_heads,
                    ffn_dim=int(d_low * int(self.cfg.ffn_mult)),
                    dropout=float(self.cfg.dropout),
                    attn_dropout=float(self.cfg.attn_dropout),
                )
                for _ in range(int(self.cfg.n_layers))
            ]
        )
        self.out_proj = nn.Linear(d_low, int(self.cfg.output_dim), bias=True)

        self._ensure_float32()

    def _ensure_float32(self) -> None:
        for p in self.parameters():
            p.data = p.data.float()

    def forward(self, atac_signal: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Encode ATAC signal.

        atac_signal: [B, L]
        position_ids: [B, L] int64 (must match DNA token coordinates)
        returns: [B, 1024, L]
        """
        if atac_signal.ndim != 2:
            raise ValueError(f"atac_signal must be [B,L], got {tuple(atac_signal.shape)}")
        if position_ids.ndim != 2:
            raise ValueError(f"position_ids must be [B,L], got {tuple(position_ids.shape)}")

        b, l = atac_signal.shape
        x = atac_signal.unsqueeze(-1).float()  # [B,L,1]
        x = self.in_proj(x)  # [B,L,d_low]

        cos, sin = self.rope(position_ids, device=x.device, dtype=x.dtype)  # [B,L,head_dim]
        for layer in self.layers:
            x = layer(x, cos=cos, sin=sin)

        x = self.out_proj(x)  # [B,L,1024]
        return x.transpose(1, 2).contiguous()  # [B,1024,L]

