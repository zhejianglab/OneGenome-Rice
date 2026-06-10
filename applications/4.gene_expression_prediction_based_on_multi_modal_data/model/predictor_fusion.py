# Author: Yu XU <xuyu@genomics.cn>
# Created: 2026-04-14
"""Fusion predictor with bidirectional cross-attention at L/4 and dual-skip injection."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.distributed import dist_print


def _shannon_entropy(probs: torch.Tensor, dim: int) -> torch.Tensor:
    """Shannon entropy in nats: -sum p log p along `dim`."""
    p = torch.clamp(probs, min=1e-8)
    return -(p * torch.log(p)).sum(dim=dim)


class CrossAttentionSDPA(nn.Module):
    def __init__(self, d_model: int = 1024, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, got {d_model=} {num_heads=}")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = d_model // num_heads
        self.dropout = float(dropout)

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor) -> torch.Tensor:
        # q_in, kv_in: [B, L, d]
        b, l, d = q_in.shape
        q = self.q_proj(q_in).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,L,D]
        k = self.k_proj(kv_in).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_in).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(b, l, d)
        return self.o_proj(out)


class MultiModalPredictorFusion(nn.Module):
    def __init__(
        self,
        base_model,
        atac_encoder,
        *,
        fusion_gate_entropy_frac: float = 0.0,
        skip_gate_entropy_frac: float = 0.0,
        unfreeze_base_last_layer: bool = False,
    ):
        super().__init__()
        self.base = base_model
        self.atac_encoder = atac_encoder
        self.fusion_gate_entropy_frac = float(fusion_gate_entropy_frac)
        self.skip_gate_entropy_frac = float(skip_gate_entropy_frac)
        self.unfreeze_base_last_layer = bool(unfreeze_base_last_layer)

        for p in self.base.parameters():
            p.requires_grad = False

        # Locate transformer layers for manual forward (common HF layer stack layouts)
        if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
            self.layers = base_model.model.layers
        elif hasattr(base_model, "encoder") and hasattr(base_model.encoder, "layer"):
            self.layers = base_model.encoder.layer
        elif hasattr(base_model, "transformer") and hasattr(base_model.transformer, "h"):
            self.layers = base_model.transformer.h
        elif hasattr(base_model, "layers"):
            self.layers = base_model.layers
        else:
            raise RuntimeError("Cannot identify transformer layer structure in base model")
        if not self.layers:
            raise RuntimeError("No transformer layers found")
        if self.unfreeze_base_last_layer:
            for param in self.layers[-1].parameters():
                param.requires_grad = True
            dist_print(
                f"Identified {len(self.layers)} transformer layers "
                "(fusion predictor; last block trainable, others frozen)"
            )
        else:
            dist_print(
                f"Identified {len(self.layers)} transformer layers (fusion predictor; base frozen)"
            )

        if hasattr(self.base, "model") and hasattr(self.base.model, "rotary_emb"):
            self.rotary_emb = self.base.model.rotary_emb
        elif hasattr(self.base, "rotary_emb"):
            self.rotary_emb = self.base.rotary_emb
        else:
            raise RuntimeError("Cannot locate rotary_emb module in base model")

        # DNA downsample for fusion at L/4
        self.dna_downsample = nn.Conv1d(1024, 1024, kernel_size=5, stride=4, padding=2)

        # ATAC conv pyramid (provides skips + X_atac_ds)
        self.enc1_atac = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.enc2_atac = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.enc3_atac = nn.Sequential(
            nn.Conv1d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Bidirectional cross-attn at L/4
        self.cross_a2d = CrossAttentionSDPA(d_model=1024, num_heads=8, dropout=0.1)
        self.cross_d2a = CrossAttentionSDPA(d_model=1024, num_heads=8, dropout=0.1)

        # 4-way gate at L/4 (token-wise)
        self.gate_mlp = nn.Sequential(
            nn.Linear(4096, 512),
            nn.GELU(),
            nn.Linear(512, 4),
        )

        # Post-fusion bottleneck at L/4
        self.post_fusion_bottleneck = nn.Sequential(
            nn.Conv1d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(1024, 1024, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(1024, 1024, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Upsample decoder
        self.up1 = nn.ConvTranspose1d(1024, 1024, kernel_size=4, stride=2, padding=1)  # L/4 -> L/2
        self.dec1 = nn.Sequential(
            nn.Conv1d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.up2 = nn.ConvTranspose1d(1024, 512, kernel_size=4, stride=2, padding=1)  # L/2 -> L
        self.dec2 = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Dual-skip at full resolution
        self.dna_skip_proj = nn.Conv1d(1024, 512, kernel_size=1)
        self.atac_skip_proj = nn.Conv1d(1024, 512, kernel_size=1)
        self.skip_gate = nn.Conv1d(1536, 2, kernel_size=1)  # [dec2, dna_skip, atac_skip] -> logits
        self.skip_merge = nn.Conv1d(1024, 512, kernel_size=3, padding=1)  # [dec2, skip_mix] -> merged

        # Final head
        self.final = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 256, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 256, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 2, kernel_size=1),
        )

        self.scale = nn.Parameter(torch.zeros(1))
        self._init_gates()
        self._ensure_float32()

    def _init_gates(self) -> None:
        # Gate MLP bias init: [-0.2, +0.3, 0.0, 0.0]
        with torch.no_grad():
            last = self.gate_mlp[-1]
            if isinstance(last, nn.Linear) and last.bias is not None and last.bias.numel() == 4:
                last.bias[:] = torch.tensor([-0.2, 0.3, 0.0, 0.0], dtype=last.bias.dtype, device=last.bias.device)

    def _ensure_float32(self) -> None:
        for module in [
            self.dna_downsample,
            self.enc1_atac, self.enc2_atac, self.enc3_atac,
            self.cross_a2d, self.cross_d2a,
            self.gate_mlp,
            self.post_fusion_bottleneck,
            self.up1, self.dec1, self.up2, self.dec2,
            self.dna_skip_proj, self.atac_skip_proj, self.skip_gate, self.skip_merge,
            self.final,
        ]:
            for p in module.parameters():
                p.data = p.data.float()
        self.scale.data = self.scale.data.float()

    def _encode_dna(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Return X_dna [B, 1024, L].

        - If ``unfreeze_base_last_layer`` is false: run the full base stack under ``no_grad``.
        - If ``unfreeze_base_last_layer`` is true: run all but the last transformer block under
          ``no_grad`` (base frozen), then run the last block with autograd enabled so gradients
          propagate to that block's parameters.
        """
        if not self.unfreeze_base_last_layer:
            with torch.no_grad():
                inputs_embeds = self.base.get_input_embeddings()(input_ids)
                position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
                hidden_states = inputs_embeds
                for layer in self.layers:
                    out = layer(hidden_states, position_embeddings=position_embeddings)
                    hidden_states = out[0] if isinstance(out, tuple) else out
            return hidden_states.transpose(1, 2).float()  # [B,1024,L] float32

        # Unfreeze last block: keep earlier computation frozen, but allow grad through final block.
        with torch.no_grad():
            inputs_embeds = self.base.get_input_embeddings()(input_ids)
            position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
            hidden_states = inputs_embeds
            for layer in self.layers[:-1]:
                out = layer(hidden_states, position_embeddings=position_embeddings)
                hidden_states = out[0] if isinstance(out, tuple) else out

        out = self.layers[-1](hidden_states, position_embeddings=position_embeddings)
        hidden_states = out[0] if isinstance(out, tuple) else out
        return hidden_states.transpose(1, 2).float()  # [B,1024,L] float32

    def forward(self, input_ids: torch.Tensor, atac_signal: torch.Tensor, labels: torch.Tensor | None = None):
        device = input_ids.device
        bsz, seq_len = input_ids.shape
        position_ids = (
            torch.arange(seq_len, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(bsz, -1)
        )

        # DNA features (frozen) and ATAC features (trainable)
        x_dna = self._encode_dna(input_ids, position_ids)  # [B,1024,L]
        x_atac = self.atac_encoder(atac_signal, position_ids)  # [B,1024,L] float32

        # Downsample for fusion @ L/4
        dna_ds = self.dna_downsample(x_dna)  # [B,1024,L/4]
        e1 = self.enc1_atac(x_atac)          # [B,512,L]
        e2 = self.enc2_atac(e1)              # [B,1024,L/2]
        e3 = self.enc3_atac(e2)              # [B,1024,L/4]  (X_atac_ds)

        # Cross-attn uses [B, L/4, 1024]
        q_atac = e3.transpose(1, 2)  # [B,L/4,1024]
        kv_dna = dna_ds.transpose(1, 2)
        q_dna = kv_dna
        kv_atac = q_atac

        cross_a2d = self.cross_a2d(q_atac, kv_dna)  # [B,L/4,1024]
        cross_d2a = self.cross_d2a(q_dna, kv_atac)  # [B,L/4,1024]

        # 4-way gate fuse (token-wise)
        atac_self = q_atac
        dna_self = q_dna
        gate_in = torch.cat([atac_self, dna_self, cross_a2d, cross_d2a], dim=-1)  # [B,L/4,4096]
        w = F.softmax(self.gate_mlp(gate_in), dim=-1)  # [B,L/4,4]
        h_fusion = _shannon_entropy(w, dim=-1)  # [B, L/4]
        fusion_gate_entropy_mean = h_fusion.mean()

        fused = (
            w[..., 0:1] * atac_self +
            w[..., 1:2] * dna_self +
            w[..., 2:3] * cross_a2d +
            w[..., 3:4] * cross_d2a
        )  # [B,L/4,1024]
        fused_ds = fused.transpose(1, 2)  # [B,1024,L/4]

        # Post-fusion refine + U-Net upsample
        b = self.post_fusion_bottleneck(fused_ds)
        d1 = self.up1(b)
        if d1.size(2) != e2.size(2):
            d1 = F.interpolate(d1, size=e2.size(2), mode="nearest")
        d1 = self.dec1(torch.cat([d1, e2], dim=1))

        d2 = self.up2(d1)
        if d2.size(2) != e1.size(2):
            d2 = F.interpolate(d2, size=e1.size(2), mode="nearest")
        d2 = self.dec2(torch.cat([d2, e1], dim=1))  # [B,512,L]

        # Dual-skip gate at full resolution (learnable weights)
        dna_skip = self.dna_skip_proj(x_dna)     # [B,512,L]
        atac_skip = self.atac_skip_proj(x_atac)  # [B,512,L]
        skip_logits = self.skip_gate(torch.cat([d2, dna_skip, atac_skip], dim=1))  # [B,2,L]
        skip_w = F.softmax(skip_logits, dim=1)
        h_skip = _shannon_entropy(skip_w, dim=1)  # [B, L]
        skip_gate_entropy_mean = h_skip.mean()

        skip_mix = skip_w[:, 0:1] * dna_skip + skip_w[:, 1:2] * atac_skip  # [B,512,L]
        merged = self.skip_merge(torch.cat([d2, skip_mix], dim=1))          # [B,512,L]

        logits = self.final(merged)
        logits = F.softplus(logits) * F.softplus(self.scale)

        loss = None
        mse_loss = None
        if labels is not None:
            mse_loss = F.mse_loss(logits, labels.to(logits.dtype))
            loss = mse_loss
            # Entropy regularization as a fraction of MSE, using normalized entropy H / log(K).
            # Scale by mse_loss.detach() so the entropy term's magnitude follows MSE without
            # changing the MSE gradient itself.
            if self.fusion_gate_entropy_frac > 0.0:
                h4_norm = fusion_gate_entropy_mean / math.log(4.0)
                loss = loss - mse_loss.detach() * self.fusion_gate_entropy_frac * h4_norm
            if self.skip_gate_entropy_frac > 0.0:
                h2_norm = skip_gate_entropy_mean / math.log(2.0)
                loss = loss - mse_loss.detach() * self.skip_gate_entropy_frac * h2_norm

        out = {
            "loss": loss,
            "logits": logits,
            "fusion_gate_entropy": fusion_gate_entropy_mean.detach(),
            "skip_gate_entropy": skip_gate_entropy_mean.detach(),
        }
        if mse_loss is not None:
            out["mse_loss"] = mse_loss.detach()
        return out

