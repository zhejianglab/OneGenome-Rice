# 标准库（内置模块）
from typing import Optional, Union, Dict, List

# 第三方库（pip 安装的包）
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from src.util import dist_print
from transformers import AutoConfig, AutoModel
from safetensors.torch import load_file
import os
import torch.distributed as dist
import json

def _sync_loss_across_gpus(loss_dict: dict) -> dict:
    """
    Given a dict of {name: scalar-tensor or float}, all-reduce across processes and
    return a dict of floats averaged across world_size.
    If torch.distributed is not initialized, returns loss_dict with numeric floats.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        # ensure floats
        return {k: float(v.detach().cpu().item()) if torch.is_tensor(v) else float(v) for k, v in loss_dict.items()}

    world_size = torch.distributed.get_world_size()
    device = next(iter(loss_dict.values())).device if len(loss_dict) > 0 and torch.is_tensor(next(iter(loss_dict.values()))) else torch.device("cpu")

    synced = {}
    for k, v in loss_dict.items():
        if not torch.is_tensor(v):
            tensor_v = torch.tensor(float(v), device=device, dtype=torch.float32)
        else:
            tensor_v = v.detach().to(device).float()
        # make a clone to avoid in-place ops on original graph tensors
        tmp = tensor_v.clone()
        torch.distributed.all_reduce(tmp, op=torch.distributed.ReduceOp.SUM)
        tmp = tmp / float(world_size)
        synced[k] = float(tmp.cpu().item())
    return synced

    
def load_finetuned_model(
    model_class,
    model_path: str,
    ckpt_path: str,
    use_flash_attn: bool = False,
    trust_remote_code: bool = True,
    revision: str = "main",
    device: str = "cuda:0",
    torch_dtype: torch.dtype = None,
    model_init_args: Optional[List] = None,         # <-- 新增：位置参数列表
    model_init_kwargs: Optional[Dict] = None,       # <-- 新增：关键字参数字典
) -> torch.nn.Module:
    """
    加载微调模型：直接在目标设备上构建并加载，避免 CPU/GPU 混合问题。

    新增参数:
      - model_init_args: 传递给 model_class 的位置参数列表（可为 None）
      - model_init_kwargs: 传递给 model_class 的关键字参数字典（可为 None）
    """
    # 1. 从 checkpoint 推断 vocab_size
    ckpt_path = str(ckpt_path)
    if ckpt_path.endswith(".safetensors"):
        # 先加载 state_dict 获取 embed_tokens.shape
        state_dict = load_file(ckpt_path, device=device)  # 🔥 直接加载到 GPU
    else:
        state_dict = torch.load(ckpt_path, map_location=device)  # 直接到 GPU

    # # 推断 vocab_size
    # embed_key = "base.embed_tokens.weight"
    # if embed_key not in state_dict:
    #     raise KeyError(f"找不到 {embed_key}，请检查 checkpoint 结构")

    # loaded_vocab_size, hidden_size = state_dict[embed_key].shape
    # dist_print(f"📌 从 checkpoint 推断 vocab_size = {loaded_vocab_size}")

    # 2. 加载并修改 config
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        revision=revision
    )

    # if config.vocab_size != loaded_vocab_size:
    #     dist_print(f"🔧 修改 vocab_size: {config.vocab_size} → {loaded_vocab_size}")
    #     config.vocab_size = loaded_vocab_size
    
    # 设置 Attention 实现
    if use_flash_attn:
        if torch_dtype not in (torch.float16, torch.bfloat16):
            dist_print("⚠️ 使用 Flash Attention 2 需要 torch.float16 或 torch.bfloat16，已自动设置为 torch.bfloat16")
            torch_dtype = torch.bfloat16
        config._attn_implementation = "flash_attention_2"

    # 3. ✅ 直接在目标设备上初始化模型
    base_model = AutoModel.from_config(config, trust_remote_code=trust_remote_code)
    # 兼容：允许传入其他构造参数给 model_class（例如 GenOmics 需要 index_stat）
    init_args = model_init_args or []
    init_kwargs = model_init_kwargs or {}
    model = model_class(base_model, *init_args, **init_kwargs)

    # 4. 设置 dtype 并移动到设备
    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)
    model = model.to(device)

    # 5. ✅ 直接注入已在 GPU 上的 state_dict
    load_info = model.load_state_dict(state_dict, strict=False)
    if load_info.missing_keys:
        dist_print(f"⚠️  缺失 keys: {load_info.missing_keys[:5]}...")
    if load_info.unexpected_keys:
        dist_print(f"⚠️  多余 keys: {load_info.unexpected_keys[:5]}...")

    return model


def targets_scaling_torch(
    targets: torch.Tensor, 
    track_means: Union[float, torch.Tensor], 
    apply_squashing: Union[bool, list, torch.Tensor] = True
) -> torch.Tensor:
    """
    Robust targets scaling that accepts:
      - targets: [B, L] (single channel), [B, L, C], or [B, C, L]
      - track_means: scalar, [C], or [B, C]
    Returns scaled targets with same layout as input.

    This implementation avoids in-place ops to keep autograd safe.
    """
    # normalize track_means to tensor on same device/dtype
    if isinstance(track_means, (int, float)):
        tm = torch.tensor(track_means, dtype=targets.dtype, device=targets.device)
    elif isinstance(track_means, torch.Tensor):
        tm = track_means.to(device=targets.device, dtype=targets.dtype)
    else:
        tm = torch.tensor(track_means, dtype=targets.dtype, device=targets.device)

    # Flags to restore layout
    transposed = False
    squeezed_single_channel = False

    # Normalize targets layout to [B, L, C]
    t = targets
    if t.ndim == 2:
        t = t.unsqueeze(-1)
        squeezed_single_channel = True

    # Detect [B, C, L] -> transpose to [B, L, C] when appropriate （启发式的转换条件，但通常用不到）
    if t.ndim == 3 and t.shape[1] <= 16 and t.shape[2] > 1 and tm.numel() == t.shape[1]:
        t = t.transpose(1, 2)
        transposed = True

    B, L, C = t.shape

    # build tm_view broadcastable to [B, L, C]
    tm_view = None
    try:
        if tm.ndim == 0:
            tm_view = tm.view(1, 1, 1)
        elif tm.ndim == 1:
            if tm.numel() == C:
                tm_view = tm.view(1, 1, C)
            elif tm.numel() == B:
                tm_view = tm.view(B, 1, 1)
            elif tm.numel() == L:
                tm_view = tm.view(1, L, 1)
            else:
                scalar = tm.mean()
                dist_print(f"[targets_scaling_torch] WARNING ambiguous track_means shape {tuple(tm.shape)} -> using scalar mean {float(scalar):.6g}")
                tm_view = scalar.view(1, 1, 1)
        elif tm.ndim == 2:
            if tm.shape[0] == B and tm.shape[1] == C:
                tm_view = tm.view(B, 1, C)
            elif tm.shape[1] == C:
                tm_view = tm.view(1, 1, C)
            elif tm.shape[0] == B and tm.shape[1] == 1:
                tm_view = tm.view(B, 1, 1)
            else:
                scalar = tm.mean()
                dist_print(f"[targets_scaling_torch] WARNING unsupported track_means shape {tuple(tm.shape)} -> using scalar mean {float(scalar):.6g}")
                tm_view = scalar.view(1, 1, 1)
        else:
            view = tm
            while view.ndim < 3:
                view = view.unsqueeze(-1)
            tm_view = view
    except Exception as e:
        scalar = tm.mean()
        dist_print(f"[targets_scaling_torch] ERROR building tm_view ({e}) -> using scalar mean {float(scalar):.6g}")
        tm_view = scalar.view(1, 1, 1)

    # do division with broadcasting into a new tensor (no in-place)
    try:
        scaled = t / tm_view
    except Exception as e:
        scalar = tm.mean()
        dist_print(f"[targets_scaling_torch] WARNING broadcasting failed: {e} -> using scalar mean {float(scalar):.6g}")
        scaled = t / scalar

    # vectorized squashing per-channel without in-place writes
    def _squash(x):
        x_pow = x.pow(0.75)
        transformed = torch.where(x_pow > 10.0, 2 * torch.sqrt(x_pow * 10.0) - 10.0, x_pow)
        return transformed

    if isinstance(apply_squashing, bool):
        if apply_squashing:
            scaled = _squash(scaled)
    else:
        # build boolean mask per channel
        if isinstance(apply_squashing, torch.Tensor):
            mask_list = [bool(x) for x in apply_squashing.to('cpu').tolist()]
        elif isinstance(apply_squashing, (list, tuple)):
            mask_list = [bool(x) for x in apply_squashing]
        else:
            mask_list = [True] * C

        mask = torch.tensor(mask_list, dtype=torch.bool, device=scaled.device)
        # shape to broadcast: [1,1,C]
        mask_view = mask.view(*([1] * (scaled.ndim - 1)), C)
        transformed = _squash(scaled)
        # select per-element from transformed or original scaled
        scaled = torch.where(mask_view, transformed, scaled)

    # Restore original layout (avoid in-place)
    out = scaled
    if transposed:
        out = out.transpose(1, 2)
    if squeezed_single_channel:
        out = out.squeeze(-1)
    return out


def predictions_scaling_torch(
    predictions: torch.Tensor, 
    track_means: Union[float, torch.Tensor], 
    apply_squashing: Union[bool, list, torch.Tensor] = True
) -> torch.Tensor:
    """
    Robust inverse-scaling for model predictions.
    Vectorized, no in-place modification so autograd remains valid.
    """
    # normalize track_means
    if isinstance(track_means, (int, float)):
        tm = torch.tensor(track_means, dtype=predictions.dtype, device=predictions.device)
    elif isinstance(track_means, torch.Tensor):
        tm = track_means.to(device=predictions.device, dtype=predictions.dtype)
    else:
        tm = torch.tensor(track_means, dtype=predictions.dtype, device=predictions.device)

    # clone preds to avoid in-place modifications on tensors needed for grad
    preds = predictions.clone()

    # handle single-channel case
    single_channel = (preds.ndim == 2)
    if single_channel:
        preds = preds.unsqueeze(-1)  # [B, L, 1]

    # build tm_view broadcastable to preds
    def build_tm_view(tm_tensor, preds_tensor):
        try:
            if tm_tensor.ndim == 0:
                return tm_tensor.view(1, 1, 1)
            if tm_tensor.ndim == 1:
                return tm_tensor.view(*([1] * (preds_tensor.ndim - 1)), -1)
            if tm_tensor.ndim == 2 and preds_tensor.ndim == 3:
                return tm_tensor.view(tm_tensor.size(0), 1, tm_tensor.size(1))
            view = tm_tensor
            while view.ndim < preds_tensor.ndim:
                view = view.unsqueeze(-1)
            return view
        except Exception:
            return None

    tm_view = build_tm_view(tm, preds)
    if tm_view is None:
        tm_view = tm.mean().view(1, 1, 1)

    C = preds.shape[-1]

    # build per-channel squashing mask
    if isinstance(apply_squashing, bool):
        squashing_mask = [apply_squashing] * C
    elif isinstance(apply_squashing, (list, tuple)):
        squashing_mask = list(apply_squashing) + [False] * max(0, C - len(apply_squashing))
        squashing_mask = squashing_mask[:C]
    elif isinstance(apply_squashing, torch.Tensor):
        squashing_mask = [bool(x) for x in apply_squashing.to('cpu').tolist()]
        squashing_mask = squashing_mask + [False] * max(0, C - len(squashing_mask))
        squashing_mask = squashing_mask[:C]
    else:
        squashing_mask = [True] * C

    mask = torch.tensor(squashing_mask, dtype=torch.bool, device=preds.device).view(*([1] * (preds.ndim - 1)), C)

    # Invert targets_scaling_torch in the exact reverse order:
    # raw/mean -> power(0.75) -> optional high-value sqrt compression.
    # Therefore prediction inverse is:
    # optional high-value quadratic expansion -> power(1/0.75) -> multiply by mean.
    quad_mask = (preds > 10.0) & mask
    preds_quad = (preds + 10.0).pow(2) / (4.0 * 10.0)
    preds = torch.where(quad_mask, preds_quad, preds)

    inv_pow = 1.0 / 0.75
    preds_pow = preds.pow(inv_pow)
    preds = torch.where(mask, preds_pow, preds)

    # multiply by tm_view (broadcast-safe)
    try:
        preds = preds * tm_view
    except Exception:
        preds = preds * tm.mean()

    # restore single-channel shape and sanitize
    if single_channel:
        preds = preds.squeeze(-1)
    preds = torch.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    return preds

# # 缩放函数    
# def targets_scaling(targets, track_means, apply_squashing=True):
#     targets = targets / track_means
#     if apply_squashing:
#         targets = targets ** 0.75
#         targets = np.where(targets > 10.0, 2 * np.sqrt(targets * 10.0) - 10.0, targets)
#     return targets

# def predictions_scaling(predictions, track_means, apply_squashing=True):
#     predictions = np.where(predictions > 10.0, (predictions + 10.0) ** 2 / (4 * 10.0), predictions)
#     if apply_squashing:
#         predictions = predictions ** (1.0 / 0.75)
#     predictions = predictions * track_means
#     return np.nan_to_num(predictions, nan=0.0)


# # 缩放函数（Torch 原生版本）
# def targets_scaling_torch(targets: torch.Tensor, track_means: Union[float, torch.Tensor], apply_squashing: bool = True) -> torch.Tensor:
#     if isinstance(track_means, (int, float)):
#         track_means = torch.tensor(track_means, dtype=targets.dtype, device=targets.device)
#     if track_means.ndim == 0:
#         track_means = track_means.view(1)
#     while track_means.ndim < targets.ndim:
#         track_means = track_means.unsqueeze(-1)
#     targets = targets / track_means
#     if apply_squashing:
#         targets = targets ** 0.75
#         mask = targets > 10.0
#         targets = torch.where(mask, 2 * torch.sqrt(targets * 10.0) - 10.0, targets)
#     return targets

# def predictions_scaling_torch(predictions: torch.Tensor, track_means: Union[float, torch.Tensor], apply_squashing: bool = True) -> torch.Tensor:
#     if isinstance(track_means, (int, float)):
#         track_means = torch.tensor(track_means, dtype=predictions.dtype, device=predictions.device)
#     if track_means.ndim == 0:
#         track_means = track_means.view(1)
#     while track_means.ndim < predictions.ndim:
#         track_means = track_means.unsqueeze(-1)
#     mask = predictions > 10.0
#     predictions = torch.where(mask, (predictions + 10.0) ** 2 / (4 * 10.0), predictions)
#     if apply_squashing:
#         predictions = predictions ** (1.0 / 0.75)
#     predictions = predictions * track_means
#     predictions = torch.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
#     return predictions

# Poisson Loss
def poisson_loss(preds, targets, eps=1e-7):
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    poisson_nll = preds - targets * torch.log(preds + eps)
    return torch.mean(poisson_nll)

# Tweedie Loss（仅一个可学习参数 p）
def tweedie_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    p: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Tweedie 回归损失（负对数似然近似），适用于 1 < p < 2 （复合泊松-伽马）
    用于建模零膨胀连续正数数据（如 RNA-seq 覆盖度）
    """
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    preds = preds + eps
    targets = targets + eps
    p_clipped = p.clamp(min=1.01, max=1.99)  # 安全边界

    term1 = -targets * torch.pow(preds, 1 - p_clipped) / (1 - p_clipped)
    term2 = torch.pow(preds, 2 - p_clipped) / (2 - p_clipped)

    loss = term1 + term2
    return torch.mean(loss)


# def poisson_multinomial_loss(preds, targets, multinomial_resolution=16384, positional_loss_weight=5, eps=1e-7):

#     preds = preds.reshape(-1, multinomial_resolution, 1)
#     targets = targets.reshape(-1, multinomial_resolution, 1)
#     sum_pred = torch.sum(preds, dim=1, keepdim=True)
#     sum_target = torch.sum(targets, dim=1, keepdim=True)
#     poisson = torch.sum(sum_pred - sum_target * torch.log(sum_pred + eps))
#     multinom_prob = preds / (sum_pred + eps)
#     positional = torch.sum(-targets * torch.log(multinom_prob + eps))
#     return poisson  + positional_loss_weight * positional


def poisson_multinomial_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    total_weight: float = 1.0,
    epsilon: float = 1e-7,
    multinomial_resolution: Optional[int] = None,
) -> torch.Tensor:
    """
    Poisson-Multinomial loss (without position weighting).
    
    Args:
        y_pred: Predicted counts, shape [B, L, C]
        y_true: True counts, shape [B, L, C]
        total_weight: Weight for Poisson total term (default: 1.0)
        epsilon: Small constant for numerical stability (default: 1e-7)

    Returns:
        Scalar tensor: mean loss over batch and channels.
    """
    # Add epsilon for numerical stability
    y_true_eps = y_true + epsilon
    y_pred_eps = y_pred + epsilon

    B, L, C = y_pred.shape

    # Determine resolution: if multinomial_resolution is None or >= L, use full-length (single group)
    if multinomial_resolution is None or multinomial_resolution <= 0 or multinomial_resolution >= L:
        res = L
    else:
        res = int(multinomial_resolution)

    groups = (L + res - 1) // res
    pad_len = groups * res - L

    # pad with small epsilon to avoid zero sums / div-by-zero
    if pad_len > 0:
        pad_pred = torch.full((B, pad_len, C), fill_value=epsilon, device=y_pred.device, dtype=y_pred.dtype)
        pad_true = torch.full((B, pad_len, C), fill_value=epsilon, device=y_true.device, dtype=y_true.dtype)
        y_pred_p = torch.cat([y_pred_eps, pad_pred], dim=1)
        y_true_p = torch.cat([y_true_eps, pad_true], dim=1)
    else:
        y_pred_p = y_pred_eps
        y_true_p = y_true_eps

    # reshape into groups: [B, groups, res, C]
    # use reshape instead of view to be robust to non-contiguous tensors
    y_pred_g = y_pred_p.reshape(B, groups, res, C)
    y_true_g = y_true_p.reshape(B, groups, res, C)

    # totals per group: [B, groups, C]
    s_pred = y_pred_g.sum(dim=2)
    s_true = y_true_g.sum(dim=2)

    # Poisson term per group
    poisson_term = s_pred - s_true * torch.log(s_pred + epsilon)

    # Multinomial probabilities and NLL per group
    p_pred = y_pred_g / (s_pred.unsqueeze(2) + epsilon)
    multinomial_term = -(y_true_g * torch.log(p_pred + epsilon)).sum(dim=2)  # [B, groups, C]

    # combine: per (B, groups, C)
    loss_per_bgc = multinomial_term + total_weight * poisson_term

    # average across groups (if groups==1 this is a no-op), then across batch & channels
    loss_per_bc = loss_per_bgc.mean(dim=1)  # [B, C]
    return loss_per_bc.mean()

# ============================================================= # 
# ============================================================= #
class Conv1DBlock(nn.Module):
    """
    Enhanced 1D convolutional block with support for different downsampling methods.
    Use strided conv, max pooling or average pooling when downsample > 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        padding: Optional[int] = None,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        downsample: int = 1,
        downsample_method: str = 'conv',  # 默认：'conv', 可选：'maxpool', 'avgpool'
        upsample: int = 1      # Still uses interpolate for safety
    ):
        super().__init__()

        if downsample < 1 or upsample < 1:
            raise ValueError("downsample and upsample must be >= 1")
        if downsample > 1 and upsample > 1:
            raise ValueError("Cannot apply both downsampling and upsampling in the same block.")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd to allow symmetric padding.")
        if downsample_method not in ['conv', 'maxpool', 'avgpool']:
            raise ValueError("downsample_method must be 'conv', 'maxpool', or 'avgpool'")

        self.downsample_factor = downsample
        self.downsample_method = downsample_method
        self.upsample_factor = upsample

        # Calculate padding to preserve length after convolution
        if padding is None:
            padding = (kernel_size - 1) * dilation // 2
        self.padding = padding

        # Build main conv layer (only for 'conv' method or when no downsampling)
        if downsample_method == 'conv' or downsample == 1:
            conv_layer = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=downsample,  # ✅ Learnable downsampling via stride
                padding=self.padding,
                dilation=dilation
            )
            layers = [conv_layer]
        else:
            # For pooling methods, use stride=1 in conv and separate pooling layer
            conv_layer = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=self.padding,
                dilation=dilation
            )
            layers = [conv_layer]
            
            # Add pooling layer
            if downsample_method == 'maxpool':
                self.downsample_pool = nn.MaxPool1d(kernel_size=downsample, stride=downsample)
            elif downsample_method == 'avgpool':
                self.downsample_pool = nn.AvgPool1d(kernel_size=downsample, stride=downsample)
            else:
                self.downsample_pool = None

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_channels))

        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*layers)

        # Upsample: handled in forward via interpolate (safe)
        self.upsample_scale = upsample if upsample > 1 else None

    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)

        # Apply additional downsampling if needed
        if hasattr(self, 'downsample_pool') and self.downsample_pool is not None:
            out = self.downsample_pool(out)

        if self.upsample_scale is not None:
            out = F.interpolate(out, scale_factor=self.upsample_scale, mode='nearest')

        return out
    


class func_genome_UNet(nn.Module):
    """
    功能性基因组信号的 U-Net 模型，用于特征提取。
    包含动态构建的编码器（Encoder）、瓶颈层（Bottleneck）和解码器（Decoder）。
    """

    def __init__(self, proj_dim, num_downsamples, bottleneck_dim):
        """
        初始化 U-Net 模型。

        参数:
            proj_dim (int): 输入特征的维度。
            num_downsamples (int): 下采样次数，建议 1 到 6 次，比如 2 或 4。
            bottleneck_dim (int): 瓶颈层的维度。
        """
        super(func_genome_UNet, self).__init__()
        assert 1 <= num_downsamples <= 6, "num_downsamples 必须在 1 到 6 之间"
        assert bottleneck_dim > proj_dim, "bottleneck_dim 必须大于 proj_dim"
        self.proj_dim = proj_dim
        self.num_downsamples = num_downsamples
        self.bottleneck_dim = bottleneck_dim

        # 自动计算每次下采样需要增加的维度
        self.dim_step = (bottleneck_dim - proj_dim) // num_downsamples

        # 动态构建编码器（Encoder）
        self.encoders = nn.ModuleList()
        in_channels = proj_dim
        for i in range(num_downsamples):
            out_channels = proj_dim + self.dim_step * (i + 1)
            self.encoders.append(Conv1DBlock(in_channels, out_channels, kernel_size=5, downsample=2))
            in_channels = out_channels

        # 瓶颈层（Bottleneck）
        self.bottleneck = nn.Sequential(
            Conv1DBlock(in_channels, bottleneck_dim, kernel_size=5, dilation=2),
            Conv1DBlock(bottleneck_dim, bottleneck_dim, kernel_size=5, dilation=4)
        )

        # 动态构建解码器（Decoder）
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(num_downsamples):
            out_channels = proj_dim + self.dim_step * (num_downsamples - i - 1)
            self.upsamplers.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            self.decoders.append(Conv1DBlock(out_channels * 2, out_channels, kernel_size=5))
            in_channels = out_channels

    def forward(self, x):
        """
        前向传播。

        参数:
            x (Tensor): 输入张量，形状为 [batch_size, proj_dim, sequence_length]。

        返回:
            Tensor: 输出张量，形状为 [batch_size, proj_dim, sequence_length]。
        """
        # 编码器（Encoder）
        skip_connections = []
        for encoder in self.encoders:
            skip_connections.append(x)
            x = encoder(x)

        # 瓶颈层（Bottleneck）
        x = self.bottleneck(x)

        # 解码器（Decoder）与跳跃连接（Skip Connections）
        for i in range(self.num_downsamples):
            x = self.upsamplers[i](x)
            skip_connection = skip_connections[-(i + 1)]
            if x.size(-1) != skip_connection.size(-1):
                print(f"Upsampled size: {x.size(-1)}, Skip connection size: {skip_connection.size(-1)}")
                x = F.interpolate(x, size=skip_connection.size(-1), mode='nearest')
            x = self.decoders[i](torch.cat([x, skip_connection], dim=1))

        return x
    


class GenOmics(nn.Module):
    """
    GenoOmics: 基于 Genos 基因组大模型的多组学信号预测框架。
    核心功能:
        输入 DNA 序列，通过 Genos 基因组大模型提取深层特征，并结合 U-Net 网络捕获功能性基因组学信号。
        实现单碱基分辨率的转录组学（RNA-seq）和表观基因组学（ATAC-seq）信号轨迹的联合预测。
        用于解析基因调控机制，助力多组学数据的功能注释与机制研究。
    """

    def __init__(self, base_model, 
                 index_stat, 
                 loss_func: str = 'mse', 
                 proj_dim: int = 512, 
                 num_downsamples: int = 2, 
                 bottleneck_dim: int = 1024):
        """
        初始化模型。

        参数:
            base_model: 预训练的 DNA 模型。
            loss_func (str): 训练时使用的损失函数。支持 'mse'、'poisson'、'tweedie'、'poisson-multinomial'。
            proj_dim (int): 投影层的维度。
            num_downsamples (int): U-Net 编码器中的下采样层数。
        """
        super().__init__()
        self.loss_func = loss_func
        self.index_stat = index_stat
        self.assay_titles = list(self.index_stat['counts']['heads'])
        self.biosample_order = list(self.index_stat['counts']['biosample_order'])
        self.num_biosamples = len(self.biosample_order)
        self.biosample_to_idx = {name: i for i, name in enumerate(self.biosample_order)}
        self.apply_squashing = [
            not (name.startswith("ATAC")) 
            for name in self.index_stat['counts']['target_file_name']
            ]

        # 数据缩放
        self.num_tracks = len(self.index_stat['counts']['target_file_name'])
        self.track_means = torch.tensor(self.index_stat['counts']['nonzero_mean'], dtype=torch.float32)  # 每个轨迹的均值
        
        # 获取基础模型的隐藏层大小
        base_model_hidden_size = getattr(base_model.config, "hidden_size", None)
        if base_model_hidden_size is None:
            raise ValueError("无法从 `base_model` 中获取 `hidden_size`")
        
        # 特征提取：使用预训练的 DNA 模型作为嵌入器
        self.base = base_model
        
        # 嵌入投影层
        self.embedd_proj = Conv1DBlock(base_model_hidden_size, proj_dim, kernel_size=1)
        
        # 使用 genome_signal_UNet 作为编码器-解码器
        self.unet = func_genome_UNet(proj_dim=proj_dim, num_downsamples=num_downsamples, bottleneck_dim=bottleneck_dim)
        
        # 任务特定的输出头
        self.output_heads = nn.ModuleDict({
            name: nn.Conv1d(proj_dim, len(self.biosample_order), kernel_size=1) 
            for name in self.assay_titles
        })

        # 可学习的缩放因子
        self.scale = nn.Parameter(torch.zeros(self.num_tracks))

    def _compute_loss(self, logits, scaled_labels):
        """
        计算每个轨迹的损失并返回总 loss 以及按轨道的 loss dict。
        简洁实现：按 assay 分片计算并汇总。
        """
        losses_by_head = {}
        for i, name in enumerate(self.assay_titles):
            s = i * self.num_biosamples
            e = s + self.num_biosamples
            pred = logits[..., s:e]
            targ = scaled_labels[..., s:e]

            if self.loss_func == 'mse':
                l = F.mse_loss(pred, targ)
            elif self.loss_func == 'poisson':
                l = poisson_loss(pred, targ)
            elif self.loss_func == 'tweedie':
                l = tweedie_loss(pred, targ, p=torch.tensor(1.2, device=pred.device, dtype=pred.dtype))
            elif self.loss_func == 'poisson-multinomial':
                l = poisson_multinomial_loss(pred, targ)
            else:
                raise ValueError(f"不支持的损失函数: {self.loss_func}")

            # weight = 0.5 if name == "ATAC-seq_" else 1.0
            weight = 1
            losses_by_head[name] = weight * l

        total_loss = sum(losses_by_head.values()) if losses_by_head else torch.tensor(0.0, device=logits.device)
        return total_loss, losses_by_head

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, Optional[Tensor]]:
        """
        前向传播。

        参数:
            input_ids (Tensor): 输入的 DNA 序列张量，形状为 [batch_size, sequence_length]。
            labels (Optional[Tensor]): 标签张量，形状为 [batch_size, sequence_length, num_tracks]。
            track_means (Optional[Tensor]): 每个轨迹的均值，用于缩放。

        返回:
            Dict[str, Optional[Tensor]]: 包含损失和预测值的字典。
        """
        # 获取基础模型的隐藏状态
        outputs = self.base(input_ids=input_ids)  
        sequence_hidden = outputs.last_hidden_state  # [B, L, H]

        # 转置为 [B, H, L] 以便 CNN 处理
        x = sequence_hidden.transpose(1, 2)  # [B, H, L]

        # 嵌入投影
        x = self.embedd_proj(x)  # [B, proj_dim, L]

        # 使用 UNet 进行特征提取
        x = self.unet(x)  # [B, proj_dim, L]

        # 每个轨迹的输出头，应用 softplus 激活和可学习缩放
        head_outputs = []
        for i, name in enumerate(self.assay_titles):
            out = self.output_heads[name](x)  # [B, 1, L]
            out = F.softplus(out) * F.softplus(self.scale[i])  # [B, 1, L]
            head_outputs.append(out)
        logits = torch.cat(head_outputs, dim=1)  # [B, num_tracks, L]

        # 转置为 [B, L, num_tracks] 以匹配下游代码
        logits = logits.transpose(1, 2)  # [B, L, num_tracks]
        

        # 计算损失
        loss = None
        per_head_losses = None
        if labels is not None:
            scaled_labels = targets_scaling_torch(
                targets=labels,
                track_means=self.track_means,
                apply_squashing=self.apply_squashing
            )
            loss, per_head_losses = self._compute_loss(logits, scaled_labels)

        # 将预测值缩放回原始尺度
        original_logits = logits.clone()  # 保存缩放前的值用于调试
        logits = predictions_scaling_torch(
            predictions=logits,
            track_means=self.track_means,
            apply_squashing=self.apply_squashing
        )

        # ========== 缩放后打印 ==========
        if labels is not None and self.training:
            # 使用自己的计数器
            if not hasattr(self, '_step_counter'):
                self._step_counter = 0
            else:
                self._step_counter += 1
                
            step = self._step_counter
            
            # 每N步打印一次
            if step % 600 == 0:
                print(f"\n{'='*80}")
                print(f"【缩放后调试】训练步数: {step}")
                print(f"【缩放前后对比】:")

                # 打印缩放前后的统计对比
                print(f"\n1. 整体统计对比:")
                print(f"   缩放前预测 - 最大值: {original_logits.max().item():.6f}, "
                    f"范围: [{original_logits.min().item():.6f}, {original_logits.max().item():.6f}]")
                print(f"   缩放后预测 - 最大值: {logits.max().item():.6f}, "
                    f"范围: [{logits.min().item():.6f}, {logits.max().item():.6f}]")
                print(f"   原始标签   - 最大值: {labels.max().item():.6f}, "
                    f"范围: [{labels.min().item():.6f}, {labels.max().item():.6f}]")

        # ========== 结束缩放后打印 ==========

        # # 将预测值缩放回原始尺度
        # logits = predictions_scaling_torch(
        #     predictions=logits,
        #     track_means=self.track_means,
        #     apply_squashing=self.apply_squashing
        # )

        return {
            "loss": loss,
            "logits": logits,
            "per_head_losses": per_head_losses
        }
    
    def predict(
        self,
        input_ids: Tensor,
        assay_names: Optional[Union[str, List[str]]] = None,
        biosample_names: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Dict[str, Tensor]]:
        """
        Run the model forward (no labels) and return selected logits.

        Returns a nested dict: { assay_name: { biosample_name: tensor[B, L, 1] } }.
        If assay_names or biosample_names is None, all available names are used.
        """
        # normalize inputs to lists
        if assay_names is None:
            assay_list = list(self.assay_titles)
        elif isinstance(assay_names, str):
            assay_list = [assay_names]
        else:
            assay_list = list(assay_names)

        if biosample_names is None:
            biosample_list = list(self.biosample_order)
        elif isinstance(biosample_names, str):
            biosample_list = [biosample_names]
        else:
            biosample_list = list(biosample_names)

        # validate
        for a in assay_list:
            if a not in self.assay_titles:
                raise KeyError(f"Assay '{a}' not found in assay_titles")
        for b in biosample_list:
            if b not in self.biosample_to_idx:
                raise KeyError(f"Biosample '{b}' not found in biosample_order")

        # forward pass without computing loss
        with torch.no_grad():
            out = self.forward(input_ids=input_ids, labels=None)
        logits = out.get("logits")
        if logits is None:
            raise RuntimeError("Model forward did not return logits")
        # logits: [B, L, num_tracks]

        result: Dict[str, Dict[str, Tensor]] = {}
        for a in assay_list:
            a_idx = self.assay_titles.index(a)
            result[a] = {}
            for b in biosample_list:
                b_idx = self.biosample_to_idx[b]
                global_idx = a_idx * self.num_biosamples + b_idx
                # select channel and ensure shape [B, L, 1]
                sel = logits[..., global_idx].unsqueeze(-1)
                result[a][b] = sel

        return result