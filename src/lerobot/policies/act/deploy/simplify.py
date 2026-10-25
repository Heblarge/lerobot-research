"""Offline graph simplification of a trained ACT policy for deployment.

Turns an `ACTPolicy` checkpoint plus its normalization statistics into a flat dict of folded weights and
constants that `ACTInference` (see `act_infer.py`) runs with only conv / relu / max_pool / add / GEMM / bmm /
softmax / layer_norm. Every transform is mathematically equivalent to the original inference path
(`preprocessor -> ACTPolicy.predict_action_chunk -> postprocessor`). The numbering (A1, B6, C10, ...) refers to
`GRAPH_SIMPLIFICATION.md`.
"""

import json
import math
from pathlib import Path
from typing import Any

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from safetensors.torch import load_file
from torch import Tensor, nn
from torchvision.models.resnet import BasicBlock
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.configs.types import NormalizationMode
from lerobot.utils.constants import ACTION, OBS_STATE

from ..configuration_act import ACTConfig
from ..modeling_act import ACT, ACTPolicy

QKV_MODES = ("add", "table")


def load_normalization_stats(pretrained_path: str | Path) -> tuple[dict[str, Tensor], float]:
    """Read the MEAN_STD statistics saved next to the checkpoint by the normalizer processor steps.

    Returns the merged stats dict (keys like `observation.state.mean`) and the normalizer epsilon.
    """
    pretrained_path = Path(pretrained_path)
    stats: dict[str, Tensor] = {}
    eps = None
    for pipeline_file, step_name in (
        ("policy_preprocessor.json", "normalizer_processor"),
        ("policy_postprocessor.json", "unnormalizer_processor"),
    ):
        pipeline = json.loads((pretrained_path / pipeline_file).read_text())
        step = next(s for s in pipeline["steps"] if s["registry_name"] == step_name)
        for key, value in load_file(pretrained_path / step["state_file"]).items():
            stats.setdefault(key, value)
        if step_name == "normalizer_processor":
            eps = step["config"]["eps"]
    if eps is None:
        raise ValueError(f"No normalizer_processor step found in {pretrained_path}.")
    return stats, eps


def check_supported(config: ACTConfig) -> None:
    """The folds below rely on these settings (see section 0 of GRAPH_SIMPLIFICATION.md)."""
    if config.n_decoder_layers != 1:
        raise ValueError("B6/B7 (constant decoder self-attention) require n_decoder_layers == 1.")
    if config.pre_norm:
        raise ValueError("Only post-norm (pre_norm=False) ACT models are supported.")
    if config.feedforward_activation != "relu":
        raise ValueError("Only relu feed-forward activation is supported.")
    if config.replace_final_stride_with_dilation:
        raise ValueError("Dilated backbones are not supported.")
    if config.robot_state_feature is None or not config.image_features:
        raise ValueError("Simplification expects both a robot state feature and at least one camera.")
    if config.env_state_feature is not None:
        raise ValueError("Environment state features are not supported.")
    for feature_type in ("VISUAL", "STATE", "ACTION"):
        if config.normalization_mapping[feature_type] != NormalizationMode.MEAN_STD:
            raise ValueError(f"Only MEAN_STD normalization is supported, got {feature_type} mode.")


def fold_bn(conv: nn.Conv2d, bn: FrozenBatchNorm2d) -> tuple[Tensor, Tensor]:
    """C10: fold a FrozenBatchNorm2d into the preceding (bias-free) conv."""
    scale = bn.weight * (bn.running_var + bn.eps).rsqrt()
    weight = conv.weight * scale[:, None, None, None]
    bias = bn.bias - bn.running_mean * scale
    if conv.bias is not None:
        bias = bias + conv.bias * scale
    return weight, bias


def _conv_spec(conv: nn.Conv2d) -> list[int]:
    if conv.stride[0] != conv.stride[1] or conv.padding[0] != conv.padding[1] or conv.dilation != (1, 1):
        raise ValueError(f"Unsupported conv geometry: {conv}")
    return [conv.stride[0], conv.padding[0]]


def simplify_backbone(model: ACT) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """C10 for the whole ResNet: returns folded conv weights and the conv geometry needed to run them."""
    bb = model.backbone
    tensors: dict[str, Tensor] = {}
    convs: dict[str, list[int]] = {}

    def add(name: str, conv: nn.Conv2d, bn: FrozenBatchNorm2d) -> None:
        tensors[f"backbone.{name}.weight"], tensors[f"backbone.{name}.bias"] = fold_bn(conv, bn)
        convs[name] = _conv_spec(conv)

    add("conv1", bb.conv1, bb.bn1)
    blocks = []
    for li in range(1, 5):
        layer = getattr(bb, f"layer{li}")
        blocks.append(len(layer))
        for bi, block in enumerate(layer):
            if not isinstance(block, BasicBlock):
                raise ValueError("Only BasicBlock ResNets (resnet18/34) are supported.")
            prefix = f"layer{li}.{bi}"
            add(f"{prefix}.conv1", block.conv1, block.bn1)
            add(f"{prefix}.conv2", block.conv2, block.bn2)
            if block.downsample is not None:
                add(f"{prefix}.downsample", block.downsample[0], block.downsample[1])

    mp = bb.maxpool
    meta = {"convs": convs, "blocks": blocks, "maxpool": [mp.kernel_size, mp.stride, mp.padding]}
    return tensors, meta


def split_in_proj(mha: nn.MultiheadAttention) -> tuple[Tensor, ...]:
    """A3: (Wq, Wk, Wv, bq, bk, bv, Wo, bo) from an nn.MultiheadAttention."""
    wq, wk, wv = mha.in_proj_weight.chunk(3, dim=0)
    bq, bk, bv = mha.in_proj_bias.chunk(3, dim=0)
    return wq, wk, wv, bq, bk, bv, mha.out_proj.weight, mha.out_proj.bias


def encoder_pos_embed(model: ACT, n_cams: int, feat_hw: tuple[int, int]) -> Tensor:
    """B5: (n_tokens, D) positional embedding for [latent, state, cam0 tokens, cam1 tokens, ...].

    The 2D sinusoidal embedding only depends on the feature map size, so it is identical for every camera.
    """
    d = model.config.dim_model
    cam_pos = model.encoder_cam_feat_pos_embed(torch.zeros(1, d, *feat_hw))
    cam_pos = einops.rearrange(cam_pos, "1 c h w -> (h w) c")
    return torch.cat([model.encoder_1d_feature_pos_embed.weight] + [cam_pos] * n_cams, dim=0)


def simplify_encoder(model: ACT, enc_pos: Tensor, qkv_mode: str) -> dict[str, Tensor]:
    """A3 + B8 + B9 for every encoder layer."""
    d = model.config.dim_model
    scale = 1.0 / math.sqrt(d // model.config.n_heads)
    tensors: dict[str, Tensor] = {}
    for i, layer in enumerate(model.encoder.layers):
        p = f"enc.{i}"
        wq, wk, wv, bq, bk, bv, wo, bo = split_in_proj(layer.self_attn)
        if qkv_mode == "add":
            # Q/K are computed from (x + enc_pos), V from x.
            tensors[f"{p}.w_qk"] = torch.cat([wq * scale, wk], dim=0)
            tensors[f"{p}.b_qk"] = torch.cat([bq * scale, bk], dim=0)
            tensors[f"{p}.w_v"] = wv
            tensors[f"{p}.b_v"] = bv
        else:
            # One GEMM with a per-token bias table that absorbs enc_pos.
            tensors[f"{p}.w_qkv"] = torch.cat([wq * scale, wk, wv], dim=0)
            tensors[f"{p}.b_qkv"] = torch.cat(
                [F.linear(enc_pos, wq, bq) * scale, F.linear(enc_pos, wk, bk), bv.expand(len(enc_pos), d)],
                dim=1,
            )
        tensors[f"{p}.w_o"], tensors[f"{p}.b_o"] = wo, bo
        tensors[f"{p}.w_1"], tensors[f"{p}.b_1"] = layer.linear1.weight, layer.linear1.bias
        tensors[f"{p}.w_2"], tensors[f"{p}.b_2"] = layer.linear2.weight, layer.linear2.bias
        tensors[f"{p}.norm1.weight"], tensors[f"{p}.norm1.bias"] = layer.norm1.weight, layer.norm1.bias
        tensors[f"{p}.norm2.weight"], tensors[f"{p}.norm2.bias"] = layer.norm2.weight, layer.norm2.bias
    return tensors


def simplify_decoder(model: ACT, enc_pos: Tensor) -> dict[str, Tensor]:
    """A3 + B6 + B7 + B9 for the single decoder layer, plus the final decoder LayerNorm."""
    cfg = model.config
    d = cfg.dim_model
    scale = 1.0 / math.sqrt(d // cfg.n_heads)
    layer = model.decoder.layers[0]

    # B6: decoder_in is all zeros, so V == bv for every row and softmax rows sum to 1: the self-attention
    # output is out_proj(bv) regardless of the attention weights.
    _, _, _, _, _, bv_self, _, _ = split_in_proj(layer.self_attn)
    dec_x = layer.norm1(layer.self_attn.out_proj(bv_self)).expand(cfg.chunk_size, d)

    # B7: the cross-attention query only depends on constants.
    wq, wk, wv, bq, bk, bv, wo, bo = split_in_proj(layer.multihead_attn)
    dec_q = F.linear(dec_x + model.decoder_pos_embed.weight, wq, bq) * scale

    return {
        "dec.x": dec_x,
        "dec.q": dec_q,
        "dec.w_k": wk,
        "dec.b_k": F.linear(enc_pos, wk, bk),  # B8 table form: (n_tokens, D)
        "dec.w_v": wv,
        "dec.b_v": bv,
        "dec.w_o": wo,
        "dec.b_o": bo,
        "dec.w_1": layer.linear1.weight,
        "dec.b_1": layer.linear1.bias,
        "dec.w_2": layer.linear2.weight,
        "dec.b_2": layer.linear2.bias,
        "dec.norm2.weight": layer.norm2.weight,
        "dec.norm2.bias": layer.norm2.bias,
        "dec.norm3.weight": layer.norm3.weight,
        "dec.norm3.bias": layer.norm3.bias,
        "dec.norm.weight": model.decoder.norm.weight,
        "dec.norm.bias": model.decoder.norm.bias,
    }


def simplify_io(model: ACT, stats: dict[str, Tensor], eps: float) -> dict[str, Tensor]:
    """B4 + C11 + C12, plus the per-camera image normalization kept as a runtime op (C13 is not folded)."""
    cfg = model.config

    # B4: the latent is zero at inference, so its projection is just the bias.
    tensors = {"latent_token": model.encoder_latent_input_proj.bias}

    # C11: s_n = (s - mu) / (std + eps) folded into the state projection.
    mu, sd = stats[f"{OBS_STATE}.mean"], stats[f"{OBS_STATE}.std"] + eps
    w_state = model.encoder_robot_state_input_proj.weight / sd[None, :]
    tensors["w_state"] = w_state
    tensors["b_state"] = model.encoder_robot_state_input_proj.bias - w_state @ mu

    # C12: a = y * std + mu (the inverse uses no eps) folded into the action head.
    mu, sd = stats[f"{ACTION}.mean"], stats[f"{ACTION}.std"]
    tensors["w_act"] = sd[:, None] * model.action_head.weight
    tensors["b_act"] = sd * model.action_head.bias + mu

    # C13 stays a runtime op: cameras have different stats, folding them into conv1 would need one conv1 per
    # camera and break batching the cameras through the backbone (D15), besides the zero-padding mismatch.
    keys = list(cfg.image_features)
    tensors["img_mean"] = torch.stack([stats[f"{k}.mean"] for k in keys])  # (n_cams, 3, 1, 1)
    tensors["img_inv_std"] = torch.stack([1.0 / (stats[f"{k}.std"] + eps) for k in keys])

    # D15: the 1x1 input projection is a GEMM on the flattened feature map.
    proj = model.encoder_img_feat_input_proj
    tensors["w_img"] = proj.weight[:, :, 0, 0]
    tensors["b_img"] = proj.bias
    return tensors


@torch.no_grad()
def simplify_act(
    policy: ACTPolicy, stats: dict[str, Tensor], eps: float, qkv_mode: str = "add"
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Run every transform and return (tensors, meta) for `ACTInference`.

    A1 (drop the VAE encoder) and A2 (drop dropout) are implicit: nothing from those modules is exported.
    """
    if qkv_mode not in QKV_MODES:
        raise ValueError(f"qkv_mode must be one of {QKV_MODES}, got {qkv_mode!r}.")
    cfg = policy.config
    check_supported(cfg)
    model = policy.model.eval().cpu()

    n_cams = len(cfg.image_features)
    img_shape = next(iter(cfg.image_features.values())).shape
    feat = model.backbone(torch.zeros(1, *img_shape))["feature_map"]
    feat_hw = tuple(feat.shape[-2:])
    enc_pos = encoder_pos_embed(model, n_cams, feat_hw)

    tensors, backbone_meta = simplify_backbone(model)
    tensors |= simplify_io(model, stats, eps)
    tensors |= simplify_encoder(model, enc_pos, qkv_mode)
    tensors |= simplify_decoder(model, enc_pos)
    if qkv_mode == "add":
        tensors["enc_pos"] = enc_pos

    meta = {
        "dim_model": cfg.dim_model,
        "n_heads": cfg.n_heads,
        "n_encoder_layers": cfg.n_encoder_layers,
        "chunk_size": cfg.chunk_size,
        "state_dim": cfg.robot_state_feature.shape[0],
        "action_dim": cfg.action_feature.shape[0],
        "camera_keys": list(cfg.image_features),
        "image_shape": list(img_shape),
        "feature_hw": list(feat_hw),
        "n_tokens": len(enc_pos),
        "qkv_mode": qkv_mode,
        "ln_eps": model.encoder.layers[0].norm1.eps,
        **backbone_meta,
    }
    tensors = {k: v.detach().float().contiguous().clone() for k, v in tensors.items()}
    return tensors, meta
