"""Simplified ACT inference graph (section E of GRAPH_SIMPLIFICATION.md).

Runs on the flat tensors produced by `simplify.simplify_act`. Batch size is fixed to 1 and every activation is
a 2D (tokens, D) tensor (D14). The only ops are conv2d(+bias), relu, max_pool2d, add, GEMM, bmm, softmax,
layer_norm and the per-camera image normalization, i.e. what has to be implemented on the target GPGPU.
"""

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

META_KEY = "act_simplified_meta"


def attention(q: Tensor, k: Tensor, v: Tensor, n_heads: int) -> Tensor:
    """A3: multi-head attention without masks or weight averaging. q (Lq, D) must already be scaled."""
    lq, d = q.shape
    lk = k.shape[0]
    hd = d // n_heads
    q = q.view(lq, n_heads, hd).transpose(0, 1)
    k = k.view(lk, n_heads, hd).transpose(0, 1)
    v = v.view(lk, n_heads, hd).transpose(0, 1)
    attn = torch.softmax(q @ k.transpose(1, 2), dim=-1)
    return (attn @ v).transpose(0, 1).reshape(lq, d)


class ACTInference(nn.Module):
    def __init__(self, tensors: dict[str, Tensor], meta: dict[str, Any]):
        super().__init__()
        self.meta = meta
        self._names = list(tensors)
        for name, value in tensors.items():
            self.register_buffer(name.replace(".", "__"), value)

    def t(self, name: str) -> Tensor:
        return getattr(self, name.replace(".", "__"))

    def _linear(self, x: Tensor, w: str, b: str) -> Tensor:
        return F.linear(x, self.t(w), self.t(b))

    def _layer_norm(self, x: Tensor, name: str) -> Tensor:
        d = self.meta["dim_model"]
        return F.layer_norm(x, (d,), self.t(f"{name}.weight"), self.t(f"{name}.bias"), self.meta["ln_eps"])

    def _conv(self, x: Tensor, name: str) -> Tensor:
        stride, padding = self.meta["convs"][name]
        return F.conv2d(
            x, self.t(f"backbone.{name}.weight"), self.t(f"backbone.{name}.bias"), stride, padding
        )

    def backbone(self, x: Tensor) -> Tensor:
        """ResNet (BasicBlock) with every BatchNorm folded into its conv (C10)."""
        x = F.relu(self._conv(x, "conv1"))
        x = F.max_pool2d(x, *self.meta["maxpool"])
        for li, n_blocks in enumerate(self.meta["blocks"], start=1):
            for bi in range(n_blocks):
                p = f"layer{li}.{bi}"
                identity = self._conv(x, f"{p}.downsample") if f"{p}.downsample" in self.meta["convs"] else x
                out = F.relu(self._conv(x, f"{p}.conv1"))
                x = F.relu(self._conv(out, f"{p}.conv2") + identity)
        return x

    def forward(self, state: Tensor, images: Tensor) -> Tensor:
        """
        Args:
            state: (state_dim,) raw robot state (unnormalized).
            images: (n_cams, 3, H, W) camera images in [0, 1], ordered as `meta["camera_keys"]`.
        Returns:
            (chunk_size, action_dim) unnormalized action chunk.
        """
        d, h = self.meta["dim_model"], self.meta["n_heads"]

        # D15: all cameras in one backbone pass; image normalization stays a per-camera runtime op.
        imgs = (images - self.t("img_mean")) * self.t("img_inv_std")
        feat = self.backbone(imgs)  # (n_cams, C, fh, fw)
        feat = feat.flatten(2).transpose(1, 2).reshape(-1, feat.shape[1])  # (n_cams*fh*fw, C)
        cam_tokens = self._linear(feat, "w_img", "b_img")

        state_token = self._linear(state, "w_state", "b_state")  # C11
        x = torch.cat([self.t("latent_token")[None], state_token[None], cam_tokens], dim=0)  # B4

        for i in range(self.meta["n_encoder_layers"]):
            p = f"enc.{i}"
            if self.meta["qkv_mode"] == "add":
                q, k = self._linear(x + self.t("enc_pos"), f"{p}.w_qk", f"{p}.b_qk").split(d, dim=1)
                v = self._linear(x, f"{p}.w_v", f"{p}.b_v")
            else:
                q, k, v = (x @ self.t(f"{p}.w_qkv").T + self.t(f"{p}.b_qkv")).split(d, dim=1)
            x = self._layer_norm(
                x + self._linear(attention(q, k, v, h), f"{p}.w_o", f"{p}.b_o"), f"{p}.norm1"
            )
            ffn = self._linear(F.relu(self._linear(x, f"{p}.w_1", f"{p}.b_1")), f"{p}.w_2", f"{p}.b_2")
            x = self._layer_norm(x + ffn, f"{p}.norm2")

        # Decoder: self-attention folded away (B6), cross-attention query is a constant (B7).
        k = x @ self.t("dec.w_k").T + self.t("dec.b_k")
        v = self._linear(x, "dec.w_v", "dec.b_v")
        attn = self._linear(attention(self.t("dec.q"), k, v, h), "dec.w_o", "dec.b_o")
        y = self._layer_norm(self.t("dec.x") + attn, "dec.norm2")
        ffn = self._linear(F.relu(self._linear(y, "dec.w_1", "dec.b_1")), "dec.w_2", "dec.b_2")
        y = self._layer_norm(y + ffn, "dec.norm3")
        y = self._layer_norm(y, "dec.norm")
        return self._linear(y, "w_act", "b_act")  # C12

    def save(self, output_dir: str | Path) -> Path:
        """Write `act_simplified.safetensors` (meta embedded) and a readable `act_simplified.json`."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tensors = {name: self.t(name).detach().cpu().contiguous() for name in self._names}
        path = output_dir / "act_simplified.safetensors"
        save_file(tensors, path, metadata={META_KEY: json.dumps(self.meta)})
        summary = {
            "meta": self.meta,
            "tensors": {k: list(v.shape) for k, v in tensors.items()},
            "n_params": sum(v.numel() for v in tensors.values()),
        }
        (output_dir / "act_simplified.json").write_text(json.dumps(summary, indent=2))
        return path

    @classmethod
    def from_file(cls, path: str | Path) -> "ACTInference":
        from safetensors import safe_open

        with safe_open(str(path), "pt") as f:
            meta = json.loads(f.metadata()[META_KEY])
        return cls(load_file(path), meta)
