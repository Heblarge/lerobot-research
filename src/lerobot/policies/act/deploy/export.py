"""Simplify an ACT checkpoint, check it against the original policy and export it.

    uv run python -m lerobot.policies.act.deploy.export \\
        --pretrained-path outputs/train/<run>/checkpoints/last/pretrained_model \\
        --output-dir outputs/deploy/<run>
"""

import argparse
from pathlib import Path

import torch
from torch import Tensor

from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_STATE

from ..modeling_act import ACTPolicy
from .act_infer import ACTInference
from .simplify import QKV_MODES, load_normalization_stats, simplify_act


@torch.no_grad()
def reference_action_chunk(
    policy: ACTPolicy, preprocessor, postprocessor, state: Tensor, images: Tensor
) -> Tensor:
    """Original path: preprocessor -> ACTPolicy.predict_action_chunk -> postprocessor, as (chunk_size, action_dim)."""
    obs = {OBS_STATE: state}
    for key, img in zip(policy.config.image_features, images, strict=True):
        obs[key] = img
    actions = policy.predict_action_chunk(preprocessor(obs))
    return postprocessor(actions)[0]


@torch.no_grad()
def compare(
    pretrained_path: str | Path, simplified: ACTInference, n_samples: int = 3, seed: int = 0
) -> float:
    """Max absolute action error between the original policy and the simplified graph on random inputs."""
    policy = ACTPolicy.from_pretrained(pretrained_path).eval().cpu()
    policy.config.device = "cpu"
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path,
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    meta = simplified.meta
    gen = torch.Generator().manual_seed(seed)
    max_err = 0.0
    for _ in range(n_samples):
        state = torch.randn(meta["state_dim"], generator=gen) * 30
        images = torch.rand(len(meta["camera_keys"]), *meta["image_shape"], generator=gen)
        ref = reference_action_chunk(policy, preprocessor, postprocessor, state, images)
        out = simplified(state, images)
        max_err = max(max_err, (ref - out).abs().max().item())
    return max_err


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pretrained-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--qkv-mode", choices=QKV_MODES, default="add")
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument(
        "--atol", type=float, default=1e-3, help="Max allowed action error (unnormalized units)."
    )
    args = parser.parse_args()

    policy = ACTPolicy.from_pretrained(args.pretrained_path).eval().cpu()
    stats, eps = load_normalization_stats(args.pretrained_path)
    simplified = ACTInference(*simplify_act(policy, stats, eps, qkv_mode=args.qkv_mode))

    n_orig = sum(p.numel() for p in policy.parameters())
    n_new = sum(b.numel() for b in simplified.buffers())
    print(f"params: original {n_orig / 1e6:.2f}M -> simplified {n_new / 1e6:.2f}M (incl. constants)")

    max_err = compare(args.pretrained_path, simplified, n_samples=args.n_samples)
    print(f"max |action error| vs original policy over {args.n_samples} random inputs: {max_err:.3e}")
    if max_err > args.atol:
        raise SystemExit(
            f"Simplified graph deviates from the original policy (> {args.atol}); not exporting."
        )

    path = simplified.save(args.output_dir)
    print(f"exported {path}")


if __name__ == "__main__":
    main()
