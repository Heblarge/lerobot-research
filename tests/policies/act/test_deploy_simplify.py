import pytest
import torch
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.deploy import ACTInference, simplify_act
from lerobot.policies.act.deploy.export import reference_action_chunk
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

CAMERAS = ("observation.images.up", "observation.images.side")
IMG_SHAPE = (3, 64, 96)
EPS = 1e-8


def make_policy(**overrides) -> ACTPolicy:
    input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,))}
    input_features |= {k: PolicyFeature(type=FeatureType.VISUAL, shape=IMG_SHAPE) for k in CAMERAS}
    config = ACTConfig(
        input_features=input_features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        device="cpu",
        pretrained_backbone_weights=None,
        dim_model=64,
        n_heads=4,
        dim_feedforward=128,
        chunk_size=10,
        n_action_steps=10,
        **overrides,
    )
    torch.manual_seed(0)
    policy = ACTPolicy(config).eval()
    # Randomize the frozen BN statistics so that folding them (C10) is actually exercised.
    for module in policy.modules():
        if isinstance(module, FrozenBatchNorm2d):
            n = module.weight.numel()
            module.weight.copy_(torch.rand(n) + 0.5)
            module.bias.copy_(torch.randn(n) * 0.1)
            module.running_mean.copy_(torch.randn(n) * 0.1)
            module.running_var.copy_(torch.rand(n) + 0.5)
    return policy


def make_stats() -> dict[str, dict[str, torch.Tensor]]:
    stats = {
        OBS_STATE: {"mean": torch.randn(6) * 10, "std": torch.rand(6) * 20 + 1},
        ACTION: {"mean": torch.randn(6) * 10, "std": torch.rand(6) * 20 + 1},
    }
    for key in CAMERAS:
        stats[key] = {"mean": torch.rand(3, 1, 1), "std": torch.rand(3, 1, 1) * 0.3 + 0.1}
    return stats


def flatten_stats(stats):
    return {f"{key}.{name}": value for key, values in stats.items() for name, value in values.items()}


@pytest.mark.parametrize("qkv_mode", ["add", "table"])
def test_simplified_matches_original(qkv_mode, tmp_path):
    policy = make_policy()
    stats = make_stats()
    preprocessor, postprocessor = make_act_pre_post_processors(policy.config, dataset_stats=stats)
    simplified = ACTInference(*simplify_act(policy, flatten_stats(stats), EPS, qkv_mode=qkv_mode))

    gen = torch.Generator().manual_seed(1)
    state = torch.randn(6, generator=gen) * 10
    images = torch.rand(len(CAMERAS), *IMG_SHAPE, generator=gen)
    ref = reference_action_chunk(policy, preprocessor, postprocessor, state, images)
    with torch.no_grad():
        out = simplified(state, images)
    assert out.shape == (policy.config.chunk_size, 6)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    # The exported file reproduces the same outputs.
    reloaded = ACTInference.from_file(simplified.save(tmp_path))
    with torch.no_grad():
        torch.testing.assert_close(reloaded(state, images), out)


def test_vae_encoder_is_dropped():
    policy = make_policy()
    tensors, _ = simplify_act(policy, flatten_stats(make_stats()), EPS)
    assert not any(name.startswith("vae") for name in tensors)
    assert not any("self_attn" in name for name in tensors)


@pytest.mark.parametrize("overrides", [{"n_decoder_layers": 2}, {"pre_norm": True}])
def test_unsupported_config_is_rejected(overrides):
    policy = make_policy(**overrides)
    with pytest.raises(ValueError):
        simplify_act(policy, flatten_stats(make_stats()), EPS)
