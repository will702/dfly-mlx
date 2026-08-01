from types import SimpleNamespace

import pytest

from dfly_mlx.convert import normalized_config, validate_config, validate_weights
from dfly_mlx.model import normalize_weight_name


def config():
    return {
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 5,
        "num_target_layers": 5,
        "target_num_hidden_layers": 36,
        "target_layer_ids": [1, 9, 17, 25, 33],
        "vocab_size": 151936,
        "mask_token_id": 151669,
        "block_size": 8,
        "enable_hidden_correction": True,
    }


def test_config_normalization_pins_target_features():
    result = normalized_config(config())
    assert result["architectures"] == ["DFlyDraftModel"]
    assert result["head_dim"] == 128
    assert result["dflash_config"]["target_layer_ids"] == [1, 9, 17, 25, 33]
    assert result["dflash_config"]["mask_token_id"] == 151669


def test_rejects_upstream_hy3_metadata_regression():
    broken = config() | {"target_num_hidden_layers": 80, "vocab_size": 120832}
    with pytest.raises(ValueError, match="corrupted upstream config"):
        validate_config(broken)


def test_weight_name_mapping():
    assert normalize_weight_name("draft_model.context_proj.weight") == "fc.weight"
    assert normalize_weight_name("context_norm.weight") == "hidden_norm.weight"
    assert normalize_weight_name("final_norm.weight") == "norm.weight"
    assert normalize_weight_name("lm_head.weight") is None


def test_weight_shape_gate():
    h = 4096
    weights = {
        "fc.weight": SimpleNamespace(shape=(h, h * 5)),
        "layer_fusion_weights": SimpleNamespace(shape=(5, 5)),
        "hidden_norm.weight": SimpleNamespace(shape=(h,)),
        "norm.weight": SimpleNamespace(shape=(h,)),
        "hidden_correction.gate_proj.weight": SimpleNamespace(shape=(h, h * 2)),
        "hidden_correction.up_proj.weight": SimpleNamespace(shape=(h, h * 2)),
        "hidden_correction.down_proj.weight": SimpleNamespace(shape=(h, h)),
    }
    validate_weights(weights, config())
    weights["layer_fusion_weights"] = SimpleNamespace(shape=(4, 5))
    with pytest.raises(ValueError, match="layer_fusion_weights"):
        validate_weights(weights, config())
