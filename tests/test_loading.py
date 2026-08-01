import pytest

from dfly_mlx.loading import validate_pair


def test_pair_validation_accepts_qwen3_8b():
    validate_pair(
        {"hidden_size": 4096, "num_hidden_layers": 36, "vocab_size": 151936},
        {
            "target_hidden_size": 4096,
            "target_num_hidden_layers": 36,
            "vocab_size": 151936,
            "dflash_config": {"target_layer_ids": [1, 9, 17, 25, 33]},
        },
    )


def test_pair_validation_rejects_wrong_target():
    with pytest.raises(ValueError, match="Incompatible target/drafter pair"):
        validate_pair(
            {"hidden_size": 4096, "num_hidden_layers": 80, "vocab_size": 120832},
            {
                "target_hidden_size": 4096,
                "target_num_hidden_layers": 36,
                "vocab_size": 151936,
                "dflash_config": {"target_layer_ids": [1, 9, 17, 25, 33]},
            },
        )
