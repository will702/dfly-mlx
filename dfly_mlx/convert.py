from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from dfly_mlx.loading import DEFAULT_DRAFT, UPSTREAM_REVISION
from dfly_mlx.model import normalize_weight_names


def normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    result["architectures"] = ["DFlyDraftModel"]
    result["head_dim"] = int(
        config.get("head_dim")
        or config["hidden_size"] // config["num_attention_heads"]
    )
    result["dflash_config"] = {
        "target_layer_ids": list(config["target_layer_ids"]),
        "mask_token_id": int(config["mask_token_id"]),
        "enable_hidden_correction": bool(config.get("enable_hidden_correction", True)),
        "hidden_correction_intermediate_size": config.get(
            "hidden_correction_intermediate_size"
        ),
    }
    return result


def validate_config(config: dict[str, Any]) -> None:
    expected = {
        "hidden_size": 4096,
        "num_hidden_layers": 5,
        "num_target_layers": 5,
        "target_num_hidden_layers": 36,
        "vocab_size": 151936,
        "block_size": 8,
    }
    mismatches = [
        f"{key}={config.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if config.get(key) != value
    ]
    if list(config.get("target_layer_ids", [])) != [1, 9, 17, 25, 33]:
        mismatches.append("target_layer_ids must be [1, 9, 17, 25, 33]")
    if mismatches:
        raise ValueError("Unsupported or corrupted upstream config: " + "; ".join(mismatches))


def validate_weights(weights: dict[str, Any], config: dict[str, Any]) -> None:
    hidden = int(config["hidden_size"])
    targets = int(config["num_target_layers"])
    expected = {
        "fc.weight": (hidden, hidden * targets),
        "layer_fusion_weights": (int(config["num_hidden_layers"]), targets),
        "hidden_norm.weight": (hidden,),
        "norm.weight": (hidden,),
        "hidden_correction.gate_proj.weight": (hidden, hidden * 2),
        "hidden_correction.up_proj.weight": (hidden, hidden * 2),
        "hidden_correction.down_proj.weight": (hidden, hidden),
    }
    missing = [name for name in expected if name not in weights]
    wrong = [
        f"{name}: {tuple(weights[name].shape)} != {shape}"
        for name, shape in expected.items()
        if name in weights and tuple(weights[name].shape) != shape
    ]
    if missing or wrong:
        details = (["missing " + ", ".join(missing)] if missing else []) + wrong
        raise ValueError("Invalid DFly checkpoint: " + "; ".join(details))


def convert_checkpoint(
    output: str | Path,
    *,
    draft_ref: str = DEFAULT_DRAFT,
    revision: str = UPSTREAM_REVISION,
) -> Path:
    try:
        import torch
        from huggingface_hub import snapshot_download
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError(
            "Checkpoint conversion requires `pip install 'dfly-mlx[convert]'`."
        ) from exc

    local_source = Path(draft_ref).expanduser()
    source = (
        local_source.resolve()
        if local_source.exists()
        else Path(
            snapshot_download(
                repo_id=draft_ref,
                revision=revision,
                allow_patterns=[
                    "config.json",
                    "pytorch_model.bin",
                    "License_AngelSlim_model_and_dataset.txt",
                ],
            )
        )
    )
    config = json.loads((source / "config.json").read_text())
    validate_config(config)
    raw = torch.load(
        source / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True
    )
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    weights = normalize_weight_names(dict(raw))
    validate_weights(weights, config)

    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    save_file(weights, destination / "model.safetensors")
    (destination / "config.json").write_text(
        json.dumps(normalized_config(config), indent=2) + "\n"
    )
    license_path = source / "License_AngelSlim_model_and_dataset.txt"
    if license_path.exists():
        shutil.copy2(license_path, destination / license_path.name)
    return destination
