from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dflash_mlx.engine.target_ops import bind_draft_to_target
from dflash_mlx.runtime import VerifyConfig
from dflash_mlx.runtime.loading import load_target_bundle
from mlx_lm.utils import load_model

from dfly_mlx.backend import DFlyDraftBackend
from dfly_mlx.model import model_classes

DEFAULT_TARGET = "mlx-community/Qwen3-8B-4bit"
DEFAULT_DRAFT = "AngelSlim/Qwen3-8B-DFly-Block8"
UPSTREAM_REVISION = "5712926"


@dataclass(frozen=True)
class RuntimeBundle:
    target_model: Any
    tokenizer: Any
    target_ops: Any
    draft_model: Any
    draft_backend: DFlyDraftBackend


def _local_path(reference: str | Path) -> Path:
    candidate = Path(reference).expanduser()
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Converted DFly checkpoint not found: {reference}. "
        "Run `dfly-mlx convert --output PATH` first."
    )


def validate_pair(target_config: dict[str, Any], draft_config: dict[str, Any]) -> None:
    checks = {
        "hidden_size": draft_config.get("target_hidden_size"),
        "num_hidden_layers": draft_config.get("target_num_hidden_layers"),
        "vocab_size": draft_config.get("vocab_size"),
    }
    mismatches = [
        f"{key}: target={target_config.get(key)} draft expects={expected}"
        for key, expected in checks.items()
        if expected is not None and int(target_config.get(key, -1)) != int(expected)
    ]
    if mismatches:
        raise ValueError("Incompatible target/drafter pair: " + "; ".join(mismatches))
    layer_ids = draft_config.get("dflash_config", {}).get("target_layer_ids", [])
    if not layer_ids or max(layer_ids) >= int(target_config["num_hidden_layers"]):
        raise ValueError("DFly target_layer_ids are missing or outside the target model")


def load_runtime(target_ref: str, draft_ref: str | Path) -> RuntimeBundle:
    target = load_target_bundle(
        target_ref, lazy=True, verify_config=VerifyConfig.from_mode("dflash")
    )
    draft_path = _local_path(draft_ref)
    draft_config = json.loads((draft_path / "config.json").read_text())
    validate_pair(target.meta["config"], draft_config)
    draft_model, _ = load_model(
        draft_path, lazy=True, get_model_classes=model_classes
    )
    bind_draft_to_target(draft_model, target.model, target_ops=target.target_ops)
    return RuntimeBundle(
        target_model=target.model,
        tokenizer=target.tokenizer,
        target_ops=target.target_ops,
        draft_model=draft_model,
        draft_backend=DFlyDraftBackend(),
    )
