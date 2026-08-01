# Copyright 2026 Willson
# Licensed under the Apache License, Version 2.0.
# Adapted from Tencent AngelSpec and bstnxbt/dflash-mlx; see NOTICE.

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
from dflash_mlx.model import DFlashDraftModel, DFlashDraftModelArgs


class HiddenStatesCorrection(nn.Module):
    """Predecessor-conditioned SwiGLU residual from DFly/TreeFlash."""

    def __init__(self, hidden_size: int, intermediate_size: int, eps: float):
        super().__init__()
        self.hidden_norm = nn.RMSNorm(hidden_size, eps=eps)
        self.embed_norm = nn.RMSNorm(hidden_size, eps=eps)
        self.gate_proj = nn.Linear(hidden_size * 2, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size * 2, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, hidden: mx.array, previous_embedding: mx.array) -> mx.array:
        joined = mx.concatenate(
            [self.hidden_norm(hidden), self.embed_norm(previous_embedding)], axis=-1
        )
        delta = self.down_proj(nn.silu(self.gate_proj(joined)) * self.up_proj(joined))
        return hidden + delta.astype(hidden.dtype)


class DFlyDraftModel(DFlashDraftModel):
    """DFlash backbone with per-layer residual fusion and hidden correction."""

    def __init__(self, args: DFlashDraftModelArgs):
        super().__init__(args)
        config = args.dflash_config or {}
        self.model_type = "dfly_qwen3"
        self.num_target_layers = len(self.target_layer_ids)
        if self.num_target_layers != int(args.num_target_layers):
            raise ValueError(
                "DFly target_layer_ids count must equal num_target_layers: "
                f"{self.num_target_layers} != {args.num_target_layers}"
            )
        self.layer_fusion_weights = mx.zeros(
            (len(self.layers), self.num_target_layers), dtype=mx.float32
        )
        if bool(config.get("enable_hidden_correction", True)):
            intermediate = int(
                config.get("hidden_correction_intermediate_size") or args.hidden_size
            )
            self.hidden_correction = HiddenStatesCorrection(
                args.hidden_size, intermediate, args.rms_norm_eps
            )
        else:
            self.hidden_correction = None

    def project_target_hidden(self, target_hidden: mx.array) -> mx.array:
        expected = self.num_target_layers * self.args.hidden_size
        if int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                f"DFly expected {expected} concatenated target features, "
                f"got {target_hidden.shape[-1]}"
            )
        return target_hidden

    def _layer_contexts(self, draft_context: mx.array) -> list[mx.array]:
        batch, length, width = draft_context.shape
        expected = self.num_target_layers * self.args.hidden_size
        if int(width) != expected:
            raise ValueError(f"DFly context width must be {expected}, got {width}")
        stacked = draft_context.reshape(
            batch, length, self.num_target_layers, self.args.hidden_size
        )
        base = self.fc(draft_context)
        fusion = mx.softmax(self.layer_fusion_weights, axis=-1)
        return [
            self.hidden_norm(
                base + mx.einsum("t,bstd->bsd", fusion[index], stacked)
            )
            for index in range(len(self.layers))
        ]

    def forward_projected_context(
        self,
        *,
        noise_embedding: mx.array,
        draft_context: mx.array,
        cache: list[Any] | None = None,
    ) -> mx.array:
        hidden = noise_embedding * self.embed_scale
        caches = cache if cache is not None else [None] * len(self.layers)
        for layer, layer_cache, context in zip(
            self.layers, caches, self._layer_contexts(draft_context), strict=True
        ):
            hidden = layer(hidden, target_hidden=context, cache=layer_cache)
        return self.norm(hidden)

    def advance_projected_context_cache(
        self, *, draft_context: mx.array, cache: list[Any]
    ) -> None:
        if cache is None:
            raise ValueError("draft context cache is required")
        for layer, layer_cache, context in zip(
            self.layers, cache, self._layer_contexts(draft_context), strict=True
        ):
            layer.advance_projected_context_cache(
                target_hidden=context, cache=layer_cache
            )

    def correct_hidden(
        self, hidden: mx.array, previous_embedding: mx.array
    ) -> mx.array:
        if self.hidden_correction is None:
            return hidden
        return self.hidden_correction(hidden, previous_embedding)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        return normalize_weight_names(weights)


def normalize_weight_name(name: str) -> str | None:
    """Map AngelSpec training names to the compact MLX inference model."""
    for prefix in ("module.draft_model.", "draft_model.", "module.", "model."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    if name.startswith(("embed_tokens.", "lm_head.", "markov_head.", "confidence_head.")):
        return None
    replacements = {
        "context_proj.": "fc.",
        "context_norm.": "hidden_norm.",
        "final_norm.": "norm.",
    }
    for source, destination in replacements.items():
        if name.startswith(source):
            return destination + name[len(source) :]
    return name


def normalize_weight_names(weights: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for name, value in weights.items():
        mapped = normalize_weight_name(name)
        if mapped is not None:
            normalized[mapped] = value
    return normalized


def model_classes(config: dict[str, Any]):
    return DFlyDraftModel, DFlashDraftModelArgs
