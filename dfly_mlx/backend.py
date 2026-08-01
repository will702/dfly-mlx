# Copyright 2026 Willson
# Licensed under the Apache License, Version 2.0.
# Adapted from vLLM's DFly proposer and bstnxbt/dflash-mlx; see NOTICE.

from __future__ import annotations

from typing import Any

import mlx.core as mx
from dflash_mlx.draft_backend import EagerDraftBackend, _astype_if_needed, _draft_compute_dtype
from dflash_mlx.engine.sampling import greedy_tokens_with_mask


class DFlyDraftBackend(EagerDraftBackend):
    """Run the parallel backbone once, then correct/sample left-to-right."""

    def draft_greedy(
        self,
        *,
        target_model: Any,
        target_ops: Any,
        draft_model: Any,
        draft_cache: list[Any],
        staged_first: mx.array,
        draft_context: mx.array,
        block_len: int,
        mask_token_tail: mx.array,
        suppress_token_mask: mx.array | None,
        async_launch: bool,
    ) -> mx.array:
        if int(block_len) <= 1:
            raise ValueError("DFly drafting requires block_len > 1")
        block_ids = mx.concatenate(
            [staged_first[:1], mask_token_tail[: int(block_len) - 1]], axis=0
        )
        dtype = _draft_compute_dtype(draft_model)
        embedding = target_ops.embed_tokens(target_model)
        noise = embedding(block_ids[None])
        if dtype is not None:
            noise = _astype_if_needed(noise, dtype)
            draft_context = _astype_if_needed(draft_context, dtype)
        hidden = draft_model.forward_projected_context(
            noise_embedding=noise,
            draft_context=draft_context,
            cache=draft_cache,
        )

        previous = staged_first[:1]
        drafted: list[mx.array] = []
        for step in range(1, int(block_len)):
            previous_embedding = embedding(previous[None])[:, 0, :]
            if dtype is not None:
                previous_embedding = _astype_if_needed(previous_embedding, dtype)
            corrected = draft_model.correct_hidden(
                hidden[:, step, :], previous_embedding
            )
            logits = target_ops.logits_from_hidden(
                target_model, corrected[:, None, :]
            )
            token = greedy_tokens_with_mask(logits, suppress_token_mask).reshape(-1)[0]
            mx.eval(token)
            drafted.append(token)
            previous = token.reshape(1)
        result = mx.stack(drafted).astype(mx.uint32)
        if async_launch:
            mx.async_eval(result)
        else:
            mx.eval(result)
        return result
