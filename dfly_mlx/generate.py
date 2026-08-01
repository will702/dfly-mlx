from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from typing import Any

from dflash_mlx.engine.events import SummaryEvent, TokenEvent
from dflash_mlx.runtime import get_stop_token_ids, stream_dflash_generate
from dflash_mlx.runtime.context import build_offline_runtime_context

from dfly_mlx.loading import RuntimeBundle


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: tuple[int, ...]
    tokens_per_second: float
    accepted_per_cycle: float
    acceptance_ratio: float
    peak_memory_gb: float | None
    summary: dict[str, Any]


def no_thinking_prompt_tokens(tokenizer: Any, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    try:
        return list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    except TypeError as exc:
        raise RuntimeError(
            "The tokenizer cannot build Qwen3 no-thinking prompts; upgrade transformers."
        ) from exc


def generate(
    bundle: RuntimeBundle,
    prompt: str,
    *,
    max_tokens: int = 256,
    stream: bool = False,
) -> GenerationResult:
    token_ids: list[int] = []
    text_parts: list[str] = []
    summary: SummaryEvent | None = None
    runtime_context = build_offline_runtime_context(verify_mode="dflash")
    if hasattr(runtime_context.runtime, "copyspec_mode"):
        runtime_context = replace(
            runtime_context,
            runtime=replace(runtime_context.runtime, copyspec_mode="off"),
        )
    events = stream_dflash_generate(
        target_model=bundle.target_model,
        target_ops=bundle.target_ops,
        tokenizer=bundle.tokenizer,
        draft_model=bundle.draft_model,
        draft_backend=bundle.draft_backend,
        prompt=prompt,
        max_new_tokens=max_tokens,
        use_chat_template=False,
        block_tokens=int(bundle.draft_model.block_size),
        stop_token_ids=get_stop_token_ids(bundle.tokenizer),
        prompt_tokens_override=no_thinking_prompt_tokens(bundle.tokenizer, prompt),
        runtime_context=runtime_context,
    )
    for event in events:
        if isinstance(event, TokenEvent):
            token_ids.append(int(event.token_id))
            piece = str(bundle.tokenizer.decode([int(event.token_id)]))
            text_parts.append(piece)
            if stream:
                sys.stdout.write(piece)
                sys.stdout.flush()
        elif isinstance(event, SummaryEvent):
            summary = event
    if summary is None:
        raise RuntimeError("DFly generation ended without a summary")
    payload = summary.to_payload()
    generation_us = max(
        0.0,
        float(summary.elapsed_us) - float(summary.phase_timings_us.get("prefill", 0.0)),
    )
    tps = len(token_ids) / (generation_us / 1e6) if generation_us else 0.0
    accepted_per_cycle = (
        sum(summary.acceptance_history) / len(summary.acceptance_history)
        if summary.acceptance_history
        else 0.0
    )
    return GenerationResult(
        text="".join(text_parts),
        token_ids=tuple(token_ids),
        tokens_per_second=tps,
        accepted_per_cycle=float(accepted_per_cycle),
        acceptance_ratio=float(summary.acceptance_ratio),
        peak_memory_gb=summary.peak_memory_gb,
        summary=payload,
    )
