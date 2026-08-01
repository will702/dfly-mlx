from __future__ import annotations

import gc
import json
import platform
import statistics
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import mlx.core as mx
from dflash_mlx.engine.events import SummaryEvent, TokenEvent
from dflash_mlx.engine.fallback import stream_baseline_generate
from dflash_mlx.runtime import VerifyConfig, get_stop_token_ids
from dflash_mlx.runtime.loading import load_target_bundle

from dfly_mlx.generate import generate, no_thinking_prompt_tokens
from dfly_mlx.loading import load_runtime


def _version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _sysctl(name: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def hardware() -> dict[str, Any]:
    memory = _sysctl("hw.memsize")
    return {
        "chip": _sysctl("machdep.cpu.brand_string"),
        "memory_gb": int(memory) // (1024**3) if memory.isdigit() else "unknown",
        "python": platform.python_version(),
        "mlx": getattr(mx, "__version__", "unknown"),
        "mlx_lm": _version("mlx-lm"),
        "dflash_mlx": _version("dflash-mlx"),
        "dfly_mlx": _version("dfly-mlx"),
    }


def _baseline_once(
    model: Any, target_ops: Any, tokenizer: Any, prompt: str, max_tokens: int
) -> dict[str, Any]:
    prompt_tokens = no_thinking_prompt_tokens(tokenizer, prompt)
    token_ids: list[int] = []
    summary = None
    for event in stream_baseline_generate(
        target_model=model,
        target_ops=target_ops,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_tokens,
        stop_token_ids=get_stop_token_ids(tokenizer),
        prompt_tokens_override=prompt_tokens,
    ):
        if isinstance(event, TokenEvent):
            token_ids.append(int(event.token_id))
        elif isinstance(event, SummaryEvent):
            summary = event
    if summary is None:
        raise RuntimeError("Baseline generation ended without a summary")
    generation_us = max(
        0.0,
        float(summary.elapsed_us) - float(summary.phase_timings_us.get("prefill", 0.0)),
    )
    return {
        "token_ids": token_ids,
        "tokens": len(token_ids),
        "tokens_per_second": (
            len(token_ids) / (generation_us / 1e6) if generation_us else 0.0
        ),
        "peak_memory_gb": summary.peak_memory_gb,
    }


def run_benchmark(
    *,
    model_ref: str,
    draft_ref: str,
    prompts: list[dict[str, str]],
    trials: int,
    max_tokens: int,
) -> dict[str, Any]:
    baseline = load_target_bundle(
        model_ref, lazy=True, verify_config=VerifyConfig.from_mode("dflash")
    )
    _baseline_once(
        baseline.model, baseline.target_ops, baseline.tokenizer, prompts[0]["prompt"], 8
    )
    baseline_runs: dict[str, list[dict[str, Any]]] = {p["id"]: [] for p in prompts}
    for prompt in prompts:
        for _ in range(trials):
            baseline_runs[prompt["id"]].append(
                _baseline_once(
                    baseline.model,
                    baseline.target_ops,
                    baseline.tokenizer,
                    prompt["prompt"],
                    max_tokens,
                )
            )
    del baseline
    gc.collect()
    mx.clear_cache()

    bundle = load_runtime(model_ref, draft_ref)
    generate(bundle, prompts[0]["prompt"], max_tokens=8)
    dfly_runs: dict[str, list[dict[str, Any]]] = {p["id"]: [] for p in prompts}
    for prompt in prompts:
        for trial in range(trials):
            result = generate(bundle, prompt["prompt"], max_tokens=max_tokens)
            baseline_ids = baseline_runs[prompt["id"]][trial]["token_ids"]
            if list(result.token_ids) != baseline_ids:
                first_mismatch = next(
                    (
                        index
                        for index, pair in enumerate(
                            zip(result.token_ids, baseline_ids, strict=False)
                        )
                        if pair[0] != pair[1]
                    ),
                    min(len(result.token_ids), len(baseline_ids)),
                )
                raise AssertionError(
                    f"Lossless parity failed for prompt {prompt['id']!r}, "
                    f"trial {trial + 1}, token {first_mismatch}: "
                    f"baseline={baseline_ids} dfly={list(result.token_ids)}"
                )
            dfly_runs[prompt["id"]].append(
                {
                    "token_ids": list(result.token_ids),
                    "tokens": len(result.token_ids),
                    "tokens_per_second": result.tokens_per_second,
                    "accepted_per_cycle": result.accepted_per_cycle,
                    "acceptance_ratio": result.acceptance_ratio,
                    "peak_memory_gb": result.peak_memory_gb,
                }
            )

    rows = []
    for prompt in prompts:
        key = prompt["id"]
        baseline_tps = statistics.median(
            run["tokens_per_second"] for run in baseline_runs[key]
        )
        dfly_tps = statistics.median(
            run["tokens_per_second"] for run in dfly_runs[key]
        )
        rows.append(
            {
                "id": key,
                "baseline_tps_median": baseline_tps,
                "dfly_tps_median": dfly_tps,
                "speedup": dfly_tps / baseline_tps if baseline_tps else 0.0,
                "accepted_per_cycle_median": statistics.median(
                    run["accepted_per_cycle"] for run in dfly_runs[key]
                ),
                "parity": True,
                "baseline_trials": baseline_runs[key],
                "dfly_trials": dfly_runs[key],
            }
        )
    return {
        "schema_version": 1,
        "model": model_ref,
        "draft": draft_ref,
        "trials": trials,
        "max_tokens": max_tokens,
        "hardware": hardware(),
        "results": rows,
    }


def load_prompts(path: str | Path) -> list[dict[str, str]]:
    prompts = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not prompts or any(set(item) < {"id", "prompt"} for item in prompts):
        raise ValueError("Prompt file must contain JSONL objects with id and prompt")
    return prompts
