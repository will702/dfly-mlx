# dfly-mlx

[![PyPI](https://img.shields.io/pypi/v/dfly-mlx)](https://pypi.org/project/dfly-mlx/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-MLX%20Community-yellow)](https://huggingface.co/mlx-community/Qwen3-8B-DFly-MLX)
[![GitHub release](https://img.shields.io/github/v/release/will702/dfly-mlx)](https://github.com/will702/dfly-mlx/releases/latest)

Native [DFly](https://arxiv.org/abs/2607.25852) speculative decoding for Apple
Silicon. One MLX target, one official drafter, one reproducible benchmark.

DFly runs a five-layer block-parallel draft backbone, fuses five intermediate
target representations differently at every draft layer, then samples the
seven proposals left-to-right through its lightweight hidden-correction head.
The Qwen3-8B target verifies every proposal before it is committed.

> Early implementation: conversion and generation pass on a 16 GB Apple M5,
> but no performance number is published yet. The 4-bit target can choose a
> different greedy token at near-tied logits when run as a verification block
> instead of one token at a time, so the strict parity benchmark currently
> fails closed on affected prompts.

## Requirements

- Apple Silicon Mac
- Python 3.10–3.13 and MLX/MLX-LM 0.31 or newer
- About 7 GB for the 4-bit target plus converted drafter weights; keep roughly
  10 GB free during conversion and leave unified-memory headroom for KV caches

## Install

From [PyPI](https://pypi.org/project/dfly-mlx/):

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install dfly-mlx
```

Or with [Homebrew](https://github.com/will702/homebrew-tap):

```sh
brew install will702/tap/dfly-mlx
```

For development from source:

```sh
git clone https://github.com/will702/dfly-mlx.git
cd dfly-mlx
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[convert,test]'
```

## Convert the official checkpoint

The upstream Hugging Face repository currently has a bad latest `config.json`:
it describes an 80-layer target instead of Qwen3-8B. The converter therefore
pins the last correct upstream revision, `5712926`, validates its architecture,
and writes MLX safetensors.

```sh
pip install 'dfly-mlx[convert]'
dfly-mlx convert --output ./models/qwen3-8b-dfly-mlx
```

An already-downloaded checkpoint directory can be supplied with `--draft`;
the same metadata and tensor-shape gates still apply.

The output remains an Apache-2.0 AngelSlim model derivative and includes its
model license.

Or download the ready-to-use MLX conversion:

```sh
hf download mlx-community/Qwen3-8B-DFly-MLX \
  --local-dir ./models/qwen3-8b-dfly-mlx
```

## Generate

```sh
dfly-mlx generate \
  --draft ./models/qwen3-8b-dfly-mlx \
  --prompt 'Write a quicksort in Python.' \
  --max-tokens 256
```

The target defaults to `mlx-community/Qwen3-8B-4bit`. Prompts use Qwen3's
no-thinking chat template because the released DFly drafter was trained in
no-thinking mode.

## Reproduce the benchmark

```sh
dfly-mlx bench \
  --draft ./models/qwen3-8b-dfly-mlx \
  --trials 3 \
  --output benchmark.json
```

The benchmark warms both paths, records exact software and hardware metadata,
and refuses to publish a row unless target-only and DFly token IDs match.
Results are medians across trials, not best runs. This is deliberately stricter
than the usual speculative-decoding guarantee, where every emitted proposal is
accepted by the batched target verifier.

## v0.1 boundaries

Qwen3-8B, no-thinking, greedy decoding, and one request at a time. There is no
server or training stack yet. Add those only after the native path demonstrates
a repeatable speedup.

## Attribution

DFly was introduced by Tencent's AngelSpec team. The MLX target execution and
draft/verify machinery come from `dflash-mlx`. See [NOTICE](NOTICE) for precise
attribution and licenses.
