from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from dfly_mlx.loading import DEFAULT_DRAFT, DEFAULT_TARGET, UPSTREAM_REVISION


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dfly-mlx", description="DFly speculative decoding on Apple Silicon"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    convert = commands.add_parser("convert", help="Convert the official DFly checkpoint")
    convert.add_argument("--draft", default=DEFAULT_DRAFT)
    convert.add_argument("--revision", default=UPSTREAM_REVISION)
    convert.add_argument("--output", required=True)

    generate = commands.add_parser("generate", help="Generate with lossless greedy DFly")
    generate.add_argument("--model", default=DEFAULT_TARGET)
    generate.add_argument("--draft", required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-tokens", type=int, default=256)

    bench = commands.add_parser("bench", help="Benchmark baseline against DFly")
    bench.add_argument("--model", default=DEFAULT_TARGET)
    bench.add_argument("--draft", required=True)
    bench.add_argument(
        "--prompts",
        default=str(Path(__file__).resolve().parent / "prompts.jsonl"),
    )
    bench.add_argument("--trials", type=int, default=3)
    bench.add_argument("--max-tokens", type=int, default=128)
    bench.add_argument("--output", default="benchmark.json")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if getattr(args, "max_tokens", 1) <= 0:
        raise SystemExit("--max-tokens must be positive")
    if getattr(args, "trials", 1) <= 0:
        raise SystemExit("--trials must be positive")

    if args.command == "convert":
        from dfly_mlx.convert import convert_checkpoint

        print(convert_checkpoint(args.output, draft_ref=args.draft, revision=args.revision))
        return

    if args.command == "generate":
        from dfly_mlx.generate import generate
        from dfly_mlx.loading import load_runtime

        bundle = load_runtime(args.model, args.draft)
        result = generate(bundle, args.prompt, max_tokens=args.max_tokens, stream=True)
        print(
            f"\n{len(result.token_ids)} tokens | {result.tokens_per_second:.1f} tok/s | "
            f"{result.accepted_per_cycle:.2f} tokens/cycle | "
            f"{result.acceptance_ratio:.1%} draft acceptance",
            file=sys.stderr,
        )
        return

    from dfly_mlx.benchmark import load_prompts, run_benchmark

    report = run_benchmark(
        model_ref=args.model,
        draft_ref=args.draft,
        prompts=load_prompts(args.prompts),
        trials=args.trials,
        max_tokens=args.max_tokens,
    )
    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output.resolve())


if __name__ == "__main__":
    main()
