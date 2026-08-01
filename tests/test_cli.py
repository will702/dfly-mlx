import pytest

from dfly_mlx.cli import _parser, main


def test_cli_defaults_to_pinned_upstream_revision():
    args = _parser().parse_args(["convert", "--output", "model"])
    assert args.revision == "5712926"
    assert args.draft == "AngelSlim/Qwen3-8B-DFly-Block8"


def test_cli_rejects_nonpositive_generation_length():
    with pytest.raises(SystemExit, match="max-tokens"):
        main(
            [
                "generate",
                "--draft",
                "model",
                "--prompt",
                "hello",
                "--max-tokens",
                "0",
            ]
        )
