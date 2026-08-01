from types import SimpleNamespace

import mlx.core as mx

from dflash_mlx.model import DFlashDraftModelArgs
from dfly_mlx.backend import DFlyDraftBackend
from dfly_mlx.model import DFlyDraftModel, HiddenStatesCorrection, model_classes


def test_mlx_lm_model_factory_accepts_keyword_config():
    assert model_classes(config={}) == (DFlyDraftModel, DFlashDraftModelArgs)


def tiny_args() -> DFlashDraftModelArgs:
    return DFlashDraftModelArgs.from_dict(
        {
            "model_type": "qwen3",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "rms_norm_eps": 1e-6,
            "vocab_size": 32,
            "num_key_value_heads": 1,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "head_dim": 4,
            "tie_word_embeddings": False,
            "num_target_layers": 2,
            "block_size": 4,
            "dflash_config": {
                "target_layer_ids": [1, 3],
                "mask_token_id": 31,
                "enable_hidden_correction": True,
            },
        }
    )


def test_dfly_context_fusion_and_correction_shapes():
    model = DFlyDraftModel(tiny_args())
    contexts = model._layer_contexts(mx.ones((1, 3, 16)))
    correction = HiddenStatesCorrection(8, 8, 1e-6)
    corrected = correction(mx.ones((2, 8)), mx.ones((2, 8)))
    mx.eval(*contexts, corrected)
    assert [tuple(context.shape) for context in contexts] == [(1, 3, 8), (1, 3, 8)]
    assert tuple(corrected.shape) == (2, 8)


def test_dfly_backend_conditions_each_step_on_previous_sample():
    class Draft:
        def __init__(self):
            self.seen = []

        def forward_projected_context(self, **_kwargs):
            return mx.array([[[0.0], [1.0], [2.0]]])

        def correct_hidden(self, hidden, previous_embedding):
            self.seen.append(int(previous_embedding.item()))
            return hidden + previous_embedding

    class Ops:
        @staticmethod
        def embed_tokens(_target):
            return lambda ids: ids.astype(mx.float32)[..., None]

        @staticmethod
        def logits_from_hidden(_target, hidden):
            values = hidden[..., 0, None]
            return -mx.abs(mx.arange(8, dtype=mx.float32) - values)

    draft = Draft()
    result = DFlyDraftBackend().draft_greedy(
        target_model=object(),
        target_ops=Ops(),
        draft_model=draft,
        draft_cache=[],
        staged_first=mx.array([1], dtype=mx.uint32),
        draft_context=mx.zeros((1, 1, 1)),
        block_len=3,
        mask_token_tail=mx.array([7, 7], dtype=mx.uint32),
        suppress_token_mask=None,
        async_launch=False,
    )
    assert result.tolist() == [2, 4]
    assert draft.seen == [1, 2]
