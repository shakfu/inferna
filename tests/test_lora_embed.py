"""Tests for LLM.load_lora / unload_lora / clear_loras and LLM.embed.

The LoRA tests do not require an actual adapter file: they verify the
state-management contract (list bookkeeping, error on unloading an
adapter that was never loaded, no-op on clear with empty list) and
that ``set_adapters_lora`` is wired through to the native context
without crashing on an empty list.

The embed tests use the default test model in embedding mode. Llama 3
is a generative model and ``embed()`` works on it for the purposes of
shape / normalisation verification, even though dedicated embedding
models (BGE etc.) would produce more semantically meaningful vectors.
"""

import gc
import math

import pytest

from inferna import LLM


@pytest.fixture(scope="module")
def llm(model_path: str):
    instance = LLM(model_path, verbose=False)
    yield instance
    instance.close()
    del instance
    gc.collect()


class TestLoraStateManagement:
    def test_initial_list_is_empty(self, llm: LLM) -> None:
        assert llm.list_loras() == []

    def test_clear_with_no_loras_is_noop(self, llm: LLM) -> None:
        llm.clear_loras()
        assert llm.list_loras() == []

    def test_unload_unknown_adapter_raises(self, llm: LLM) -> None:
        # Construct a sentinel that compares non-identical to anything in
        # the empty apply-list. We use object() rather than a real adapter
        # so the test stays adapter-file-free.
        sentinel = object()
        with pytest.raises(ValueError, match="not currently applied"):
            llm.unload_lora(sentinel)  # type: ignore[arg-type]

    def test_load_lora_missing_file_raises(self, llm: LLM, tmp_path) -> None:
        missing = tmp_path / "does-not-exist.gguf"
        with pytest.raises(FileNotFoundError):
            llm.load_lora(str(missing))


class TestLoraContextWiring:
    """The ``set_adapters_lora`` binding must accept an empty apply-list
    on a real context without crashing. Triggering ``_ensure_context``
    via a short generation exercises the post-creation re-apply path."""

    def test_empty_apply_through_generation(self, llm: LLM) -> None:
        # First generation creates a context; the post-creation
        # ``_apply_loras_to(ctx)`` call must accept an empty list.
        result = llm("Say hi.", config=None)
        assert result.text  # generation produced something
        assert llm.list_loras() == []


class TestEmbed:
    def test_embed_scalar_returns_flat_vector(self, llm: LLM) -> None:
        vec = llm.embed("hello world")
        assert isinstance(vec, list)
        assert len(vec) == llm.model.n_embd
        assert all(isinstance(x, float) for x in vec)

    def test_embed_list_returns_list_of_vectors(self, llm: LLM) -> None:
        vecs = llm.embed(["hello", "world", "foo bar"])
        assert isinstance(vecs, list)
        assert len(vecs) == 3
        for v in vecs:
            assert isinstance(v, list)
            assert len(v) == llm.model.n_embd

    def test_embed_empty_list_returns_empty(self, llm: LLM) -> None:
        assert llm.embed([]) == []

    def test_embed_normalize_default_unit_norm(self, llm: LLM) -> None:
        vec = llm.embed("hello world")
        norm = math.sqrt(sum(x * x for x in vec))
        assert norm == pytest.approx(1.0, abs=1e-4)

    def test_embed_normalize_false_keeps_magnitude(self, llm: LLM) -> None:
        vec = llm.embed("hello world", normalize=False)
        norm = math.sqrt(sum(x * x for x in vec))
        # Generative-model hidden states are not unit-norm; just assert
        # we got a non-trivial magnitude that differs from the
        # normalised case.
        assert norm > 1.0

    def test_embed_pooling_strategies_return_correct_shape(self, llm: LLM) -> None:
        # Each pooling strategy returns a vector of n_embd floats.
        # We don't assert pairwise distinctness here: the default test
        # model is a generative LLM, which in embedding mode often
        # returns only one (final-token) hidden state, making mean /
        # cls / last collapse to the same vector. The shape and the
        # absence of crashes are what this test guards.
        text = "the quick brown fox jumps over the lazy dog"
        for pooling in ("mean", "cls", "last"):
            vec = llm.embed(text, pooling=pooling, normalize=False)
            assert isinstance(vec, list)
            assert len(vec) == llm.model.n_embd

    def test_embed_invalid_pooling_raises(self, llm: LLM) -> None:
        with pytest.raises(ValueError, match="Invalid pooling type"):
            llm.embed("hello", pooling="bogus")

    def test_embed_context_reused_across_calls(self, llm: LLM) -> None:
        # First call creates the embed context; subsequent calls must
        # not bump _embed_n_ctx unless the input grows.
        llm.embed("a")
        first_size = llm._embed_n_ctx
        llm.embed("b")
        assert llm._embed_n_ctx == first_size

    def test_embed_does_not_disturb_generation(self, llm: LLM) -> None:
        # Embedding should not poison the generation context's KV state.
        before = llm("Say hi.").text
        llm.embed("some unrelated text")
        after = llm("Say hi.").text
        # Both generations succeed and produce content. (Stochastic
        # sampling means we can't compare equality without a fixed seed,
        # but both should be non-empty.)
        assert before
        assert after

    def test_embed_after_close_recreates_context(self, llm: LLM) -> None:
        llm.embed("warm up")
        assert llm._embed_ctx is not None
        llm.close()
        assert llm._embed_ctx is None
        # Recreate via fresh embed() call.
        vec = llm.embed("hello again")
        assert llm._embed_ctx is not None
        assert len(vec) == llm.model.n_embd
