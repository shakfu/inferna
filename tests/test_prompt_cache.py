"""Tests for the prompt-cache layer (KV reuse across calls).

Coverage:
  * Shadow tracking: ``LLM._kv_seq_tokens`` reflects the actual KV
    state after each generation (prompt + sampled).
  * Prefix-share fast path: an identical follow-up call is materially
    faster than the first (full-prefill) call, and produces identical
    output under a deterministic seed.
  * Invalidation: ``reset_context()``, ``close()``, and LoRA changes
    drop the shadow.
  * Boundary correctness: the last prompt token is reserved for fresh
    decode so the sampler always reads valid logits, even when the
    new prompt is exactly equal to the previously cached prompt.
"""

from __future__ import annotations

import gc
import time

import pytest

from inferna import LLM, GenerationConfig


@pytest.fixture(scope="module")
def llm(model_path: str):
    instance = LLM(model_path, verbose=False)
    yield instance
    instance.close()
    del instance
    gc.collect()


@pytest.fixture
def deterministic_config() -> GenerationConfig:
    return GenerationConfig(max_tokens=10, temperature=0.0, seed=42)


class TestShadowTracking:
    def test_shadow_starts_empty(self, model_path: str) -> None:
        # A fresh LLM has no KV state yet.
        llm = LLM(model_path, verbose=False)
        try:
            assert llm._kv_seq_tokens == []
        finally:
            llm.close()
            del llm
            gc.collect()

    def test_shadow_populated_after_call(self, llm: LLM, deterministic_config) -> None:
        llm.reset_context()
        r = llm("hello world", config=deterministic_config)
        # Shadow holds the prompt tokens followed by the sampled tokens
        # (which is what's currently in seq 0's KV).
        prompt_tokens = llm.vocab.tokenize("hello world", add_special=True, parse_special=False)
        sampled = llm.vocab.tokenize(r.text, add_special=False, parse_special=False)
        assert len(llm._kv_seq_tokens) >= len(prompt_tokens)
        assert llm._kv_seq_tokens[: len(prompt_tokens)] == list(prompt_tokens)
        # The exact tail length depends on whether the model EOG'd
        # before max_tokens, so just check the prefix matches.
        assert llm._kv_seq_tokens[-1] == (sampled[-1] if sampled else prompt_tokens[-1])


class TestPrefixHitFastPath:
    def test_identical_prompt_is_faster(self, llm: LLM, deterministic_config) -> None:
        """A long-prefix prompt repeated should be materially faster
        the second time. We don't pin a hard ratio (CI noise) but
        require the second call to be at least 1.5× faster than a
        full-prefill baseline to show the cache is doing real work.
        """
        # Long prompt so prefill dominates the first call.
        prefix = "You are a helpful assistant. " * 30 + "Now tell me about"
        prompt = prefix + " the moon."

        llm.reset_context()
        # Warm up once so prior CT-state doesn't perturb timings.
        llm(prompt, config=deterministic_config)

        # Force a full prefill, time it.
        llm.reset_context()
        t0 = time.perf_counter()
        llm(prompt, config=deterministic_config)
        t_cold = time.perf_counter() - t0

        # Cache should be populated now; identical prompt = full prefix hit.
        t0 = time.perf_counter()
        llm(prompt, config=deterministic_config)
        t_warm = time.perf_counter() - t0

        assert t_warm < t_cold / 1.5, f"warm={t_warm * 1000:.0f}ms cold={t_cold * 1000:.0f}ms"

    def test_identical_prompt_is_deterministic(self, llm: LLM, deterministic_config) -> None:
        """Cache hit must produce byte-identical output to a cold run.

        This is correctness, not perf: prefix replay must be
        equivalent to recomputation.
        """
        prompt = "The capital of France is"
        llm.reset_context()
        cold = llm(prompt, config=deterministic_config).text

        # Same call again -- cache hit reserves the last token but
        # otherwise replays.
        warm = llm(prompt, config=deterministic_config).text
        assert cold == warm

    def test_extended_prompt_partial_hit(self, llm: LLM, deterministic_config) -> None:
        """A new prompt that extends the cached one should still
        produce coherent output (and not crash on the partial-hit
        path)."""
        base = "Once upon a time"
        ext = base + ", in a land far away,"

        llm.reset_context()
        llm(base, config=deterministic_config)
        # Cache holds base+sampled; new prompt diverges from sampled
        # at the comma. Partial hit on `base` itself (or up to
        # whatever continues to match).
        r = llm(ext, config=deterministic_config)
        assert r.text  # produced something

    def test_different_prompt_full_prefill(self, llm: LLM, deterministic_config) -> None:
        """A prompt that shares no useful prefix should still produce
        output -- the no-overlap branch wipes seq 0 and starts over."""
        llm.reset_context()
        llm("The first prompt is here.", config=deterministic_config)
        r = llm("Completely different topic.", config=deterministic_config)
        assert r.text


class TestInvalidation:
    def test_reset_context_clears_shadow(self, llm: LLM, deterministic_config) -> None:
        llm("warm up the cache", config=deterministic_config)
        assert len(llm._kv_seq_tokens) > 0
        llm.reset_context()
        assert llm._kv_seq_tokens == []

    def test_close_clears_shadow(self, model_path: str, deterministic_config) -> None:
        instance = LLM(model_path, verbose=False)
        try:
            instance("warm up", config=deterministic_config)
            assert len(instance._kv_seq_tokens) > 0
            instance.close()
            assert instance._kv_seq_tokens == []
        finally:
            del instance
            gc.collect()

    def test_clear_loras_invalidates_shadow_when_context_exists(self, llm: LLM, deterministic_config) -> None:
        # No LoRAs are loaded, but ``clear_loras`` still routes through
        # ``_apply_loras_to`` when a live context exists, which drops
        # the shadow.
        llm("warm", config=deterministic_config)
        assert len(llm._kv_seq_tokens) > 0
        llm.clear_loras()
        assert llm._kv_seq_tokens == []
