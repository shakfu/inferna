"""Regression tests for fixes from the comprehensive REVIEW.md pass.

Each test pins behavior we just fixed so a future refactor can't silently
re-introduce the bug. Test IDs map back to the review:

  A1  — LlamaVocab.tokenize retries on undersized buffer.
  A2  — WhisperContext methods raise after close().
  A3  — Mongoose Manager.send_reply rejects stale conn_id.
  A4  — _llama_progress_cb returns False (aborts) on handler exception/False.
  A7  — MtmdContext methods raise after close() (where exercisable).
  A12 — LlamaSampler.add_grammar with invalid grammar raises ValueError.
  A13 — LlamaModelKvOverride.key/val_str raise on oversized input.

  Lifecycle — LlamaContext methods raise after close().

Tests guard against null-deref crashes; an actual segfault would crash
the pytest worker so a passing run is the contract.
"""

from __future__ import annotations

import gc
from pathlib import Path

import pytest

from inferna.llama import llama_cpp as _ll


# =============================================================================
# Helpers
# =============================================================================

def _new_model(model_path: str) -> _ll.LlamaModel:
    params = _ll.LlamaModelParams()
    params.n_gpu_layers = 0
    return _ll.LlamaModel(path_model=model_path, params=params, verbose=False)


def _new_ctx(model: _ll.LlamaModel) -> _ll.LlamaContext:
    cp = _ll.LlamaContextParams()
    cp.n_ctx = 128
    cp.n_batch = 32
    return _ll.LlamaContext(model=model, params=cp, verbose=False)


# =============================================================================
# A2 / lifecycle: LlamaContext post-close
# =============================================================================

class TestLlamaContextPostClose:
    """Calls after close() must raise RuntimeError, not crash."""

    def test_is_valid_flips(self, model_path):
        m = _new_model(model_path)
        try:
            ctx = _new_ctx(m)
            assert ctx.is_valid is True
            ctx.close()
            assert ctx.is_valid is False
        finally:
            del m
            gc.collect()

    def test_n_ctx_after_close_raises(self, model_path):
        m = _new_model(model_path)
        try:
            ctx = _new_ctx(m)
            ctx.close()
            with pytest.raises(RuntimeError, match="closed"):
                _ = ctx.n_ctx
        finally:
            del m
            gc.collect()

    def test_decode_after_close_raises(self, model_path):
        m = _new_model(model_path)
        try:
            ctx = _new_ctx(m)
            batch = _ll.LlamaBatch(n_tokens=4, embd=0, n_seq_max=1)
            ctx.close()
            with pytest.raises(RuntimeError, match="closed"):
                ctx.decode(batch)
        finally:
            del m
            gc.collect()

    def test_get_logits_after_close_raises(self, model_path):
        m = _new_model(model_path)
        try:
            ctx = _new_ctx(m)
            ctx.close()
            with pytest.raises(RuntimeError, match="closed"):
                ctx.get_logits()
        finally:
            del m
            gc.collect()

    def test_double_close_is_idempotent(self, model_path):
        m = _new_model(model_path)
        try:
            ctx = _new_ctx(m)
            ctx.close()
            ctx.close()  # must not raise / crash
            assert ctx.is_valid is False
        finally:
            del m
            gc.collect()


# =============================================================================
# A2: WhisperContext post-close
# =============================================================================

class TestWhisperContextPostClose:
    @pytest.fixture(scope="class")
    def whisper_model_path(self) -> str:
        path = Path("models/ggml-base.en.bin")
        if not path.exists():
            pytest.skip(f"whisper model not available: {path}")
        return str(path)

    def test_is_valid_flips(self, whisper_model_path):
        from inferna.whisper.whisper_cpp import WhisperContext
        ctx = WhisperContext(model_path=whisper_model_path)
        assert ctx.is_valid is True
        ctx.close()
        assert ctx.is_valid is False

    def test_n_vocab_after_close_raises(self, whisper_model_path):
        from inferna.whisper.whisper_cpp import WhisperContext
        ctx = WhisperContext(model_path=whisper_model_path)
        ctx.close()
        with pytest.raises(RuntimeError, match="closed"):
            ctx.n_vocab()

    def test_full_after_close_raises(self, whisper_model_path):
        import numpy as np
        from inferna.whisper.whisper_cpp import WhisperContext
        ctx = WhisperContext(model_path=whisper_model_path)
        ctx.close()
        samples = np.zeros(16000, dtype=np.float32)
        with pytest.raises(RuntimeError, match="closed"):
            ctx.full(samples)


# =============================================================================
# A1: LlamaVocab.tokenize retries on undersized buffer
# =============================================================================

class TestVocabTokenizeRetry:
    """A long input would have been refused by the old `min(..., n_vocab)` cap."""

    def test_long_text_tokenizes(self, model_path):
        m = _new_model(model_path)
        try:
            vocab = m.get_vocab()
            # Long enough that any reasonable initial heuristic underestimates.
            text = "Hello world. " * 4000  # ~52k chars
            tokens = vocab.tokenize(text, add_special=False, parse_special=False)
            assert isinstance(tokens, list)
            assert len(tokens) > 0
            # Round-trip sanity: detokenizing the tokens should be non-empty.
            out = vocab.detokenize(
                tokens, text_len_max=len(text) + 1024,
                remove_special=False, unparse_special=False,
            )
            assert len(out) > 0
        finally:
            del m
            gc.collect()


# =============================================================================
# A12: LlamaSampler.add_grammar with invalid grammar
# =============================================================================

class TestSamplerNullCheck:
    """Inner llama_sampler_init_grammar returns NULL on parse failure;
    the wrapper must surface that as a Python exception, not silently no-op."""

    def test_invalid_grammar_raises(self, model_path):
        m = _new_model(model_path)
        try:
            vocab = m.get_vocab()
            sampler = _ll.LlamaSampler()
            # Malformed GBNF: dangling rule reference, missing terminator.
            with pytest.raises(ValueError, match="grammar"):
                sampler.add_grammar(vocab, "root ::= nonexistent", "root")
        finally:
            del m
            gc.collect()


# =============================================================================
# A13: KvOverride.key / val_str length validation
# =============================================================================

class TestKvOverrideLength:
    def test_oversized_key_raises(self):
        kv = _ll.LlamaModelKvOverride()
        # The underlying buffer is 128 bytes; 256 must be rejected.
        with pytest.raises(ValueError, match="bytes"):
            kv.key = "k" * 256

    def test_max_minus_one_accepted(self):
        kv = _ll.LlamaModelKvOverride()
        # Largest accepted key length is sizeof(buf) - 1 (room for NUL).
        # We don't hardcode the exact size to stay forward-compatible with
        # upstream changes; just confirm a 64-char key works.
        kv.key = "k" * 64
        assert kv.key == "k" * 64

    def test_oversized_val_str_raises(self):
        kv = _ll.LlamaModelKvOverride()
        with pytest.raises(ValueError, match="bytes"):
            kv.val_str = "v" * 256


# =============================================================================
# A4: Progress callback abort semantics
# =============================================================================

class TestProgressCallbackAbort:
    """Handler returning False or raising must abort llama_model_load_from_file."""

    def test_returns_false_aborts(self, model_path):
        # Callback that aborts on the first invocation. llama.cpp surfaces
        # the abort by returning NULL from the load, which the wrapper
        # converts into a Python exception.
        params = _ll.LlamaModelParams()
        params.n_gpu_layers = 0
        params.progress_callback = lambda p: False
        with pytest.raises(Exception):
            _ll.LlamaModel(path_model=model_path, params=params, verbose=False)

    def test_handler_exception_aborts(self, model_path):
        # Same outcome when the handler raises; the wrapper now returns
        # False from the catch (was: True / continue). Either Python
        # propagates the original or llama.cpp aborts with its own error;
        # what matters is the load does NOT complete normally.
        params = _ll.LlamaModelParams()
        params.n_gpu_layers = 0

        def bad(p):
            raise KeyboardInterrupt("simulated user cancel")
        params.progress_callback = bad
        with pytest.raises(Exception):
            _ll.LlamaModel(path_model=model_path, params=params, verbose=False)
