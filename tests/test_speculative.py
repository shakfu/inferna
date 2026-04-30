"""
Tests for speculative decoding functionality.

This module tests the Cython wrappers for llama.cpp's speculative decoding API.
"""

import pytest
from inferna.llama.llama_cpp import (
    LlamaBatch,
    LlamaModel,
    LlamaContext,
    LlamaContextParams,
    LlamaModelParams,
    Speculative,
    SpeculativeParams,
)


class TestSpeculativeParams:
    """Tests for SpeculativeParams class."""

    def test_default_initialization(self):
        """Test default parameter initialization."""
        params = SpeculativeParams()
        assert params.n_max == 16
        assert params.n_min == 0
        assert abs(params.p_split - 0.1) < 0.001
        assert params.p_min == 0.75

    def test_custom_initialization(self):
        """Test custom parameter initialization."""
        params = SpeculativeParams(n_max=32, n_min=4, p_split=0.2, p_min=0.9)
        assert params.n_max == 32
        assert params.n_min == 4
        assert abs(params.p_split - 0.2) < 0.001
        assert abs(params.p_min - 0.9) < 0.001

    def test_property_setters(self):
        """Test property setters."""
        params = SpeculativeParams()

        params.n_max = 24
        assert params.n_max == 24

        params.n_min = 2
        assert params.n_min == 2

        params.p_split = 0.3
        assert abs(params.p_split - 0.3) < 0.001

        params.p_min = 0.85
        assert abs(params.p_min - 0.85) < 0.001

    def test_repr(self):
        """Test string representation."""
        params = SpeculativeParams(n_max=20, n_min=2, p_split=0.1, p_min=0.8)
        repr_str = repr(params)
        assert "SpeculativeParams" in repr_str
        assert "n_max=20" in repr_str
        assert "n_min=2" in repr_str
        assert "p_min=0.8" in repr_str

    def test_n_max_bounds(self):
        """Test n_max parameter with various values."""
        # Small values
        params = SpeculativeParams(n_max=1)
        assert params.n_max == 1

        # Large values
        params = SpeculativeParams(n_max=128)
        assert params.n_max == 128

    def test_p_min_bounds(self):
        """Test p_min parameter with various values."""
        # Minimum probability
        params = SpeculativeParams(p_min=0.0)
        assert params.p_min == 0.0

        # Maximum probability
        params = SpeculativeParams(p_min=1.0)
        assert params.p_min == 1.0

        # Mid-range
        params = SpeculativeParams(p_min=0.5)
        assert params.p_min == 0.5


class TestSpeculativeCompatibility:
    """Tests for speculative decoding compatibility checks."""

    @pytest.mark.slow
    def test_is_compat(self, model_path):
        """Test compatibility check with a model context."""
        # Load model
        model_params = LlamaModelParams()
        model_params.n_gpu_layers = 0  # CPU only for testing
        model = LlamaModel(model_path, model_params)

        # Create context
        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = 512
        ctx = LlamaContext(model, ctx_params)

        # Check compatibility (result depends on model, just ensure no crash)
        result = Speculative.is_compat(ctx)
        assert isinstance(result, bool)

    def test_is_compat_none_context(self):
        """Test compatibility check with None raises appropriate error."""
        # Skip this test as it can cause segfaults
        # The C API doesn't handle NULL pointers gracefully
        pytest.skip("Skipping None test to avoid segfaults in C API")

    @pytest.mark.slow
    def test_is_compat_preserves_kv_cache(self, model_path):
        """is_compat must not clobber a pre-existing KV cache when an empty
        sequence is available to probe with.

        Regression: the prior implementation called kv_cache_clear(True) after
        its probe decode, silently destroying any prefill the caller had built
        up on the target context.
        """
        model_params = LlamaModelParams()
        model_params.n_gpu_layers = 0
        model = LlamaModel(model_path, model_params)

        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = 512
        ctx_params.n_seq_max = 2  # leave seq 1 free for the probe
        ctx = LlamaContext(model, ctx_params)

        prompt_tokens = [1, 2, 3, 4, 5]
        batch = LlamaBatch(
            n_tokens=len(prompt_tokens), embd=0, n_seq_max=1, verbose=False
        )
        for i, tok in enumerate(prompt_tokens):
            batch.add(tok, i, [0], i == len(prompt_tokens) - 1)
        ctx.decode(batch)

        pos_max_before = ctx.memory_seq_pos_max(0)
        assert pos_max_before == len(prompt_tokens) - 1

        Speculative.is_compat(ctx)

        pos_max_after = ctx.memory_seq_pos_max(0)
        assert pos_max_after == pos_max_before, (
            f"is_compat clobbered seq 0: pos_max went "
            f"{pos_max_before} -> {pos_max_after}"
        )
        # The probe seq itself must be left empty after is_compat returns.
        assert ctx.memory_seq_pos_max(1) < 0

    @pytest.mark.slow
    def test_is_compat_raises_when_no_free_seq(self, model_path):
        """When every sequence is prefilled, is_compat cannot probe without
        clobbering caller data — it must raise rather than silently corrupt."""
        model_params = LlamaModelParams()
        model_params.n_gpu_layers = 0
        model = LlamaModel(model_path, model_params)

        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = 512  # default n_seq_max=1
        ctx = LlamaContext(model, ctx_params)

        prompt_tokens = [1, 2, 3, 4, 5]
        batch = LlamaBatch(
            n_tokens=len(prompt_tokens), embd=0, n_seq_max=1, verbose=False
        )
        for i, tok in enumerate(prompt_tokens):
            batch.add(tok, i, [0], i == len(prompt_tokens) - 1)
        ctx.decode(batch)

        pos_max_before = ctx.memory_seq_pos_max(0)
        with pytest.raises(RuntimeError, match="no empty sequence"):
            Speculative.is_compat(ctx)
        # KV must be intact even on the raise path.
        assert ctx.memory_seq_pos_max(0) == pos_max_before


class TestSpeculativeInitialization:
    """Tests for Speculative class initialization."""

    @pytest.mark.slow
    def test_initialization_requires_speculative_type(self, model_path):
        """Test that initialization fails without a speculative type configured.

        The upstream API requires a valid speculative type (draft model, ngram, etc.)
        to be configured. Default params (type=NONE) will fail initialization.
        """
        model_params = LlamaModelParams()
        model_params.n_gpu_layers = 0
        model = LlamaModel(model_path, model_params)

        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = 512
        ctx_target = LlamaContext(model, ctx_params)

        params = SpeculativeParams()
        with pytest.raises(RuntimeError, match="Failed to initialize"):
            Speculative(params, ctx_target)


class TestSpeculativeEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_params_negative_values(self):
        """Test behavior with negative parameter values."""
        # Cython int allows negative values, but they may not be meaningful
        params = SpeculativeParams(n_max=-1, n_min=-1, p_min=-0.5)
        assert params.n_max == -1
        assert params.n_min == -1
        assert params.p_min == -0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
