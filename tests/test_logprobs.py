"""Tests for ``LLM(..., logprobs=, top_logprobs=)``.

Two layers:
  1. Pure-Python: arg-validation paths on ``__call__`` (negative
     ``top_logprobs``, mutual exclusion with ``stream=True``).
  2. Live: end-to-end against the standard test model. Verifies that
     ``Response.logprobs`` is populated, that the recorded logprobs are
     valid (non-positive, finite), that ``top_logprobs`` returns
     descending-logprob entries, and that the sampled token's logprob
     is bounded above by the top entry's logprob (the sampler may not
     pick the argmax once temperature/top-k/top-p kick in, so equality
     is not guaranteed).
"""

from __future__ import annotations

import gc
import math

import pytest

from inferna import LLM, GenerationConfig
from inferna.api import TokenLogprob, TopLogprob


# Per-token logprob capture is ~90ms over a 128k Llama vocab (pure
# Python). Capping every test's generation at 8 tokens is enough to
# verify correctness without paying for a 100-token reply.
_FAST_CFG = GenerationConfig(max_tokens=8)


@pytest.fixture(scope="module")
def llm(model_path: str):
    instance = LLM(model_path, verbose=False, config=_FAST_CFG)
    yield instance
    instance.close()
    del instance
    gc.collect()


class TestArgumentValidation:
    def test_negative_top_logprobs_raises(self, llm: LLM) -> None:
        with pytest.raises(ValueError, match="top_logprobs must be >= 0"):
            llm("hi", top_logprobs=-1)

    def test_logprobs_with_stream_raises(self, llm: LLM) -> None:
        with pytest.raises(NotImplementedError, match="logprobs=True"):
            llm("hi", logprobs=True, stream=True)

    def test_top_logprobs_implies_logprobs(self, llm: LLM) -> None:
        # top_logprobs > 0 should turn on the capture even if the caller
        # did not pass logprobs=True explicitly. Smoke-check by asking
        # for top-K and asserting the field is populated.
        r = llm("hi", top_logprobs=2)
        assert r.logprobs is not None


class TestLogprobsLive:
    def test_default_call_leaves_logprobs_none(self, llm: LLM) -> None:
        r = llm("Hi.")
        assert r.logprobs is None

    def test_logprobs_true_returns_one_per_token(self, llm: LLM) -> None:
        r = llm("Say one short word.", logprobs=True)
        assert r.logprobs is not None
        assert len(r.logprobs) > 0
        for lp in r.logprobs:
            assert isinstance(lp, TokenLogprob)
            assert isinstance(lp.token, str)
            assert isinstance(lp.token_id, int)
            assert math.isfinite(lp.logprob)
            # logprobs are non-positive (log-probability is <= 0).
            # Floating-point round-off near zero can land at a tiny
            # positive value; treat anything within 1e-4 of zero as OK.
            assert lp.logprob <= 1e-4
            # top_logprobs default is 0, so the list should be empty.
            assert lp.top_logprobs == []

    def test_top_logprobs_returns_descending(self, llm: LLM) -> None:
        r = llm("Hi.", top_logprobs=4)
        assert r.logprobs is not None
        for lp in r.logprobs:
            assert len(lp.top_logprobs) == 4
            for tl in lp.top_logprobs:
                assert isinstance(tl, TopLogprob)
                assert math.isfinite(tl.logprob)
            # Descending order
            lps = [tl.logprob for tl in lp.top_logprobs]
            assert lps == sorted(lps, reverse=True)

    def test_top_logprobs_first_entry_dominates_sampled(self, llm: LLM) -> None:
        # The top-K entries are pulled from the raw logits (argmax-
        # style), while the sampled token comes out of the temperature/
        # top-p chain. The top-1 raw entry's logprob is therefore an
        # upper bound on the sampled token's logprob (within a tiny
        # numerical-stability epsilon when they happen to coincide).
        r = llm("Hi.", top_logprobs=3)
        for lp in r.logprobs:
            assert lp.top_logprobs[0].logprob >= lp.logprob - 1e-4

    def test_top_logprobs_capped_at_vocab(self, llm: LLM) -> None:
        # Asking for more top-K entries than the vocab has should
        # silently cap at vocab size, not crash. Verified via the
        # capping-helper directly so we don't pay the cost of
        # materialising n_vocab TopLogprob objects per generated token.
        from inferna.api import TopLogprob

        # Drive a tiny generation just to populate the live context's
        # logits, then call the helper with an oversized k.
        llm("Hi.", logprobs=True, top_logprobs=1)
        oversize = llm.vocab.n_vocab + 5
        record = llm._build_token_logprob(llm._ctx, llm.vocab.token_bos(), oversize)
        assert len(record.top_logprobs) == llm.vocab.n_vocab
        assert all(isinstance(tl, TopLogprob) for tl in record.top_logprobs)

    def test_logprobs_bypasses_response_cache(self, llm_with_cache: LLM) -> None:
        # The cache stores Response objects. A cached entry from a
        # plain call would not carry the per-token logprobs the new
        # caller asked for, so the second call must miss the cache and
        # produce its own logprobs list.
        plain = llm_with_cache("Hi.")
        assert plain.logprobs is None
        with_lp = llm_with_cache("Hi.", logprobs=True)
        assert with_lp.logprobs is not None and len(with_lp.logprobs) > 0


@pytest.fixture(scope="module")
def llm_with_cache(model_path: str):
    instance = LLM(model_path, verbose=False, cache_size=4, config=_FAST_CFG)
    yield instance
    instance.close()
    del instance
    gc.collect()
