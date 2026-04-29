"""Speculative decoding using the public llama API (nanobind port of speculative.pxi).

Provides speculative decoding using a draft model to generate candidate tokens
that are verified by the target model, potentially providing 2-3x speedup.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from . import _llama_native as _N

if TYPE_CHECKING:
    from ._llama_native import LlamaContext


class SpeculativeParams:
    """Parameters for speculative decoding."""

    def __init__(
        self,
        n_max: int = 16,
        n_min: int = 0,
        p_split: float = 0.1,
        p_min: float = 0.75,
    ) -> None:
        self.n_max = n_max
        self.n_min = n_min
        self.p_split = p_split
        self.p_min = p_min

    def __repr__(self) -> str:
        return f"SpeculativeParams(n_max={self.n_max}, n_min={self.n_min}, p_split={self.p_split}, p_min={self.p_min})"


class Speculative:
    """Speculative decoding manager using the public llama API.

    Uses a draft model context to generate candidate tokens quickly, which are
    then verified by the target model.
    """

    def __init__(
        self,
        params: SpeculativeParams,
        ctx_target: "LlamaContext",
        ctx_draft: "LlamaContext | None" = None,
    ) -> None:
        if not self.is_compat(ctx_target):
            raise ValueError("Target context is not compatible for speculative decoding")
        if ctx_draft is None:
            raise RuntimeError("Failed to initialize speculative decoding: no draft context provided")

        self.ctx_tgt = ctx_target
        self.ctx_dft = ctx_draft
        self._draft_prompt: list[int] = []
        self._n_acc_drafts = 0
        self._n_acc_tokens = 0
        self._n_gen_drafts = 0
        self._n_gen_tokens = 0

        sparams = _N.LlamaSamplerChainParams()
        sparams.no_perf = True
        self.sampler = _N.LlamaSampler(sparams)
        self.sampler.add_top_k(10)
        self.sampler.add_dist(0)

    @staticmethod
    def is_compat(ctx_target: "LlamaContext") -> bool:
        """Check if the target context supports partial KV cache removal."""
        # Decode 2 dummy tokens, then attempt partial removal of position 1+.
        batch = _N.LlamaBatch(n_tokens=2, embd=0, n_seq_max=1, verbose=False)
        batch.add(0, 0, [0], False)
        batch.add(0, 1, [0], False)
        try:
            ctx_target.decode(batch)
        except Exception:
            ctx_target.kv_cache_clear(True)
            return False

        can_rm = ctx_target.memory_seq_rm(0, 1, -1)
        ctx_target.kv_cache_clear(True)
        ctx_target.synchronize()
        return bool(can_rm)

    def begin(self, prompt_tokens: list[int]) -> None:
        """Reset draft state for a new generation."""
        self._draft_prompt = []

    def draft(
        self,
        params: SpeculativeParams,
        prompt_tokens: list[int],
        last_token_id: int,
    ) -> list[int]:
        """Generate draft tokens using the draft model."""
        n_max = params.n_max
        n_ctx = self.ctx_dft.n_ctx - n_max
        if n_ctx <= 0:
            return []

        prompt = list(prompt_tokens)
        if len(prompt) > n_ctx:
            prompt = prompt[-n_ctx:]

        # KV cache reuse: count common prefix with the previous draft prompt.
        old_prompt = self._draft_prompt
        reuse_n = 0
        for a, b in zip(old_prompt, prompt):
            if a == b:
                reuse_n += 1
            else:
                break

        if reuse_n == 0:
            self.ctx_dft.kv_cache_clear(True)
        elif reuse_n < len(old_prompt):
            self.ctx_dft.memory_seq_rm(0, reuse_n, -1)

        # Encode new prompt tokens not in cache.
        i_start = reuse_n
        n_new = len(prompt) - i_start
        if n_new > 0:
            batch = _N.LlamaBatch(n_tokens=n_new, embd=0, n_seq_max=1, verbose=False)
            for i in range(n_new):
                batch.add(prompt[i_start + i], i_start + i, [0], i == n_new - 1)
            self.ctx_dft.decode(batch)

        self._draft_prompt = list(prompt)

        # Draft generation loop.
        n_past = len(prompt)
        self.sampler.reset()
        result: list[int] = []
        for i in range(n_max):
            sampled = self.sampler.sample(self.ctx_dft, -1)
            self.sampler.accept(sampled)
            result.append(int(sampled))

            if len(result) >= n_max:
                break

            batch = _N.LlamaBatch(n_tokens=1, embd=0, n_seq_max=1, verbose=False)
            batch.add(sampled, n_past + i, [0], True)
            self.ctx_dft.decode(batch)
            self._draft_prompt.append(int(sampled))

        if result:
            self._n_gen_drafts += 1
            self._n_gen_tokens += len(result)

        return result

    def accept(self, n_accepted: int) -> None:
        """Inform the speculative decoder that n_accepted tokens were accepted."""
        if n_accepted > 0:
            self._n_acc_drafts += 1
            self._n_acc_tokens += n_accepted

    def print_stats(self) -> None:
        acc_rate = 100.0 * self._n_acc_tokens / self._n_gen_tokens if self._n_gen_tokens > 0 else 0.0
        print(
            f"speculative: gen_drafts={self._n_gen_drafts}, "
            f"acc_drafts={self._n_acc_drafts}, "
            f"gen_tokens={self._n_gen_tokens}, "
            f"acc_tokens={self._n_acc_tokens}, "
            f"acc_rate={acc_rate:.1f}%",
            file=sys.stderr,
        )

    def __repr__(self) -> str:
        return f"Speculative(target={self.ctx_tgt})"
