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

# upstream's draft chain is top-k(10); p_min is compared against its top candidate
_DRAFT_TOP_K = 10


class SpeculativeParams:
    """Parameters for speculative decoding.

    Attributes:
        n_max: Maximum number of tokens to draft.
        n_min: A shorter draft is discarded.
        p_min: Drafting stops when the top candidate's probability falls below this.
    """

    def __init__(self, n_max: int = 16, *, n_min: int = 0, p_min: float = 0.75) -> None:
        self.n_max = n_max
        self.n_min = n_min
        self.p_min = p_min

    def __repr__(self) -> str:
        return f"SpeculativeParams(n_max={self.n_max}, n_min={self.n_min}, p_min={self.p_min})"


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
        self.sampler.add_top_k(_DRAFT_TOP_K)
        self.sampler.add_greedy()

    @staticmethod
    def is_compat(ctx_target: "LlamaContext") -> bool:
        """Check if the target context supports partial KV cache removal.

        Probes by decoding two dummy tokens into a sequence id that is currently
        empty in the target context, then attempting partial removal. Using an
        empty seq id means a caller-prefilled seq 0 (the typical case) is left
        untouched.
        """
        # Find an empty sequence to probe with so we don't disturb caller data.
        # n_seq_max defaults to 1 in many configs; in that case the only seq
        # available is 0, and we can only probe non-destructively when it's
        # already empty.
        probe_seq = -1
        for sid in range(ctx_target.n_seq_max):
            if ctx_target.memory_seq_pos_max(sid) < 0:
                probe_seq = sid
                break
        if probe_seq < 0:
            raise RuntimeError(
                "Speculative.is_compat: target context has no empty sequence "
                f"to probe (n_seq_max={ctx_target.n_seq_max}, all in use). "
                "Call is_compat on a fresh context, or free a sequence with "
                "memory_seq_rm before constructing Speculative."
            )

        batch = _N.LlamaBatch(n_tokens=2, embd=0, n_seq_max=1, verbose=False)
        batch.add(0, 0, [probe_seq], False)
        batch.add(0, 1, [probe_seq], False)
        try:
            ctx_target.decode(batch)
        except Exception:
            ctx_target.memory_seq_rm(probe_seq, -1, -1)
            return False

        can_rm = ctx_target.memory_seq_rm(probe_seq, 1, -1)
        # Drop any remaining probe tokens; full removal of the probe seq is
        # always supported even when partial is not.
        ctx_target.memory_seq_rm(probe_seq, -1, -1)
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
        """Generate draft tokens using the draft model.

        Mirrors upstream's draft-model speculator: decode ``last_token_id``
        after the prompt, then extend greedily with the top candidate while
        its probability (softmax over the top 10 logits) is at least
        ``params.p_min``. A draft shorter than ``params.n_min`` is discarded.

        Args:
            params: SpeculativeParams instance.
            prompt_tokens: Tokens the target has processed, excluding ``last_token_id``.
            last_token_id: The target's newest sampled token.

        Returns:
            Draft token IDs predicted to follow ``last_token_id``.

        Raises:
            ValueError: ``last_token_id`` is not in the draft vocabulary.
        """
        n_max = params.n_max
        n_vocab = self.ctx_dft.model.n_vocab
        if not 0 <= last_token_id < n_vocab:
            raise ValueError(f"last_token_id {last_token_id} is outside the draft vocabulary [0, {n_vocab})")
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

        # Encode new prompt tokens not in cache; no logits needed.
        if len(prompt) > reuse_n:
            self._decode_draft(prompt[reuse_n:], reuse_n, False)

        # Seed with the target's newest token: its logits predict the first draft token.
        n_past = len(prompt)
        self._decode_draft([last_token_id], n_past, True)
        self._draft_prompt = prompt + [last_token_id]

        self.sampler.reset()
        result: list[int] = []
        for i in range(n_max):
            if params.p_min > 0.0 and _N._draft_top_p(self.ctx_dft, -1, _DRAFT_TOP_K) < params.p_min:
                break
            # sample() also accepts the token
            sampled = int(self.sampler.sample(self.ctx_dft, -1))
            result.append(sampled)
            if len(result) >= n_max:
                break
            self._decode_draft([sampled], n_past + 1 + i, True)
            self._draft_prompt.append(sampled)

        if len(result) < params.n_min:
            result = []
        if result:
            self._n_gen_drafts += 1
            self._n_gen_tokens += len(result)

        return result

    def _decode_draft(self, tokens: list[int], pos0: int, last_logits: bool) -> None:
        """Decode ``tokens`` on seq 0 of the draft context from position ``pos0``."""
        n = len(tokens)
        batch = _N.LlamaBatch(n_tokens=n, embd=0, n_seq_max=1, verbose=False)
        for i, tok in enumerate(tokens):
            batch.add(tok, pos0 + i, [0], last_logits and i == n - 1)
        self.ctx_dft.decode(batch)

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
