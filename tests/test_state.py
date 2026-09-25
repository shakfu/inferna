"""LlamaContext state save/restore: whole-context and per-sequence, in memory and on disk.

Each round trip is checked by behaviour, not only by size: a restored context
must predict the same next token as the one that produced the state.
"""

import pytest

import inferna.llama.llama_cpp as cy

PROMPT = "The capital of France is"


@pytest.fixture(scope="module")
def model(model_path):
    return cy.LlamaModel(model_path, verbose=False)


@pytest.fixture(scope="module")
def tokens(model):
    return model.get_vocab().tokenize(PROMPT, add_special=True, parse_special=False)


def _ctx(model, n_seq_max=1):
    p = cy.LlamaContextParams()
    p.n_ctx = 128
    p.n_seq_max = n_seq_max
    return cy.LlamaContext(model, p)


def _greedy(ctx, idx=-1):
    s = cy.LlamaSampler()
    s.add_greedy()
    return s.sample(ctx, idx)


def _next_after(ctx, token, pos, seq_id=0):
    """Decode ``token`` at ``pos`` on ``seq_id`` and return the greedy next token."""
    batch = cy.LlamaBatch(n_tokens=1, embd=0, n_seq_max=1, verbose=False)
    batch.add(token, pos, [seq_id], True)
    ctx.decode(batch)
    return _greedy(ctx)


@pytest.fixture
def decoded(model, tokens):
    ctx = _ctx(model, n_seq_max=2)
    ctx.decode(cy.llama_batch_get_one(tokens, 0))
    first = _greedy(ctx)
    return ctx, first


class TestWholeState:
    def test_round_trip_in_memory(self, model, tokens, decoded):
        ctx, first = decoded
        state = ctx.get_state_data()
        assert isinstance(state, bytes)
        assert len(state) == ctx.get_state_size()
        expected = _next_after(ctx, first, len(tokens))

        fresh = _ctx(model, n_seq_max=2)
        assert fresh.set_state_data(state) == len(state)
        assert fresh.memory_seq_pos_max(0) == len(tokens) - 1
        assert _next_after(fresh, first, len(tokens)) == expected

    def test_round_trip_file(self, model, tokens, decoded, tmp_path):
        ctx, first = decoded
        path = str(tmp_path / "session.bin")
        ctx.save_state_file(path, tokens)
        expected = _next_after(ctx, first, len(tokens))

        fresh = _ctx(model, n_seq_max=2)
        assert fresh.load_state_file(path) == tokens
        assert fresh.memory_seq_pos_max(0) == len(tokens) - 1
        assert _next_after(fresh, first, len(tokens)) == expected

    @pytest.mark.parametrize("bad", [b"", b"\0" * 16], ids=["empty", "zeros"])
    def test_bad_data_raises(self, decoded, bad):
        ctx, _ = decoded
        with pytest.raises(ValueError, match="Failed to load state"):
            ctx.set_state_data(bad)

    def test_truncated_data_raises(self, decoded):
        ctx, _ = decoded
        with pytest.raises(ValueError, match="Failed to load state"):
            ctx.set_state_data(ctx.get_state_data()[:100])

    def test_load_missing_file_raises(self, decoded, tmp_path):
        ctx, _ = decoded
        with pytest.raises(RuntimeError, match="llama_state_load_file failed"):
            ctx.load_state_file(str(tmp_path / "absent.bin"))

    def test_load_token_capacity_too_small_raises(self, decoded, tokens, tmp_path):
        ctx, _ = decoded
        path = str(tmp_path / "session.bin")
        ctx.save_state_file(path, tokens)
        with pytest.raises(RuntimeError, match="llama_state_load_file failed"):
            ctx.load_state_file(path, max_n_tokens=len(tokens) - 1)

    def test_closed_context_raises(self, model):
        ctx = _ctx(model)
        ctx.close()
        with pytest.raises(RuntimeError, match="closed"):
            ctx.get_state_data()


class TestSequenceState:
    def test_copy_sequence_in_memory(self, tokens, decoded):
        ctx, first = decoded
        data = ctx.get_state_seq_data(0)
        assert len(data) == ctx.get_state_seq_size(0)

        assert ctx.set_state_seq_data(data, 1) == len(data)
        assert ctx.memory_seq_pos_max(1) == len(tokens) - 1
        # seq 1 continues exactly as seq 0 would
        assert _next_after(ctx, first, len(tokens), seq_id=1) == _next_after(ctx, first, len(tokens), seq_id=0)

    def test_round_trip_file(self, model, tokens, decoded, tmp_path):
        ctx, first = decoded
        path = str(tmp_path / "seq.bin")
        assert ctx.save_state_seq_file(path, 0, tokens) > 0
        expected = _next_after(ctx, first, len(tokens))

        fresh = _ctx(model, n_seq_max=2)
        assert fresh.load_state_seq_file(path, 1) == tokens
        assert fresh.memory_seq_pos_max(1) == len(tokens) - 1
        assert fresh.memory_seq_pos_max(0) < 0
        assert _next_after(fresh, first, len(tokens), seq_id=1) == expected

    def test_bad_data_raises(self, decoded):
        ctx, _ = decoded
        with pytest.raises(ValueError, match="Failed to load sequence state"):
            ctx.set_state_seq_data(b"\0" * 8, 1)

    def test_load_missing_file_raises(self, decoded, tmp_path):
        ctx, _ = decoded
        with pytest.raises(RuntimeError, match="llama_state_seq_load_file failed"):
            ctx.load_state_seq_file(str(tmp_path / "absent.bin"), 0)

    def test_flags_are_exported(self):
        assert cy.LLAMA_STATE_SEQ_FLAGS_NONE == 0
        assert cy.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY == cy.LLAMA_STATE_SEQ_FLAGS_SWA_ONLY == 1
        assert cy.LLAMA_STATE_SEQ_FLAGS_ON_DEVICE == 2

    def test_flags_round_trip(self, tokens, decoded):
        ctx, _ = decoded
        flags = cy.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY
        data = ctx.get_state_seq_data(0, flags)
        assert len(data) == ctx.get_state_seq_size(0, flags)
        assert ctx.set_state_seq_data(data, 1, flags) == len(data)
