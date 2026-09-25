"""Smaller llama.h wrappers: module functions, model and vocab queries, sampler
chain editing, Python samplers, control vectors, and KV-cache position ops."""

import copy
import gc
import math

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


def _decoded(model, tokens, n_seq_max=1):
    ctx = _ctx(model, n_seq_max)
    ctx.decode(cy.llama_batch_get_one(tokens, 0))
    return ctx


def _greedy():
    s = cy.LlamaSampler()
    s.add_greedy()
    return s


class TestModuleFunctions:
    def test_ftype_name(self, model):
        assert model.ftype == cy.LLAMA_FTYPE_MOSTLY_Q8_0
        assert cy.llama_ftype_name(model.ftype) == "Q8_0"
        assert "unknown" in cy.llama_ftype_name(9999)

    @pytest.mark.parametrize(
        "mode",
        [
            cy.LLAMA_LOAD_MODE_AUTO,
            cy.LLAMA_LOAD_MODE_NONE,
            cy.LLAMA_LOAD_MODE_MMAP,
            cy.LLAMA_LOAD_MODE_MLOCK,
            cy.LLAMA_LOAD_MODE_MMAP_MLOCK,
            cy.LLAMA_LOAD_MODE_DIRECT_IO,
        ],
    )
    def test_load_mode_round_trip(self, mode):
        assert cy.llama_load_mode_from_str(cy.llama_load_mode_name(mode)) == mode

    def test_load_mode_invalid(self):
        # llama.cpp aborts on an unknown enum value; the binding raises
        with pytest.raises(ValueError, match="unknown load mode"):
            cy.llama_load_mode_name(99)
        with pytest.raises(ValueError, match="unknown load mode"):
            cy.llama_load_mode_from_str("bogus")

    def test_meta_key_str(self):
        assert cy.llama_model_meta_key_str(cy.LLAMA_MODEL_META_KEY_SAMPLING_TOP_K) == "general.sampling.top_k"
        assert cy.llama_model_meta_key_str(999) is None

    def test_split_path_and_prefix(self):
        # split_no is 0-based
        path = cy.llama_split_path("/m/model", 1, 4)
        assert path == "/m/model-00002-of-00004.gguf"
        assert cy.llama_split_prefix(path, 1, 4) == "/m/model"
        assert cy.llama_split_prefix(path, 0, 4) is None

    def test_limits_and_system_info(self):
        assert cy.llama_max_parallel_sequences() >= 1
        assert cy.llama_max_tensor_buft_overrides() >= 1
        assert "CPU" in cy.llama_print_system_info()

    def test_get_log_callback(self):
        def cb(level, text):
            pass

        try:
            cy.set_log_callback(cb)
            assert cy.get_log_callback() is cb
            cy.disable_logging()
            assert cy.get_log_callback() is None
        finally:
            cy.set_log_callback(None)
        assert cy.get_log_callback() is None


class TestModelAndVocab:
    def test_model_queries(self, model):
        assert model.n_swa == 0
        assert model.is_diffusion is False
        assert model.cls_label(0) is None
        with pytest.raises(IndexError):
            model.cls_label(model.n_cls_out)

    def test_from_splits_single_file(self, model_path, model):
        m = cy.LlamaModel.from_splits([model_path], verbose=False)
        assert (m.n_params, m.desc) == (model.n_params, model.desc)
        assert m.path_model == model_path

    def test_from_splits_rejects_empty_and_bad_files(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            cy.LlamaModel.from_splits([])
        bad = tmp_path / "bad.gguf"
        bad.write_bytes(b"\0" * 64)
        with pytest.raises(ValueError):
            cy.LlamaModel.from_splits([str(bad)])

    def test_vocab_mask_and_suppress_tokens(self, model):
        vocab = model.get_vocab()
        assert vocab.token_mask() == cy.LLAMA_TOKEN_NULL
        assert vocab.get_suppress_tokens() == []


class TestChainEditing:
    def _chain(self):
        s = cy.LlamaSampler()
        s.add_top_k(40)
        s.add_temp(0.8)
        s.add_dist(7)
        return s

    def test_len_and_get(self):
        s = self._chain()
        assert len(s) == 3
        assert [s.chain_get(i).name() for i in range(3)] == ["top-k", "temp", "dist"]
        assert s.chain_get(-1).name() == "dist"
        with pytest.raises(IndexError):
            s.chain_get(3)

    def test_view_keeps_chain_alive(self):
        link = self._chain().chain_get(0)
        gc.collect()
        assert link.name() == "top-k"

    def test_remove_transfers_ownership(self, model, tokens):
        s = self._chain()
        removed = s.chain_remove(1)
        assert removed.name() == "temp"
        assert [s.chain_get(i).name() for i in range(len(s))] == ["top-k", "dist"]
        del s
        gc.collect()
        assert removed.name() == "temp"

    def test_link_is_not_a_chain(self):
        link = self._chain().chain_get(0)
        assert len(link) == 0
        with pytest.raises(ValueError, match="not a chain"):
            link.add_greedy()
        with pytest.raises(ValueError, match="not a chain"):
            link.chain_remove(0)

    def test_copy_state_from(self, model, tokens):
        ctx = _decoded(model, tokens)
        a = self._chain()
        a.sample(ctx, -1)
        a.sample(ctx, -1)  # advance the RNG
        b = self._chain()
        b.copy_state_from(a)
        assert [a.sample(ctx, -1) for _ in range(5)] == [b.sample(ctx, -1) for _ in range(5)]

    def test_copy_state_from_different_shape(self):
        with pytest.raises(ValueError, match="differ"):
            self._chain().copy_state_from(_greedy())

    def test_lazy_grammar_inactive_until_triggered(self, model, tokens):
        ctx = _decoded(model, tokens)
        s = cy.LlamaSampler()
        s.add_grammar_lazy_patterns(model.get_vocab(), 'root ::= "yes"', "root", trigger_patterns=["NEVER_SEEN"])
        s.add_greedy()
        assert len(s) == 2
        assert s.sample(ctx, -1) == _greedy().sample(ctx, -1)


class TestCustomSampler:
    def test_argmax_matches_greedy(self, model, tokens):
        class Argmax:
            name = "argmax"

            def apply(self, ids, logits, probs):
                return max(range(len(logits)), key=logits.__getitem__)

        ctx = _decoded(model, tokens)
        s = cy.LlamaSampler()
        s.add_custom(Argmax())
        assert s.chain_get(0).name() == "argmax"
        assert s.sample(ctx, -1) == _greedy().sample(ctx, -1)

    def test_logit_edits_are_applied(self, model, tokens):
        ctx = _decoded(model, tokens)
        banned = _greedy().sample(ctx, -1)

        class Ban:
            def apply(self, ids, logits, probs):
                for i, tok in enumerate(ids):
                    if tok == banned:
                        logits[i] = -math.inf

        s = cy.LlamaSampler()
        s.add_custom(Ban())
        s.add_greedy()
        assert s.chain_get(0).name() == "Ban"
        assert s.sample(ctx, -1) != banned

    def test_accept_reset_and_clone(self, model, tokens):
        class Recorder:
            def __init__(self):
                self.accepted, self.resets = [], 0

            def apply(self, ids, logits, probs):
                return 0

            def accept(self, token):
                self.accepted.append(token)

            def reset(self):
                self.resets += 1

        rec = Recorder()
        s = cy.LlamaSampler()
        s.add_custom(rec)
        tok = s.sample(_decoded(model, tokens), -1)
        assert rec.accepted == [tok]
        s.reset()
        assert rec.resets == 1
        # clone deep-copies the object when it has no clone()
        s.clone().reset()
        assert rec.resets == 1

    def test_exception_propagates(self, model, tokens):
        class Broken:
            def apply(self, ids, logits, probs):
                raise KeyError("broken sampler")

        s = cy.LlamaSampler()
        s.add_custom(Broken())
        with pytest.raises(KeyError, match="broken sampler"):
            s.sample(_decoded(model, tokens), -1)

    def test_bad_index_raises(self, model, tokens):
        class OutOfRange:
            def apply(self, ids, logits, probs):
                return len(ids)

        s = cy.LlamaSampler()
        s.add_custom(OutOfRange())
        with pytest.raises(ValueError, match="outside"):
            s.sample(_decoded(model, tokens), -1)

    def test_requires_apply(self):
        with pytest.raises(TypeError, match="apply"):
            cy.LlamaSampler().add_custom(object())

    def test_clone_failure_raises(self):
        class NoCopy:
            def apply(self, ids, logits, probs):
                return 0

            def clone(self):
                raise RuntimeError("no clone")

        s = cy.LlamaSampler()
        s.add_custom(NoCopy())
        with pytest.raises(RuntimeError, match="no clone"):
            s.clone()
        copy.deepcopy(NoCopy())  # the object itself is fine


class TestControlVector:
    def test_zero_vector_is_a_noop_and_nonzero_changes_logits(self, model, tokens):
        n = model.n_embd * model.n_layer

        def logits_with(data):
            ctx = _ctx(model)
            if data is not None:
                ctx.set_adapter_cvec(data, model.n_embd, 1, model.n_layer)
            ctx.decode(cy.llama_batch_get_one(tokens, 0))
            return ctx.get_logits_ith(-1)

        base = logits_with(None)
        assert logits_with([0.0] * n) == pytest.approx(base, abs=1e-4)
        assert logits_with([0.05] * n) != pytest.approx(base, abs=1e-2)

    def test_clear(self, model):
        ctx = _ctx(model)
        ctx.set_adapter_cvec([0.1] * (model.n_embd * model.n_layer), model.n_embd, 1, model.n_layer)
        ctx.set_adapter_cvec(None, model.n_embd, 1, model.n_layer)

    def test_wrong_n_embd_raises(self, model):
        with pytest.raises(ValueError, match="llama_set_adapter_cvec"):
            _ctx(model).set_adapter_cvec([0.0] * 10, 5, 1, 2)


class TestMemoryPositions:
    def test_can_shift_and_div(self, model, tokens):
        ctx = _decoded(model, tokens)
        assert ctx.memory_can_shift() is True
        ctx.memory_seq_div(0, 0, -1, 2)
        assert ctx.memory_seq_pos_max(0) == (len(tokens) - 1) // 2

    def test_div_rejects_zero(self, model, tokens):
        with pytest.raises(ValueError, match="d must be"):
            _decoded(model, tokens).memory_seq_div(0, 0, -1, 0)

    @pytest.mark.parametrize(
        "call",
        [
            lambda c: c.memory_seq_pos_max(300),
            lambda c: c.memory_seq_pos_min(-1),
            lambda c: c.memory_seq_keep(1),
            lambda c: c.memory_seq_add(5, 0, -1, 1),
            lambda c: c.memory_seq_div(5, 0, -1, 2),
            lambda c: c.memory_seq_cp(0, 9, 0, -1),
            lambda c: c.memory_seq_rm(-2, 0, -1),
        ],
        ids=["pos_max", "pos_min", "keep", "add", "div", "cp", "rm"],
    )
    def test_seq_id_out_of_range_raises(self, model, call):
        # each of these aborted the process via GGML_ASSERT before the guard
        with pytest.raises(IndexError, match="n_seq_max"):
            call(_ctx(model))

    def test_rm_all_sequences_allowed(self, model, tokens):
        ctx = _decoded(model, tokens)
        assert ctx.memory_seq_rm(-1, 0, -1)
        assert ctx.memory_seq_pos_max(0) < 0
