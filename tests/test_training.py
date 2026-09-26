"""LlamaModel.from_metadata, LlamaContext.opt_init / opt_epoch, and save_to_file.

Training needs float weights, and the repo's test models are quantized and too
large to train on CPU. These tests therefore build a 2-layer F32 llama from
GGUF metadata with llama.cpp's "test" tokenizer, as upstream's
tests/test-llama-archs.cpp does.
"""

import array
import random

import pytest

import inferna.llama.llama_cpp as cy

N_VOCAB, N_EMBD, N_HEAD, N_FF, N_LAYER = 128, 64, 2, 128, 2
PERIOD = 16  # training data repeats every PERIOD tokens


def tiny_metadata():
    g = cy.GGUFContext.empty()
    g.set_val_str("general.architecture", "llama")
    g.set_val_u32("llama.vocab_size", N_VOCAB)
    g.set_val_u32("llama.context_length", 256)
    g.set_val_u32("llama.embedding_length", N_EMBD)
    g.set_val_u32("llama.feed_forward_length", N_FF)
    g.set_val_u32("llama.block_count", N_LAYER)
    g.set_val_u32("llama.attention.head_count", N_HEAD)
    g.set_val_u32("llama.attention.head_count_kv", N_HEAD)
    g.set_val_f32("llama.attention.layer_norm_rms_epsilon", 1e-5)
    g.set_val_str("tokenizer.ggml.model", "test")
    g.set_arr_str("tokenizer.ggml.tokens", [f"tok_{i}" for i in range(N_VOCAB)])
    return g


def init_tensor(name, shape):
    n = 1
    for d in shape:
        n *= d
    if "norm" in name:
        return array.array("f", [1.0]) * n
    rng = random.Random(name)
    return array.array("f", (rng.gauss(0.0, 0.02) for _ in range(n)))


def tiny_model(n_gpu_layers=None):
    p = cy.LlamaModelParams()
    if n_gpu_layers is not None:
        p.n_gpu_layers = n_gpu_layers
    return cy.LlamaModel.from_metadata(tiny_metadata(), init_tensor, p, verbose=False)


def train_ctx(model, **overrides):
    p = cy.LlamaContextParams()
    p.n_ctx = 256
    p.n_batch = 256
    p.n_ubatch = 256
    p.type_k = cy.GGML_TYPE_F32
    p.type_v = cy.GGML_TYPE_F32
    for k, v in overrides.items():
        setattr(p, k, v)
    return cy.LlamaContext(model, p)


def data(n_ctx, windows=6):
    return [i % PERIOD for i in range(n_ctx * windows)]


def gpu_present():
    return any(d["type"] in ("GPU", "iGPU") for d in cy.ggml_backend_dev_info())


def trainable(**opt):
    """(model, ctx) after opt_init; SGD on CPU if the default device cannot run the step."""
    m = tiny_model()
    ctx = train_ctx(m)
    try:
        ctx.opt_init(**opt)
        return m, ctx
    except ValueError as e:
        if "no device can run" not in str(e):
            raise
    # e.g. paravirtual Metal; AdamW cannot fall back to CPU while a GPU is present
    m = tiny_model(n_gpu_layers=0)
    ctx = train_ctx(m)
    ctx.opt_init(**{**opt, "optimizer": "sgd", "learning_rate": 1e-2})
    return m, ctx


def last_logits(model, tokens):
    ctx = train_ctx(model)
    ctx.decode(cy.llama_batch_get_one(tokens, 0))
    return ctx.get_logits_ith(-1)


class TestFromMetadata:
    def test_builds_a_usable_model(self):
        m = tiny_model()
        assert (m.n_vocab, m.n_layer, m.n_embd) == (N_VOCAB, N_LAYER, N_EMBD)
        assert m.get_vocab().vocab_type == cy.LLAMA_VOCAB_TYPE_TEST
        assert len(last_logits(m, [1, 2, 3])) == N_VOCAB

    def test_init_tensor_sees_every_tensor_and_sets_values(self):
        seen = {}

        def record(name, shape):
            seen[name] = shape
            return init_tensor(name, shape)

        m = cy.LlamaModel.from_metadata(tiny_metadata(), record, verbose=False)
        assert seen["token_embd.weight"] == (N_EMBD, N_VOCAB)
        assert "blk.1.ffn_down.weight" in seen
        # the same initialiser gives the same model
        assert last_logits(m, [1, 2, 3]) == last_logits(tiny_model(), [1, 2, 3])

    def test_wrong_size_raises(self):
        with pytest.raises(ValueError, match="float32 values"):
            cy.LlamaModel.from_metadata(tiny_metadata(), lambda name, shape: array.array("f", [0.0]))

    def test_callback_exception_propagates(self):
        def boom(name, shape):
            raise KeyError(name)

        with pytest.raises(KeyError):
            cy.LlamaModel.from_metadata(tiny_metadata(), boom)


class TestTraining:
    def test_loss_falls_and_save_round_trips(self, tmp_path):
        m, ctx = trainable(learning_rate=1e-3)
        tokens = data(ctx.n_ctx)
        first = ctx.opt_epoch(tokens, val_split=0.2)
        for _ in range(3):
            last = ctx.opt_epoch(tokens, val_split=0.2)
        assert first["train"]["n_tokens"] > 0 and first["eval"]["n_tokens"] > 0
        assert last["eval"]["loss"] < 0.7 * first["eval"]["loss"]
        assert last["eval"]["accuracy"] > first["eval"]["accuracy"]

        path = str(tmp_path / "trained.gguf")
        m.save_to_file(path)
        reloaded = cy.LlamaModel(path, verbose=False)
        probe = [3, 4, 5, 6]
        assert last_logits(reloaded, probe) == pytest.approx(last_logits(m, probe), abs=1e-4)

    def test_param_filter_limits_training(self):
        names = []

        def only_output_norm(name):
            names.append(name)
            return name == "output_norm.weight"

        m, ctx = trainable(learning_rate=1e-2, param_filter=only_output_norm)
        before = last_logits(m, [1, 2, 3])
        ctx.opt_epoch(data(ctx.n_ctx), val_split=0.2)
        assert "output_norm.weight" in names
        assert last_logits(m, [1, 2, 3]) != before

    def test_param_filter_exception_propagates(self):
        ctx = train_ctx(tiny_model())

        def boom(name):
            raise RuntimeError("filter failed")

        with pytest.raises(RuntimeError, match="filter failed"):
            ctx.opt_init(param_filter=boom)
        with pytest.raises(RuntimeError, match="opt_init failed"):
            ctx.opt_epoch(data(ctx.n_ctx))


class TestTrainingGuards:
    """Each case aborts the process inside llama.cpp without the wrapper's check."""

    def test_opt_init_twice(self):
        _, ctx = trainable()
        with pytest.raises(RuntimeError, match="already"):
            ctx.opt_init()

    def test_f16_kv_cache_rejected(self):
        ctx = train_ctx(tiny_model(), type_k=cy.GGML_TYPE_F16)
        with pytest.raises(ValueError, match="F32 KV cache"):
            ctx.opt_init()

    def test_batch_divisibility(self):
        ctx = train_ctx(tiny_model(), n_batch=256, n_ubatch=96)
        with pytest.raises(ValueError, match="multiple"):
            ctx.opt_init()

    def test_unknown_optimizer(self):
        with pytest.raises(ValueError, match="adamw"):
            train_ctx(tiny_model()).opt_init(optimizer="lion")

    def test_quantized_model_rejected(self, model_path):
        m = cy.LlamaModel(model_path, verbose=False)
        p = cy.LlamaContextParams()
        p.n_ctx = 64
        p.type_k = p.type_v = cy.GGML_TYPE_F32
        ctx = cy.LlamaContext(m, p)
        with pytest.raises(ValueError, match="F32 model"):
            ctx.opt_init()

    @pytest.mark.parametrize("optimizer", ["adamw", "sgd"])
    def test_optimizer_step_device(self, optimizer):
        # Metal without simdgroup reduction (e.g. paravirtual macOS VMs) cannot
        # run the step on its weight buffers; there opt_init must raise.
        ctx = train_ctx(tiny_model())
        try:
            ctx.opt_init(optimizer=optimizer)
        except ValueError as e:
            assert "no device can run" in str(e)
            with pytest.raises(RuntimeError, match="opt_init failed"):
                ctx.opt_epoch(data(ctx.n_ctx))
            return
        ctx.opt_epoch(data(ctx.n_ctx), val_split=0.2)

    @pytest.mark.parametrize("n_gpu_layers", [0, 1])
    def test_adamw_rejects_weights_off_first_device(self, n_gpu_layers):
        # AdamW's m/v live on the first GPU; a step elsewhere silently drops momentum
        if not gpu_present():
            pytest.skip("needs a GPU device")
        ctx = train_ctx(tiny_model(n_gpu_layers=n_gpu_layers))
        with pytest.raises(ValueError, match="first device|no device can run"):
            ctx.opt_init()

    def test_sgd_on_cpu_matches_default_placement(self):
        def losses(model):
            ctx = train_ctx(model)
            ctx.opt_init(optimizer="sgd", learning_rate=1e-2)
            return [ctx.opt_epoch(data(ctx.n_ctx), val_split=0.2)["eval"]["loss"] for _ in range(2)]

        cpu = losses(tiny_model(n_gpu_layers=0))
        try:
            ref = losses(tiny_model())
        except ValueError as e:
            assert "no device can run" in str(e)
            assert cpu[1] < cpu[0]
            return
        assert cpu == pytest.approx(ref, rel=1e-4)

    def test_epoch_before_init(self):
        with pytest.raises(RuntimeError, match="opt_init first"):
            train_ctx(tiny_model()).opt_epoch([0] * 1000)

    def test_too_few_tokens(self):
        _, ctx = trainable()
        with pytest.raises(ValueError, match="at least"):
            ctx.opt_epoch([1] * ctx.n_ctx)
        # one window cannot be split into train and eval
        with pytest.raises(ValueError, match="none for training"):
            ctx.opt_epoch(data(ctx.n_ctx, windows=2), val_split=0.5)

    def test_token_outside_vocab(self):
        _, ctx = trainable()
        with pytest.raises(ValueError, match="outside the vocabulary"):
            ctx.opt_epoch(data(ctx.n_ctx)[:-1] + [N_VOCAB])

    @pytest.mark.parametrize("val_split", [-0.1, 1.0])
    def test_val_split_range(self, val_split):
        _, ctx = trainable()
        with pytest.raises(ValueError, match="val_split"):
            ctx.opt_epoch(data(ctx.n_ctx), val_split=val_split)
