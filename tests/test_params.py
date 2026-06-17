import pytest
# pytest.skip(allow_module_level=True)

import platform

from pytest import approx

PLATFORM = platform.system()
# ARCH = platform.machine()

import inferna.llama.llama_cpp as cy


def test_default_model_params():
    params = cy.LlamaModelParams()
    # n_gpu_layers defaults to -1 (auto-detect: use all available GPU layers)
    assert params.n_gpu_layers == -1
    assert params.split_mode == 1  # LLAMA_SPLIT_MODE_LAYER = 1
    assert params.main_gpu == 0
    assert params.vocab_only == False
    assert params.use_mmap == True
    assert params.use_mlock == False
    assert params.check_tensors == False
    assert params.progress_callback is None
    assert params.tensor_split == []  # Default is empty (no custom split)


def test_model_params_tensor_split():
    """Test that tensor_split can be set and retrieved."""
    params = cy.LlamaModelParams()

    # Initially empty
    assert params.tensor_split == []

    # Set tensor_split
    params.tensor_split = [0.5, 0.5]
    result = params.tensor_split
    assert len(result) == cy.llama_max_devices()
    assert result[0] == approx(0.5, rel=1e-5)
    assert result[1] == approx(0.5, rel=1e-5)
    # Remaining should be zero-filled
    assert all(v == 0.0 for v in result[2:])

    # Set different values
    params.tensor_split = [1.0, 2.0, 1.0]
    result = params.tensor_split
    assert result[0] == approx(1.0, rel=1e-5)
    assert result[1] == approx(2.0, rel=1e-5)
    assert result[2] == approx(1.0, rel=1e-5)

    # Clear with None
    params.tensor_split = None
    assert params.tensor_split == []

    # Set again after clearing
    params.tensor_split = [0.3, 0.7]
    result = params.tensor_split
    assert result[0] == approx(0.3, rel=1e-5)
    assert result[1] == approx(0.7, rel=1e-5)

    # Clear with empty list
    params.tensor_split = []
    assert params.tensor_split == []


def test_model_params_tensor_split_validation():
    """Test tensor_split validation."""
    params = cy.LlamaModelParams()

    # Should raise if too many elements
    max_devices = cy.llama_max_devices()
    with pytest.raises(ValueError):
        params.tensor_split = [0.5] * (max_devices + 1)


def test_model_params_kv_overrides():
    """kv_overrides round-trips through the owned, empty-key-terminated array."""
    params = cy.LlamaModelParams()
    assert params.kv_overrides == []

    o_bool = cy.LlamaModelKvOverride()
    o_bool.tag = cy.LLAMA_KV_OVERRIDE_TYPE_BOOL
    o_bool.key = "general.some_flag"
    o_bool.val_bool = True

    o_int = cy.LlamaModelKvOverride()
    o_int.tag = cy.LLAMA_KV_OVERRIDE_TYPE_INT
    o_int.key = "general.some_int"
    o_int.val_i64 = 42

    o_str = cy.LlamaModelKvOverride()
    o_str.tag = cy.LLAMA_KV_OVERRIDE_TYPE_STR
    o_str.key = "general.some_str"
    o_str.val_str = "hello"

    params.kv_overrides = [o_bool, o_int, o_str]
    got = params.kv_overrides
    assert len(got) == 3  # terminator is not surfaced
    assert got[0].key == "general.some_flag"
    assert got[0].tag == cy.LLAMA_KV_OVERRIDE_TYPE_BOOL
    assert got[0].val_bool is True
    assert got[1].key == "general.some_int"
    assert got[1].val_i64 == 42
    assert got[2].key == "general.some_str"
    assert got[2].val_str == "hello"

    # Clearing
    params.kv_overrides = []
    assert params.kv_overrides == []


def test_model_params_kv_override_empty_key_rejected():
    params = cy.LlamaModelParams()
    bad = cy.LlamaModelKvOverride()  # key defaults to empty
    bad.tag = cy.LLAMA_KV_OVERRIDE_TYPE_INT
    bad.val_i64 = 1
    with pytest.raises(ValueError):
        params.kv_overrides = [bad]


def test_model_params_tensor_buft_overrides():
    """tensor_buft_overrides round-trips; buft resolves by backend name.

    CPU is always a registered buffer type, so this is platform-independent.
    """
    params = cy.LlamaModelParams()
    assert params.tensor_buft_overrides == []

    t = cy.LlamaModelTensorBuftOverride()
    t.pattern = "attn.*"
    t.buft = "CPU"
    assert t.buft == "CPU"  # resolved + readable

    params.tensor_buft_overrides = [t]
    got = params.tensor_buft_overrides
    assert len(got) == 1  # {nullptr,nullptr} terminator not surfaced
    assert got[0].pattern == "attn.*"
    assert got[0].buft == "CPU"

    params.tensor_buft_overrides = []
    assert params.tensor_buft_overrides == []


def test_tensor_buft_override_unknown_buft_rejected():
    t = cy.LlamaModelTensorBuftOverride()
    with pytest.raises(ValueError):
        t.buft = "NoSuchBackend"


def test_tensor_buft_override_requires_pattern_and_buft():
    """Wiring an override with only a pattern (no buft) is rejected."""
    params = cy.LlamaModelParams()
    t = cy.LlamaModelTensorBuftOverride()
    t.pattern = "attn.*"  # no buft set
    with pytest.raises(ValueError):
        params.tensor_buft_overrides = [t]


def test_model_params_progress_callback():
    """Test that progress callback can be set and retrieved."""
    params = cy.LlamaModelParams()

    # Initially None
    assert params.progress_callback is None

    # Set a callback
    progress_values = []

    def on_progress(progress: float) -> bool:
        progress_values.append(progress)
        return True  # continue loading

    params.progress_callback = on_progress
    assert params.progress_callback is on_progress

    # Set to None to disable
    params.progress_callback = None
    assert params.progress_callback is None


def test_model_params_progress_callback_with_model(model_path):
    """Test that progress callback is actually called during model loading."""
    progress_values = []

    def on_progress(progress: float) -> bool:
        progress_values.append(progress)
        return True  # continue loading

    params = cy.LlamaModelParams()
    params.progress_callback = on_progress

    # Load model - this should trigger progress callbacks
    model = cy.LlamaModel(model_path, params)

    # Verify callback was called with progress values
    assert len(progress_values) > 0, "Progress callback should have been called"
    assert all(0.0 <= p <= 1.0 for p in progress_values), "Progress values should be between 0 and 1"
    # The final progress should be 1.0 (complete)
    assert progress_values[-1] == 1.0, "Final progress should be 1.0"


def test_model_params_progress_callback_abort(model_path):
    """Test that returning False from progress callback aborts loading."""
    progress_values = []

    def on_progress(progress: float) -> bool:
        progress_values.append(progress)
        # Abort after first callback
        return False

    params = cy.LlamaModelParams()
    params.progress_callback = on_progress

    # Loading should fail due to abort
    with pytest.raises(Exception):
        model = cy.LlamaModel(model_path, params)

    # Callback should have been called at least once
    assert len(progress_values) >= 1, "Progress callback should have been called before abort"


def test_default_context_params():
    params = cy.LlamaContextParams()
    assert params.n_ctx == 512
    assert params.n_batch == 2048
    assert params.n_ubatch == 512
    assert params.n_seq_max == 1
    assert params.n_threads == cy.GGML_DEFAULT_N_THREADS
    assert params.n_threads_batch == cy.GGML_DEFAULT_N_THREADS
    assert params.rope_scaling_type == -1  # LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1
    assert params.pooling_type == -1  # LLAMA_POOLING_TYPE_UNSPECIFIED = -1
    assert params.attention_type == -1  # LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1
    assert params.rope_freq_base == 0.0
    assert params.rope_freq_scale == 0.0
    assert params.yarn_ext_factor == -1.0
    assert params.yarn_attn_factor == approx(-1.0)
    assert params.yarn_beta_fast == -1.0
    assert params.yarn_beta_slow == -1.0
    assert params.yarn_orig_ctx == 0
    assert params.type_k == 1  # GGML_TYPE_F16 = 1
    assert params.type_v == 1  # GGML_TYPE_F16 = 1
    assert params.offload_kqv == True
    assert params.no_perf == True


def test_default_model_quantize_params():
    params = cy.LlamaModelQuantizeParams()
    assert params.nthread == 0
    assert params.ftype == 7  # LLAMA_FTYPE_MOSTLY_Q8_0 = 7
    assert params.output_tensor_type == 42  # GGML_TYPE_COUNT = 42
    assert params.token_embedding_type == 42  # GGML_TYPE_COUNT = 42
    assert params.allow_requantize == False
    assert params.quantize_output_tensor == True
    assert params.only_copy == False
    assert params.pure == False
    assert params.keep_split == False


def test_quantize_params_imatrix_roundtrip():
    """Importance-matrix data round-trips through the owned, name-terminated array."""
    params = cy.LlamaModelQuantizeParams()
    assert params.imatrix == {}

    params.imatrix = {
        "blk.0.attn_q.weight": [0.5, 0.25, 0.125],
        "blk.0.ffn_down.weight": [1.0, 2.0],
    }
    got = params.imatrix
    assert set(got.keys()) == {"blk.0.attn_q.weight", "blk.0.ffn_down.weight"}
    assert list(got["blk.0.ffn_down.weight"]) == [1.0, 2.0]
    assert [approx(x) for x in got["blk.0.attn_q.weight"]] == [0.5, 0.25, 0.125]

    params.imatrix = None
    assert params.imatrix == {}


def test_quantize_function_is_exposed():
    """The quantize entry point must exist (and validate inputs)."""
    assert callable(cy.llama_model_quantize)
    with pytest.raises(Exception):
        # nonexistent input file -> native failure surfaced as an exception
        cy.llama_model_quantize(
            "/nonexistent/in.gguf", "/tmp/out.gguf", cy.LlamaModelQuantizeParams()
        )


def test_default_ggml_threadpool_params():
    params = cy.GgmlThreadPoolParams(n_threads=10)
    assert params.n_threads == 10
    assert params.prio == 0
    assert params.poll == 50
    assert params.strict_cpu == False
    assert params.paused == False
    assert params.cpumask == [False] * cy.GGML_MAX_N_THREADS
