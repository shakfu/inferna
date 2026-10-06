#!/usr/bin/env python3
"""
Tests for inferna.whisper module.

This module tests the Whisper wrapper functionality including model loading,
parameter configuration, and basic transcription operations.
Tests replicate what `make test-whisper` does but via the Python wrapper.
"""

import gc

import numpy as np
import wave
import struct
from pathlib import Path
import pytest

# Import the whisper module
from inferna.whisper import whisper_cpp as wh

JFK_PATH = Path(__file__).parent / "samples" / "jfk.wav"


@pytest.fixture
def whisper_model_path():
    """Fixture for whisper model path."""
    model_path = Path("models/ggml-base.en.bin")
    if not model_path.exists():
        pytest.skip(f"Whisper model not found at {model_path}")
    return str(model_path)


@pytest.fixture
def sample_audio_path():
    """Fixture for sample audio path."""
    audio_path = JFK_PATH
    if not audio_path.exists():
        pytest.skip(f"Sample audio not found at {audio_path}")
    return str(audio_path)


def load_wav_file(filepath):
    """Load a WAV file and return samples as float32 array."""
    with wave.open(filepath, "rb") as wav_file:
        frames = wav_file.readframes(-1)
        sound_info = wav_file.getparams()

        # Convert to float32
        if sound_info.sampwidth == 1:
            fmt = f"{len(frames)}B"
            samples = struct.unpack(fmt, frames)
            samples = [(s - 128) / 128.0 for s in samples]
        elif sound_info.sampwidth == 2:
            fmt = f"{len(frames) // 2}h"
            samples = struct.unpack(fmt, frames)
            samples = [s / 32768.0 for s in samples]
        else:
            raise ValueError(f"Unsupported sample width: {sound_info.sampwidth}")

        return np.array(samples, dtype=np.float32), sound_info.framerate


def test_whisper_constants():
    """Test whisper constants are accessible."""
    assert wh.WHISPER.SAMPLE_RATE == 16000
    assert wh.WHISPER.N_FFT > 0
    assert wh.WHISPER.HOP_LENGTH > 0
    assert wh.WHISPER.CHUNK_SIZE > 0


def test_whisper_sampling_strategy():
    """Test whisper sampling strategy constants."""
    assert hasattr(wh.WhisperSamplingStrategy, "GREEDY")
    assert hasattr(wh.WhisperSamplingStrategy, "BEAM_SEARCH")


def test_whisper_aheads_preset():
    """Test whisper attention heads presets."""
    assert hasattr(wh.WhisperAheadsPreset, "NONE")
    assert hasattr(wh.WhisperAheadsPreset, "BASE_EN")
    assert hasattr(wh.WhisperAheadsPreset, "BASE")


def test_whisper_version():
    """Test getting whisper version."""
    version = wh.version()
    assert isinstance(version, str)
    assert len(version) > 0


def test_whisper_system_info():
    """Test getting system info."""
    info = wh.print_system_info()
    assert isinstance(info, str)
    assert len(info) > 0
    assert "WHISPER" in info


def test_whisper_context_params():
    """Test whisper context parameters."""
    params = wh.WhisperContextParams()

    # Test default values and setters
    assert isinstance(params.use_gpu, bool)
    assert isinstance(params.flash_attn, bool)
    assert isinstance(params.gpu_device, int)
    assert isinstance(params.dtw_token_timestamps, bool)

    # Test setting values
    params.use_gpu = True
    assert params.use_gpu is True

    params.gpu_device = 1
    assert params.gpu_device == 1


def test_whisper_context_params_dtw_config():
    """DTW alignment-head config must be settable (needed for word timestamps)."""
    params = wh.WhisperContextParams()
    assert isinstance(params.dtw_aheads_preset, int)
    assert isinstance(params.dtw_n_top, int)
    assert isinstance(params.dtw_mem_size, int)

    params.dtw_token_timestamps = True
    params.dtw_aheads_preset = wh.WhisperAheadsPreset.TINY
    assert params.dtw_aheads_preset == int(wh.WhisperAheadsPreset.TINY)
    params.dtw_n_top = 4
    assert params.dtw_n_top == 4
    params.dtw_mem_size = 2 * 1024 * 1024
    assert params.dtw_mem_size == 2 * 1024 * 1024


def test_whisper_full_params_vad_params_roundtrip():
    """The VAD tuning struct can be attached to full params (was unreachable)."""
    full = wh.WhisperFullParams()
    vad = wh.WhisperVadParams()
    vad.threshold = 0.65
    vad.min_speech_duration_ms = 222
    vad.max_speech_duration_s = 12.0
    full.vad_params = vad

    got = full.vad_params
    assert abs(got.threshold - 0.65) < 1e-6
    assert got.min_speech_duration_ms == 222
    assert abs(got.max_speech_duration_s - 12.0) < 1e-6


def test_whisper_vad_params():
    """Test whisper VAD parameters."""
    params = wh.WhisperVadParams()

    # Test threshold parameter
    assert isinstance(params.threshold, float)

    params.threshold = 0.5
    assert abs(params.threshold - 0.5) < 1e-6


def test_whisper_full_params():
    """Test whisper full processing parameters."""
    params = wh.WhisperFullParams()

    # Test basic properties
    assert isinstance(params.strategy, int)
    assert isinstance(params.n_threads, int)
    assert isinstance(params.n_max_text_ctx, int)
    assert isinstance(params.offset_ms, int)
    assert isinstance(params.duration_ms, int)
    assert isinstance(params.translate, bool)
    assert isinstance(params.no_context, bool)
    assert isinstance(params.no_timestamps, bool)
    assert isinstance(params.single_segment, bool)
    assert isinstance(params.print_special, bool)
    assert isinstance(params.print_progress, bool)
    assert isinstance(params.print_realtime, bool)
    assert isinstance(params.print_timestamps, bool)
    assert isinstance(params.token_timestamps, bool)
    assert isinstance(params.temperature, float)

    # Test setters
    params.n_threads = 2
    assert params.n_threads == 2

    params.translate = True
    assert params.translate is True

    params.temperature = 0.8
    assert abs(params.temperature - 0.8) < 1e-6

    # Test language setting
    params.language = "en"
    assert params.language == "en"

    params.language = None
    assert params.language is None


def test_whisper_context_initialization(whisper_model_path):
    """Test whisper context initialization."""
    ctx = wh.WhisperContext(whisper_model_path)

    # Test that context was created successfully
    assert ctx is not None

    # Test version method
    version = ctx.version()
    assert isinstance(version, str)
    assert len(version) > 0

    # Test system info
    info = ctx.system_info()
    assert isinstance(info, str)
    assert len(info) > 0


def test_whisper_context_model_properties(whisper_model_path):
    """Test whisper context model properties."""
    ctx = wh.WhisperContext(whisper_model_path)

    # Test vocabulary size
    n_vocab = ctx.n_vocab()
    assert n_vocab > 0

    # Test context sizes
    n_text_ctx = ctx.n_text_ctx()
    assert n_text_ctx > 0

    n_audio_ctx = ctx.n_audio_ctx()
    assert n_audio_ctx > 0

    # Test multilingual capability
    is_multilingual = ctx.is_multilingual()
    assert isinstance(is_multilingual, bool)

    # Test model architecture details
    assert ctx.model_n_vocab() > 0
    assert ctx.model_n_audio_ctx() > 0
    assert ctx.model_n_audio_state() > 0
    assert ctx.model_n_audio_head() > 0
    assert ctx.model_n_audio_layer() > 0
    assert ctx.model_n_text_ctx() > 0
    assert ctx.model_n_text_state() > 0
    assert ctx.model_n_text_head() > 0
    assert ctx.model_n_text_layer() > 0
    assert ctx.model_n_mels() > 0

    # Test model type
    model_type = ctx.model_type()
    assert model_type >= 0

    model_type_str = ctx.model_type_readable()
    assert isinstance(model_type_str, str)
    assert len(model_type_str) > 0


def test_whisper_tokens(whisper_model_path):
    """Test whisper token operations."""
    ctx = wh.WhisperContext(whisper_model_path)

    # Test special tokens
    eot_token = ctx.token_eot()
    assert eot_token >= 0

    sot_token = ctx.token_sot()
    assert sot_token >= 0

    # Test token to string conversion
    token_str = ctx.token_to_str(eot_token)
    assert isinstance(token_str, str)

    # Test tokenization
    text = "hello world"
    tokens = ctx.tokenize(text)
    assert isinstance(tokens, list)
    assert len(tokens) > 0

    # Test token count
    token_count = ctx.token_count(text)
    assert token_count > 0
    assert token_count == len(tokens)


def test_whisper_language_operations(whisper_model_path):
    """Test whisper language operations."""
    ctx = wh.WhisperContext(whisper_model_path)

    # Test language operations
    lang_max_id = ctx.lang_max_id()
    assert lang_max_id > 0

    # Test English language
    en_id = ctx.lang_id("en")
    assert en_id >= 0

    lang_str = ctx.lang_str(en_id)
    assert lang_str == "en"

    lang_str_full = ctx.lang_str_full(en_id)
    assert isinstance(lang_str_full, str)
    assert len(lang_str_full) > 0


def test_whisper_module_functions():
    """Test module-level whisper functions."""
    # Test version
    version = wh.version()
    assert isinstance(version, str)
    assert len(version) > 0

    # Test system info
    info = wh.print_system_info()
    assert isinstance(info, str)
    assert len(info) > 0

    # Test language functions
    lang_max_id = wh.lang_max_id()
    assert lang_max_id > 0

    en_id = wh.lang_id("en")
    assert en_id >= 0

    lang_str = wh.lang_str(en_id)
    assert lang_str == "en"

    lang_str_full = wh.lang_str_full(en_id)
    assert isinstance(lang_str_full, str)
    assert len(lang_str_full) > 0


def test_whisper_context_invalid_model():
    """Test whisper context with invalid model path."""
    with pytest.raises(FileNotFoundError):
        wh.WhisperContext("/nonexistent/model.bin")


def test_whisper_audio_loading_basic(sample_audio_path):
    """Test basic audio file loading."""
    samples, sample_rate = load_wav_file(sample_audio_path)

    # Verify audio properties
    assert isinstance(samples, np.ndarray)
    assert samples.dtype == np.float32
    assert len(samples) > 0
    assert sample_rate > 0

    # Audio should be around 11 seconds (JFK sample)
    duration = len(samples) / sample_rate
    assert 10 < duration < 12  # JFK sample is about 11 seconds


def test_whisper_context_with_params(whisper_model_path):
    """Test whisper context creation with custom parameters."""
    params = wh.WhisperContextParams()
    params.use_gpu = True
    params.gpu_device = 0

    ctx = wh.WhisperContext(whisper_model_path, params)
    assert ctx is not None

    # Verify context works
    version = ctx.version()
    assert isinstance(version, str)


def test_whisper_full_params_comprehensive():
    """Test comprehensive whisper full parameters."""
    params = wh.WhisperFullParams()

    # Test all parameter types and ranges
    params.strategy = wh.WhisperSamplingStrategy.GREEDY
    assert params.strategy == wh.WhisperSamplingStrategy.GREEDY

    params.n_threads = 4
    assert params.n_threads == 4

    params.offset_ms = 1000
    assert params.offset_ms == 1000

    params.duration_ms = 5000
    assert params.duration_ms == 5000

    params.translate = True
    assert params.translate is True

    params.no_context = True
    assert params.no_context is True

    params.single_segment = True
    assert params.single_segment is True

    params.print_timestamps = True
    assert params.print_timestamps is True

    params.token_timestamps = True
    assert params.token_timestamps is True


def test_whisper_timing_functions(whisper_model_path):
    """Test whisper timing functions."""
    ctx = wh.WhisperContext(whisper_model_path)

    # reset_timings() and print_timings() must not raise and both must be
    # idempotent (call each twice to catch state corruption on second call).
    ctx.reset_timings()
    ctx.reset_timings()
    ctx.print_timings()
    ctx.print_timings()
    # Native context must still be queryable after the timing operations;
    # n_vocab calls through to the C API, so a corrupted ctx would crash.
    assert ctx.n_vocab() > 0


# Integration test that mimics the actual whisper CLI usage
# @pytest.mark.integration
def test_whisper_basic_transcription_setup(whisper_model_path, sample_audio_path):
    """Test basic whisper transcription setup (without actual transcription due to missing full() method)."""
    # Load audio
    samples, sample_rate = load_wav_file(sample_audio_path)

    # Verify sample rate matches Whisper's expectation
    if sample_rate != wh.WHISPER.SAMPLE_RATE:
        # In a real implementation, we'd resample here
        print(f"Warning: Sample rate {sample_rate} != {wh.WHISPER.SAMPLE_RATE}")

    # Create context
    ctx = wh.WhisperContext(whisper_model_path)

    # Create parameters
    params = wh.WhisperFullParams()
    params.language = "en"
    params.print_timestamps = True
    params.print_progress = True
    params.n_threads = 4

    # Verify context is ready
    assert ctx.n_vocab() > 50000  # Base model should have ~51k vocab
    assert ctx.model_type_readable() in ["base", "base.en"]

    # Test that we can get language info
    en_id = ctx.lang_id("en")
    assert en_id >= 0

    print(f"Model loaded: {ctx.model_type_readable()}")
    print(f"Vocabulary size: {ctx.n_vocab()}")
    print(f"Audio length: {len(samples) / sample_rate:.1f}s")
    print(f"Sample rate: {sample_rate}Hz")

    # Note: Actual transcription would require implementing the full() method
    # or finding an alternative approach in the whisper wrapper


def test_whisper_token_data():
    """Test whisper token data structure."""
    token_data = wh.WhisperTokenData()

    # Test that properties are accessible (they return default values)
    assert hasattr(token_data, "id")
    assert hasattr(token_data, "tid")
    assert hasattr(token_data, "p")
    assert hasattr(token_data, "plog")
    assert hasattr(token_data, "pt")
    assert hasattr(token_data, "ptsum")
    assert hasattr(token_data, "t0")
    assert hasattr(token_data, "t1")
    assert hasattr(token_data, "t_dtw")
    assert hasattr(token_data, "vlen")


def _params():
    p = wh.WhisperFullParams()
    p.language = "en"
    p.n_threads = 2
    return p


def _text(r):
    return "".join(r.full_get_segment_text(i) for i in range(r.full_n_segments()))


@pytest.fixture(params=["context", "state"])
def pipeline(request, whisper_model_path):
    """A WhisperContext or a WhisperState on one: both expose the pipeline."""
    ctx = wh.WhisperContext(whisper_model_path)
    state = wh.WhisperState(ctx) if request.param == "state" else None
    yield ctx, state or ctx
    del state, ctx
    gc.collect()


class TestPipeline:
    def test_greedy_decode_reproduces_transcript(self, pipeline, sample_audio_path):
        ctx, p = pipeline
        samples, _ = load_wav_file(sample_audio_path)
        p.pcm_to_mel(samples, 4)
        assert abs(p.n_len() - len(samples) / wh.WHISPER.HOP_LENGTH) <= 1  # 10 ms frames
        p.encode(0, 4)
        prompt = [ctx.token_sot(), ctx.token_not()]
        p.decode(prompt, 0, 4)
        out = []
        while len(out) < 8:
            logits = p.get_logits()
            assert logits.shape == (ctx.model_n_vocab(),)
            out.append(int(np.argmax(logits)))
            p.decode(out[-1:], len(prompt) + len(out) - 1, 4)
        assert "".join(ctx.token_to_str(t) for t in out).startswith(" And so my fellow Americans")

    def test_set_mel(self, pipeline):
        ctx, p = pipeline
        n_mels = ctx.model_n_mels()
        p.set_mel(np.zeros((n_mels, 300), dtype=np.float32))
        assert p.n_len() == 300
        p.encode()
        with pytest.raises(RuntimeError, match="set_mel failed"):
            p.set_mel(np.zeros((n_mels + 1, 300), dtype=np.float32))
        with pytest.raises(TypeError):
            p.set_mel(np.zeros(300, dtype=np.float32))

    def test_bad_arguments_raise(self, pipeline):
        ctx, p = pipeline
        n_ctx = ctx.n_text_ctx()
        sot = ctx.token_sot()
        with pytest.raises(ValueError, match="offset"):
            p.encode(-1)
        with pytest.raises(ValueError, match="empty"):
            p.decode([], 0)
        with pytest.raises(ValueError, match="n_text_ctx"):
            p.decode([sot], -1)
        with pytest.raises(ValueError, match="n_text_ctx"):
            p.decode([sot] * (n_ctx + 1), 0)
        with pytest.raises(ValueError, match="n_text_ctx"):
            p.decode([sot, sot], n_ctx - 1)
        with pytest.raises(ValueError, match="token id"):
            p.decode([ctx.model_n_vocab()], 0)
        with pytest.raises(ValueError, match="multilingual"):
            p.lang_auto_detect()

    @pytest.mark.parametrize(
        "read",
        [
            lambda p: p.full_n_segments(),
            lambda p: p.full_get_segment_text(0),
            lambda p: p.full_get_token_text(0, 0),
            lambda p: p.full_n_vad_segments(),
            lambda p: p.n_len(),
            lambda p: p.get_logits(),
        ],
    )
    def test_reads_raise_while_busy(self, pipeline, read):
        # full() clears and rebuilds the results on another thread without the GIL
        _, p = pipeline
        assert p._busy_lock.acquire(blocking=False)
        try:
            with pytest.raises(RuntimeError, match="another thread"):
                read(p)
        finally:
            p._busy_lock.release()

    def test_logits_need_a_decode(self, pipeline, sample_audio_path):
        ctx, p = pipeline
        with pytest.raises(RuntimeError, match="decode"):
            p.get_logits()
        samples, _ = load_wav_file(sample_audio_path)
        p.pcm_to_mel(samples)
        p.encode()
        p.decode([ctx.token_sot()], 0)
        assert p.get_logits().shape == (ctx.model_n_vocab(),)
        p.full(samples, _params())
        # full() rewrites the logits
        with pytest.raises(RuntimeError, match="decode"):
            p.get_logits()


class TestWhisperState:
    def test_full_matches_context(self, whisper_model_path, sample_audio_path):
        samples, _ = load_wav_file(sample_audio_path)
        ctx = wh.WhisperContext(whisper_model_path)
        ctx.full(samples, _params())
        state = wh.WhisperState(ctx)
        assert state.ctx is ctx
        state.full(samples, _params())
        assert _text(state) == _text(ctx)
        assert "fellow Americans" in _text(state)
        assert state.full_lang_id() == ctx.full_lang_id()
        assert state.full_get_token_id(0, 1) == ctx.full_get_token_id(0, 1)
        with pytest.raises(IndexError, match="outside"):
            state.full_get_segment_text(state.full_n_segments())
        del state, ctx
        gc.collect()

    def test_states_run_concurrently(self, whisper_model_path, sample_audio_path):
        import threading

        samples, _ = load_wav_file(sample_audio_path)
        ctx = wh.WhisperContext(whisper_model_path)
        states = [wh.WhisperState(ctx) for _ in range(2)]
        errors: list[Exception] = []

        def run(st):
            try:
                st.full(samples, _params())
            except Exception as e:  # noqa: BLE001 -- surfaced by the assert below
                errors.append(e)

        threads = [threading.Thread(target=run, args=(st,)) for st in states]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        assert not errors
        assert _text(states[0]) == _text(states[1])
        assert "fellow Americans" in _text(states[0])
        del states, ctx
        gc.collect()

    def test_concurrent_full_on_one_state_raises(self, whisper_model_path):
        ctx = wh.WhisperContext(whisper_model_path)
        state = wh.WhisperState(ctx)
        samples = np.zeros(wh.WHISPER.SAMPLE_RATE, dtype=np.float32)
        assert state._busy_lock.acquire(blocking=False)
        try:
            with pytest.raises(RuntimeError, match="another thread"):
                state.full(samples, _params())
        finally:
            state._busy_lock.release()
        del state, ctx
        gc.collect()

    def test_close_raises_while_busy(self, whisper_model_path):
        # close() during full() on another thread would free memory under it
        def assert_close_refused(obj):
            assert obj._busy_lock.acquire(blocking=False)
            try:
                with pytest.raises(RuntimeError, match="another thread"):
                    obj.close()
                assert obj.is_valid
            finally:
                obj._busy_lock.release()

        ctx = wh.WhisperContext(whisper_model_path)
        state = wh.WhisperState(ctx)
        assert_close_refused(state)
        state.close()
        assert_close_refused(ctx)
        ctx.close()
        assert not ctx.is_valid
        del state, ctx
        gc.collect()

    def test_context_close_waits_for_states(self, whisper_model_path):
        ctx = wh.WhisperContext(whisper_model_path)
        state = wh.WhisperState(ctx)
        other = wh.WhisperState(ctx)
        with pytest.raises(RuntimeError, match="2 open WhisperState"):
            ctx.close()
        state.close()
        assert not state.is_valid
        with pytest.raises(RuntimeError, match="closed"):
            state.full_n_segments()
        state.close()  # idempotent
        del other  # freed by refcount
        ctx.close()
        with pytest.raises(RuntimeError, match="closed"):
            wh.WhisperState(ctx)
        del state, ctx
        gc.collect()

    def test_full_parallel(self, whisper_model_path, sample_audio_path):
        samples, _ = load_wav_file(sample_audio_path)
        ctx = wh.WhisperContext(whisper_model_path)
        ctx.full_parallel(samples, _params(), 2)
        assert ctx.full_n_segments() >= 2  # one or more per chunk
        assert "Americans" in _text(ctx)
        with pytest.raises(ValueError, match="n_processors"):
            ctx.full_parallel(samples, _params(), 0)
        del ctx
        gc.collect()


# Test edge cases and error conditions
def test_whisper_tokenize_edge_cases(whisper_model_path):
    """Test whisper tokenization edge cases."""
    ctx = wh.WhisperContext(whisper_model_path)

    # Test empty string
    tokens = ctx.tokenize("")
    assert isinstance(tokens, list)

    # Test very long string (should handle gracefully)
    long_text = "hello " * 1000
    try:
        tokens = ctx.tokenize(long_text, max_tokens=2000)
        assert isinstance(tokens, list)
    except RuntimeError:
        # Expected if text is too long for max_tokens
        pass

    # Test unicode text
    unicode_text = "Hello 世界"
    tokens = ctx.tokenize(unicode_text)
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_whisper_invalid_language_id(whisper_model_path):
    """Test whisper with invalid language IDs."""
    ctx = wh.WhisperContext(whisper_model_path)

    # Test invalid language ID
    result = ctx.lang_str(9999)
    assert result is None

    result = ctx.lang_str_full(9999)
    assert result is None


def test_whisper_audio_processing_requirements():
    """Test that we understand whisper audio requirements."""
    # Test that we know the expected format
    assert wh.WHISPER.SAMPLE_RATE == 16000

    # Create a simple sine wave for testing
    duration = 1.0  # 1 second
    sample_rate = wh.WHISPER.SAMPLE_RATE
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    frequency = 440  # A note
    samples = np.sin(frequency * 2 * np.pi * t).astype(np.float32)

    # Test that we can create properly formatted audio
    assert samples.dtype == np.float32
    assert len(samples) == sample_rate
    assert -1.0 <= samples.min() <= samples.max() <= 1.0


def test_whisper_language_setter_fixed():
    """Test that the language setter bugs have been fixed."""
    params = wh.WhisperFullParams()

    # Test 1: Setting string values works correctly
    params.language = "en"
    assert params.language == "en"

    # Test 2: Setting None value works correctly
    params.language = None
    assert params.language is None

    # Test 3: Setting different string values
    params.language = "es"
    assert params.language == "es"

    params.language = "fr"
    assert params.language == "fr"

    # Test 4: Back to None
    params.language = None
    assert params.language is None

    # The bugs were fixed in whisper_cpp.pyx:
    # 1. Added _language_bytes member to keep bytes object alive
    # 2. Fixed None handling by checking value before encoding
    # 3. Use proper const char* casting from bytes object


class TestWhisperContextConcurrencyGuard:
    """Tests for the concurrent-use guard on WhisperContext.

    whisper_context is not thread-safe under concurrent native calls.
    The guard catches actual contention via a non-blocking lock
    acquired around `encode()` and `full()`. A second concurrent
    caller hits the guard and raises a clear RuntimeError before any
    native code runs.

    The test pattern matches TestSDContextConcurrencyGuard and
    TestLLMConcurrencyGuard: hold the busy-lock from the test thread
    to simulate "another thread is currently inside encode()/full()",
    then invoke the real public method on a worker thread and assert
    it raises RuntimeError. The worker never reaches native code
    because `_try_acquire_busy()` rejects it first.
    """

    def test_concurrent_encode_raises(self, whisper_model_path):
        """A worker thread calling encode() while the busy-lock is
        already held must raise RuntimeError without entering native
        code."""
        import threading

        ctx = wh.WhisperContext(whisper_model_path)

        assert ctx._busy_lock.acquire(blocking=False) is True
        try:
            errors: list[Exception] = []

            def worker():
                try:
                    ctx.encode(offset=0, n_threads=1)
                except RuntimeError as e:
                    errors.append(e)

            t = threading.Thread(target=worker)
            t.start()
            t.join(timeout=10)

            assert not t.is_alive(), "worker did not return — guard may be missing"
            assert len(errors) == 1, f"Expected concurrent encode() to raise, got: {errors}"
            msg = str(errors[0])
            assert "another thread" in msg
            assert "not thread-safe" in msg
        finally:
            ctx._busy_lock.release()

    def test_concurrent_full_raises(self, whisper_model_path):
        """A worker thread calling full() while the busy-lock is
        already held must raise RuntimeError without entering native
        code. Uses a tiny synthetic samples buffer because the worker
        never actually processes it."""
        import threading

        ctx = wh.WhisperContext(whisper_model_path)
        # 1 second of silence at 16kHz; the buffer is never consumed
        # because the worker raises before reaching whisper_full.
        samples = np.zeros(wh.WHISPER.SAMPLE_RATE, dtype=np.float32)

        assert ctx._busy_lock.acquire(blocking=False) is True
        try:
            errors: list[Exception] = []

            def worker():
                try:
                    ctx.full(samples)
                except RuntimeError as e:
                    errors.append(e)

            t = threading.Thread(target=worker)
            t.start()
            t.join(timeout=10)

            assert not t.is_alive(), "worker did not return — guard may be missing"
            assert len(errors) == 1, f"Expected concurrent full() to raise, got: {errors}"
            msg = str(errors[0])
            assert "another thread" in msg
            assert "not thread-safe" in msg
        finally:
            ctx._busy_lock.release()

    def test_lock_release_allows_subsequent_acquire(self, whisper_model_path):
        """Sanity check: after releasing the busy-lock,
        `_try_acquire_busy()` succeeds again so a normal call would
        proceed."""
        ctx = wh.WhisperContext(whisper_model_path)

        assert ctx._busy_lock.acquire(blocking=False) is True
        ctx._busy_lock.release()

        # Should not raise
        ctx._try_acquire_busy()
        ctx._busy_lock.release()


def test_package_reexports_public_api():
    """The docs import these from inferna.whisper directly."""
    import inferna.whisper as pkg

    for name in pkg.__all__:
        assert getattr(pkg, name) is getattr(wh, name)


VAD_MODEL = Path("models/ggml-silero-v5.1.2.bin")


def _transcribed(model_path, audio_path, vad=False):
    samples, _ = load_wav_file(audio_path)
    ctx = wh.WhisperContext(model_path)
    params = wh.WhisperFullParams()
    params.language = "en"
    params.n_threads = 4
    params.token_timestamps = True
    if vad:
        params.vad = True
        params.vad_model_path = str(VAD_MODEL)
    ctx.full(samples, params)
    return ctx


@pytest.fixture(scope="class")
def transcribed():
    if not (Path("models/ggml-base.en.bin").exists() and JFK_PATH.exists()):
        pytest.skip("whisper model or sample audio not found")
    ctx = _transcribed("models/ggml-base.en.bin", str(JFK_PATH))
    yield ctx
    del ctx
    gc.collect()


class TestResultGetters:
    def test_token_times_and_speaker_turn(self, transcribed):
        ctx = transcribed
        assert ctx.full_n_segments() > 0
        assert ctx.full_n_tokens(0) > 0
        # token 0 is a special token; times need token_timestamps
        t0, t1 = ctx.full_get_token_t0(0, 1), ctx.full_get_token_t1(0, 1)
        assert 0 <= t0 <= t1
        assert isinstance(ctx.full_get_segment_speaker_turn_next(0), bool)
        assert ctx.full_n_vad_segments() == 0
        timings = ctx.get_timings()
        assert set(timings) == {"sample_ms", "encode_ms", "decode_ms", "batchd_ms", "prompt_ms"}
        assert timings["encode_ms"] > 0

    @pytest.mark.parametrize(
        "call",
        [
            lambda c, n: c.full_get_segment_t0(n),
            lambda c, n: c.full_get_segment_t1(n),
            lambda c, n: c.full_get_segment_text(n),
            lambda c, n: c.full_get_segment_speaker_turn_next(n),
            lambda c, n: c.full_get_segment_no_speech_prob(n),
            lambda c, n: c.full_n_tokens(-1),
            lambda c, n: c.full_get_token_text(0, 9999),
            lambda c, n: c.full_get_token_id(0, -1),
            lambda c, n: c.full_get_token_data(n, 0),
            lambda c, n: c.full_get_token_p(0, 9999),
            lambda c, n: c.full_get_token_t0(0, 9999),
            lambda c, n: c.full_get_token_t1(n, 0),
            lambda c, n: c.full_get_vad_segment_t0(0),
            lambda c, n: c.full_get_vad_segment_t1(-1),
        ],
    )
    def test_out_of_range_index_raises(self, transcribed, call):
        # whisper.cpp indexes its result vectors unchecked
        with pytest.raises(IndexError, match="outside"):
            call(transcribed, transcribed.full_n_segments())

    @pytest.mark.skipif(not VAD_MODEL.exists(), reason=f"VAD model not found at {VAD_MODEL}")
    def test_vad_segments(self, whisper_model_path, sample_audio_path):
        ctx = _transcribed(whisper_model_path, sample_audio_path, vad=True)
        n = ctx.full_n_vad_segments()
        assert n > 0
        bounds = [(ctx.full_get_vad_segment_t0(i), ctx.full_get_vad_segment_t1(i)) for i in range(n)]
        assert all(0 <= a < b for a, b in bounds)
        assert all(b1 <= a2 for (_, b1), (a2, _) in zip(bounds, bounds[1:]))
        del ctx
        gc.collect()


# Any multilingual model enables the language-detection tests.
MULTILINGUAL_MODEL = next(
    (p for p in (Path(f"models/ggml-{n}.bin") for n in ("tiny", "base", "small")) if p.exists()), None
)


@pytest.mark.skipif(MULTILINGUAL_MODEL is None, reason="no multilingual model (models/ggml-tiny.bin)")
@pytest.mark.parametrize("on_state", [False, True], ids=["context", "state"])
def test_lang_auto_detect(on_state):
    samples, _ = load_wav_file(str(JFK_PATH))
    ctx = wh.WhisperContext(str(MULTILINGUAL_MODEL))
    p = wh.WhisperState(ctx) if on_state else ctx
    p.pcm_to_mel(samples, 4)
    lang_id, probs = p.lang_auto_detect(0, 4)
    assert wh.lang_str(lang_id) == "en"
    assert probs.shape == (wh.lang_max_id() + 1,)
    assert int(np.argmax(probs)) == lang_id
    assert probs.sum() == pytest.approx(1.0, abs=1e-3)
    with pytest.raises(RuntimeError, match="lang_auto_detect failed"):
        p.lang_auto_detect(p.n_len() * 10 + 10)  # past the end of the audio
    del p, ctx
    gc.collect()


@pytest.fixture
def vad():
    if not VAD_MODEL.exists():
        pytest.skip(f"VAD model not found at {VAD_MODEL}")
    v = wh.WhisperVadContext(str(VAD_MODEL))
    yield v
    del v
    gc.collect()


class TestVadContext:
    def test_segments(self, vad, sample_audio_path):
        samples, _ = load_wav_file(sample_audio_path)
        vad.detect_speech(samples)
        probs = vad.probs()
        assert probs.dtype == np.float32 and len(probs) > 0
        assert 0 <= probs.min() and probs.max() <= 1
        segments = vad.segments_from_probs()
        assert segments == vad.segments_from_samples(samples)
        assert len(segments) > 1
        end_cs = len(samples) / wh.WHISPER.SAMPLE_RATE * 100
        assert all(0 <= t0 < t1 <= end_cs for t0, t1 in segments)
        assert all(a[1] <= b[0] for a, b in zip(segments, segments[1:]))

    def test_matches_full_vad(self, vad, whisper_model_path, sample_audio_path):
        samples, _ = load_wav_file(sample_audio_path)
        ctx = _transcribed(whisper_model_path, sample_audio_path, vad=True)
        internal = [(ctx.full_get_vad_segment_t0(i), ctx.full_get_vad_segment_t1(i)) for i in range(ctx.full_n_vad_segments())]
        assert vad.segments_from_samples(samples) == internal
        del ctx
        gc.collect()

    def test_params_change_segments(self, vad, sample_audio_path):
        samples, _ = load_wav_file(sample_audio_path)
        params = wh.WhisperVadParams()
        params.threshold = 1.0
        assert vad.segments_from_samples(samples, params) == []

    def test_silence_and_empty(self, vad):
        assert vad.segments_from_samples(np.zeros(wh.WHISPER.SAMPLE_RATE, dtype=np.float32)) == []
        vad.detect_speech(np.zeros(0, dtype=np.float32))
        assert vad.probs().shape == (0,)
        assert vad.segments_from_probs() == []

    def test_no_reset_keeps_lstm_state(self, vad, sample_audio_path):
        samples, _ = load_wav_file(sample_audio_path)
        half = len(samples) // 2
        vad.detect_speech(samples[half:])
        fresh = vad.probs()
        vad.detect_speech(samples[:half])
        vad.detect_speech(samples[half:], reset=False)
        carried = vad.probs()
        assert carried.shape == fresh.shape
        assert np.abs(carried - fresh).max() > 0.1
        vad.reset_state()
        vad.detect_speech(samples[half:], reset=False)
        np.testing.assert_array_equal(vad.probs(), fresh)

    def test_rejects_non_vad_models(self, whisper_model_path, tmp_path):
        with pytest.raises(ValueError, match="not a VAD model"):
            wh.WhisperVadContext(whisper_model_path)
        gguf = tmp_path / "x.bin"
        gguf.write_bytes(b"GGUF" + b"\0" * 64)
        with pytest.raises(ValueError, match="does not look like a valid VAD model"):
            wh.WhisperVadContext(str(gguf))

    def test_close(self, vad):
        samples = np.zeros(wh.WHISPER.SAMPLE_RATE, dtype=np.float32)
        assert vad._busy_lock.acquire(blocking=False)
        try:
            with pytest.raises(RuntimeError, match="another thread"):
                vad.close()
            with pytest.raises(RuntimeError, match="another thread"):
                vad.detect_speech(samples)
        finally:
            vad._busy_lock.release()
        vad.close()
        assert not vad.is_valid
        with pytest.raises(RuntimeError, match="closed"):
            vad.probs()
        vad.close()  # idempotent
