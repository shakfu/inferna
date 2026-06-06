"""Release gate: whisper GPU-backend transcription correctness/parity.

This is a regression + release-gate test for a *silent* GPU miscomputation
class. Background:

  inferna links the whisper extension against ggml. llama.cpp and whisper.cpp
  each vendor their own ggml snapshot (both stamped 0.13.1, but different
  upstream commits). When `_whisper_native` was linked against *llama's*
  ggml-metal, whisper's encoder produced ZERO segments on the GPU (Metal)
  backend while the CPU backend was correct -- `whisper_full` returned rc=0
  with no error, so nothing downstream noticed.

  Bisection traced it to llama's ggml-metal `im2col_ext` path: its
  `ggml-metal-device.cpp` routes large im2col (the conv front-end of whisper's
  encoder, `ne00*ne01 > 1024`) to `kernel_im2col_ext`, which miscomputes for
  whisper. The base `kernel_im2col` is correct in both snapshots; llama's own
  LLMs never use im2col, so the regression was invisible upstream. The fix is
  to link whisper against its OWN ggml (see CMakeLists.txt `_whisper_native`).

What this test guards: for the current llama/whisper/ggml pin combination on
the current platform, the GPU backend must transcribe correctly and agree with
the CPU backend. If it fails, inferna must NOT be released with GPU enabled for
whisper on this platform/pin combination -- it is exactly the failure that the
rest of the suite (which only transcribes implicitly, or on CPU) does not
catch.

On CPU-only hosts `use_gpu=True` falls back to CPU, so GPU==CPU trivially and
this test is benign; it only has teeth where a real GPU backend (Metal/CUDA/
Vulkan) is active.
"""

from __future__ import annotations

import gc
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from inferna.whisper import whisper_cpp as wh

WHISPER_MODEL_PATH = Path("models/ggml-base.en.bin")
JFK_PATH = Path("tests/samples/jfk.wav")

# Canonical fragments of the JFK clip transcript; tokenisation varies across
# builds, so we check robust lowercase substrings rather than exact equality.
JFK_PHRASES = ("ask not", "country")


@pytest.fixture(scope="module")
def whisper_model_path() -> str:
    if not WHISPER_MODEL_PATH.exists():
        pytest.skip(f"Whisper model not found at {WHISPER_MODEL_PATH}")
    return str(WHISPER_MODEL_PATH)


@pytest.fixture(scope="module")
def jfk_samples() -> np.ndarray:
    if not JFK_PATH.exists():
        pytest.skip(f"JFK sample not found at {JFK_PATH}")
    with wave.open(str(JFK_PATH), "rb") as w:
        frames = w.readframes(-1)
        params = w.getparams()
    if params.sampwidth != 2 or params.framerate != 16000:
        pytest.skip(f"unexpected jfk.wav format (sampwidth={params.sampwidth}, fr={params.framerate})")
    raw = struct.unpack(f"{len(frames) // 2}h", frames)
    return np.array([s / 32768.0 for s in raw], dtype=np.float32)


def _transcribe(model_path: str, samples: np.ndarray, *, use_gpu: bool) -> str:
    """Run one whisper_full pass on the given backend; return joined text."""
    wh.disable_logging()
    wh.ggml_backend_load_all()
    cparams = wh.WhisperContextParams()
    cparams.use_gpu = use_gpu
    ctx = wh.WhisperContext(model_path, cparams)
    try:
        params = wh.WhisperFullParams()
        params.n_threads = 4
        params.language = "en"
        params.print_progress = False
        params.print_realtime = False
        params.print_timestamps = False
        params.print_special = False
        rc = ctx.full(samples, params)
        assert rc == 0, f"whisper_full failed (use_gpu={use_gpu}): rc={rc}"
        n = ctx.full_n_segments()
        text = " ".join(ctx.full_get_segment_text(i) for i in range(n)).strip()
    finally:
        ctx = None
    del ctx
    gc.collect()
    return text


def test_gpu_backend_transcribes_correctly(whisper_model_path: str, jfk_samples: np.ndarray) -> None:
    """GPU transcription must be non-empty and contain the canonical phrases.

    This is the direct symptom of the ggml-metal `im2col_ext` miscomputation:
    a clean rc=0 but an empty transcript on GPU.
    """
    gpu_text = _transcribe(whisper_model_path, jfk_samples, use_gpu=True)
    assert gpu_text, (
        "GPU backend produced an EMPTY transcript (whisper_full rc=0, 0 segments). "
        "This is the ggml-metal/im2col_ext miscomputation signature -- the active "
        "ggml linked into _whisper_native is incompatible with whisper's encoder "
        "on this GPU backend. Do not release with GPU enabled. See this file's "
        "module docstring."
    )
    low = gpu_text.lower()
    for phrase in JFK_PHRASES:
        assert phrase in low, f"GPU transcript missing {phrase!r}: {gpu_text!r}"


def test_gpu_matches_cpu(whisper_model_path: str, jfk_samples: np.ndarray) -> None:
    """GPU and CPU backends must agree on the canonical phrases.

    CPU is the reference (it tolerated the bug); GPU must match it. Divergence
    means the GPU backend miscomputes whisper's graph for this pin combination.
    """
    cpu_text = _transcribe(whisper_model_path, jfk_samples, use_gpu=False).lower()
    gpu_text = _transcribe(whisper_model_path, jfk_samples, use_gpu=True).lower()

    # CPU is the reference and must itself be correct.
    for phrase in JFK_PHRASES:
        assert phrase in cpu_text, f"CPU reference transcript missing {phrase!r}: {cpu_text!r}"

    # GPU must agree with CPU on every canonical phrase.
    for phrase in JFK_PHRASES:
        assert phrase in gpu_text, (
            f"GPU transcript missing {phrase!r} that CPU produced -- GPU backend "
            f"miscomputes whisper's graph for this ggml pin. GPU={gpu_text!r}"
        )
