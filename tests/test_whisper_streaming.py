"""Tests for ``inferna.whisper.streaming``.

Coverage:
  * Argument validation (length_ms < step_ms, n_threads, sample-rate
    contract via the public constant).
  * Synthesised silence: feed produces no segments; flush is graceful.
  * Real audio (jfk.wav, 11s @ 16 kHz): chunked feeding produces
    segments whose final-flagged texts roughly match the expected
    JFK transcript.
  * Iterator helper ``transcribe_stream`` matches what manual
    feed/flush would yield.
"""

from __future__ import annotations

import gc
import struct
import wave
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from inferna.whisper.streaming import (
    WHISPER_SAMPLE_RATE,
    StreamSegment,
    WhisperStreamer,
    transcribe_stream,
)


WHISPER_MODEL_PATH = Path("models/ggml-base.en.bin")
JFK_PATH = Path("tests/samples/jfk.wav")


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
    if params.sampwidth != 2 or params.framerate != WHISPER_SAMPLE_RATE:
        pytest.skip(f"unexpected jfk.wav format (sampwidth={params.sampwidth}, fr={params.framerate})")
    fmt = f"{len(frames) // 2}h"
    raw = struct.unpack(fmt, frames)
    return np.array([s / 32768.0 for s in raw], dtype=np.float32)


def chunk(arr: np.ndarray, size: int) -> Iterator[np.ndarray]:
    """Split ``arr`` into ``size``-sample chunks (mimicking a mic source)."""
    for start in range(0, arr.shape[0], size):
        yield arr[start : start + size]


# ----------------------------------------------------------------------
# Pure-Python validation paths -- no model required.
# ----------------------------------------------------------------------


class TestStreamerArgValidation:
    def test_length_lt_step_raises(self) -> None:
        with pytest.raises(ValueError, match="must be >= step_ms"):
            WhisperStreamer("nonexistent.bin", step_ms=3000, length_ms=1000)

    def test_zero_step_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            WhisperStreamer("nonexistent.bin", step_ms=0, length_ms=10)

    def test_zero_threads_raises(self) -> None:
        with pytest.raises(ValueError, match="n_threads"):
            WhisperStreamer("nonexistent.bin", n_threads=0)


class TestSegmentDataclass:
    def test_immutable(self) -> None:
        seg = StreamSegment(text="hi", t0=0.0, t1=1.0, is_final=True)
        with pytest.raises(Exception):
            # frozen dataclass: setattr must raise (FrozenInstanceError
            # is a subclass of AttributeError; check both via Exception).
            seg.text = "bye"  # type: ignore[misc]


# ----------------------------------------------------------------------
# Live model tests -- exercise the rolling-window pass.
# ----------------------------------------------------------------------


class TestStreamerSilence:
    def test_silent_input_yields_nothing_or_blank(self, whisper_model_path: str) -> None:
        # Feed half a second of silence; whisper should either return
        # no segments or empty-string segments. Either is acceptable;
        # the contract is that feed/flush don't crash and don't emit
        # spurious non-blank text.
        streamer = WhisperStreamer(
            whisper_model_path,
            step_ms=500,
            length_ms=2000,
            verbose=False,
        )
        try:
            silence = np.zeros(WHISPER_SAMPLE_RATE // 2, dtype=np.float32)
            streamer.feed(silence)
            for _ in range(3):
                streamer.feed(silence)
            final = streamer.flush()
            for seg in final:
                # No false-positive transcription on pure silence.
                # Whisper occasionally emits things like "[BLANK_AUDIO]"
                # in brackets; allow those, reject plain words.
                if seg.text:
                    assert seg.text.startswith("[") or seg.text.startswith("(")
        finally:
            streamer.close()
            del streamer
            gc.collect()


class TestStreamerJFK:
    """End-to-end on the JFK sample.

    The clip is ~11 seconds; we feed it as 250 ms chunks and assert
    that the joined transcript contains a couple of canonical phrases.
    """

    def test_chunked_feed_produces_jfk_transcript(self, whisper_model_path: str, jfk_samples: np.ndarray) -> None:
        streamer = WhisperStreamer(
            whisper_model_path,
            step_ms=2000,
            length_ms=8000,
            verbose=False,
        )
        try:
            collected: list[StreamSegment] = []
            chunk_size = WHISPER_SAMPLE_RATE // 4  # 250 ms
            for c in chunk(jfk_samples, chunk_size):
                collected.extend(streamer.feed(c))
            collected.extend(streamer.flush())
        finally:
            streamer.close()
            del streamer
            gc.collect()

        finals = [s for s in collected if s.is_final]
        assert len(finals) >= 1, "no final segments produced"

        # Stitch all final segments into one transcript and lower-case
        # for a robust phrase check (whisper's exact tokenisation
        # varies across builds).
        text = " ".join(s.text for s in finals).lower()
        # JFK clip canonical phrase fragments.
        assert "ask not" in text, f"transcript missing 'ask not': {text!r}"
        assert "country" in text, f"transcript missing 'country': {text!r}"

    def test_iterator_helper_matches_class(self, whisper_model_path: str, jfk_samples: np.ndarray) -> None:
        chunks = list(chunk(jfk_samples, WHISPER_SAMPLE_RATE // 4))
        out = list(
            transcribe_stream(
                iter(chunks),
                model_path=whisper_model_path,
                step_ms=2000,
                length_ms=8000,
                verbose=False,
            )
        )
        finals = [s for s in out if s.is_final]
        assert finals, "transcribe_stream produced no final segments"
        text = " ".join(s.text for s in finals).lower()
        assert "ask not" in text


class TestStreamerMonotonicTime:
    def test_finals_are_non_overlapping(self, whisper_model_path: str, jfk_samples: np.ndarray) -> None:
        # Each final segment's t0 should be >= the previous final
        # segment's t1 (modulo a small ε for whisper's centisecond
        # rounding). This pins the stream-relative-timestamp logic.
        streamer = WhisperStreamer(
            whisper_model_path,
            step_ms=2000,
            length_ms=8000,
            verbose=False,
        )
        try:
            for c in chunk(jfk_samples, WHISPER_SAMPLE_RATE // 4):
                streamer.feed(c)
            finals = [s for s in streamer.flush() if s.is_final]
        finally:
            streamer.close()
            del streamer
            gc.collect()
        last_t1 = -1.0
        for s in finals:
            assert s.t1 >= s.t0
            assert s.t0 + 1e-2 >= last_t1, f"overlapping finals: {finals}"
            last_t1 = s.t1


class TestStreamerLifecycle:
    def test_feed_after_close_raises(self, whisper_model_path: str) -> None:
        streamer = WhisperStreamer(whisper_model_path, verbose=False)
        streamer.close()
        with pytest.raises(RuntimeError, match="closed"):
            streamer.feed(np.zeros(100, dtype=np.float32))
        del streamer
        gc.collect()

    def test_context_manager_closes(self, whisper_model_path: str) -> None:
        with WhisperStreamer(whisper_model_path, verbose=False) as streamer:
            assert not streamer._closed
        assert streamer._closed
