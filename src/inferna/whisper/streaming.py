"""Streaming transcription via a rolling-window re-pass.

whisper.cpp's encoder consumes a full mel spectrogram per call -- there
is no incremental decoder. The standard way to do "streaming"
transcription on top of it (as in upstream's ``examples/stream``) is the
rolling-window pattern:

  * Buffer audio as it arrives.
  * Every ``step_ms`` of accumulated new audio, re-run ``whisper_full``
    on the trailing ``length_ms`` of the buffer.
  * Segments older than the rolling window have rolled past and are
    "final"; segments inside the window can still change in later
    passes.

This module wraps that pattern in a :class:`WhisperStreamer` class plus
a :func:`transcribe_stream` iterator helper. The trade-off vs. one-shot
``ctx.full(buffer)`` is responsiveness vs. CPU: smaller ``step_ms``
gives faster partial transcripts but more passes per second.

Whisper expects 16 kHz mono float32 audio; ``feed`` enforces that.
Callers from microphone sources are responsible for resampling /
channel-mixing upstream (e.g. via ``sounddevice`` + ``scipy.signal``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List, Optional, cast

import numpy as np

logger = logging.getLogger(__name__)


# Whisper requires 16 kHz mono float32 audio. The encoder hard-codes
# this in its mel-spectrogram step, so anything else has to be
# resampled before reaching the streamer.
WHISPER_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class StreamSegment:
    """One transcribed segment from a streaming pass.

    Times are seconds from the stream's first ``feed()`` call.

    ``is_final`` is set when the segment has rolled past the trailing
    edge of the active window -- subsequent passes won't reprocess it,
    so its text is committed. Non-final segments may be re-emitted
    (with possibly different text) on the next pass; callers that want
    a single-pass transcript can ignore non-final entries until they
    flip.
    """

    text: str
    t0: float
    t1: float
    is_final: bool


class WhisperStreamer:
    """Rolling-window streaming transcriber.

    Construct once, ``feed`` audio chunks as they arrive, consume the
    segments returned by each ``feed`` call, then call ``flush`` when
    the source is exhausted to drain the trailing audio.

    Args:
        model_path: Path to a GGML whisper model.
        step_ms: How often to run a transcription pass. Smaller values
            cut latency but raise CPU. The upstream stream example
            defaults to 3000 ms; we match.
        length_ms: Rolling window size. Must be ≥ ``step_ms``. Larger
            window = better continuity at segment boundaries (the
            decoder sees more context) but more CPU per pass.
        language: ISO 639-1 code; ``None`` = auto-detect each pass.
        translate: If True, output is translated to English regardless
            of source language (whisper's built-in translate mode).
        n_threads: Encoder/decoder threads.
        verbose: If False, native whisper.cpp log noise is suppressed.

    Example:
        >>> streamer = WhisperStreamer("models/ggml-base.en.bin")
        >>> for chunk in mic_chunks():        # 16 kHz float32 ndarrays
        ...     for seg in streamer.feed(chunk):
        ...         marker = "FINAL" if seg.is_final else " ~  "
        ...         print(f"{marker} [{seg.t0:.1f}-{seg.t1:.1f}] {seg.text}")
        >>> for seg in streamer.flush():
        ...     print(f"FINAL [{seg.t0:.1f}-{seg.t1:.1f}] {seg.text}")
    """

    def __init__(
        self,
        model_path: str,
        *,
        step_ms: int = 3000,
        length_ms: int = 10000,
        language: Optional[str] = "en",
        translate: bool = False,
        n_threads: int = 4,
        verbose: bool = False,
    ) -> None:
        if length_ms < step_ms:
            raise ValueError(f"length_ms ({length_ms}) must be >= step_ms ({step_ms})")
        if step_ms <= 0 or length_ms <= 0:
            raise ValueError("step_ms and length_ms must be positive")
        if n_threads <= 0:
            raise ValueError("n_threads must be positive")

        from . import whisper_cpp as _wh

        self._wh: Any = _wh
        self._model_path = model_path
        self.step_ms = step_ms
        self.length_ms = length_ms
        self.language = language
        self.translate = translate
        self.n_threads = n_threads
        self.verbose = verbose

        if not verbose:
            _wh.disable_logging()
        _wh.ggml_backend_load_all()

        self._ctx = _wh.WhisperContext(model_path)

        # Step / length expressed in 16 kHz samples for fast comparisons.
        self._step_samples = step_ms * WHISPER_SAMPLE_RATE // 1000
        self._length_samples = length_ms * WHISPER_SAMPLE_RATE // 1000

        # Rolling buffer holds at most ``length_samples`` of audio. We
        # use a list-of-arrays + concat-on-pass rather than a circular
        # ndarray because every transcription pass needs a contiguous
        # ndarray anyway, and the bookkeeping is simpler.
        self._pending: List[np.ndarray] = []
        self._pending_samples = 0

        # Audio that's already inside the rolling window from prior
        # passes. Concatenated with ``_pending`` to form the
        # transcription input.
        self._window: Optional[np.ndarray] = None
        # Sample index (relative to stream start) of the first sample
        # in ``_window``. Used to convert pass-local timestamps to
        # absolute stream-relative timestamps.
        self._window_start_samples = 0
        # Total samples ingested so far.
        self._total_samples = 0
        # Highest ``t1`` we've emitted as final, seconds since stream
        # start. Anything below this in a future pass is suppressed --
        # the watermark guarantees each finalised time region is
        # emitted exactly once, even though whisper re-transcribes
        # the rolling window's overlap on every pass.
        self._final_until_sec = 0.0

        self._closed = False

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def feed(self, samples: np.ndarray) -> List[StreamSegment]:
        """Append ``samples`` (mono float32 @ 16 kHz) to the buffer.

        Runs a transcription pass for every ``step_ms`` of newly
        accumulated audio (a single ``feed`` may trigger several
        passes when the caller batches a long file). Returns the
        segments produced across those passes -- in stream-relative
        time order, with ``is_final=True`` on segments whose audio has
        rolled past the trailing edge of the active window.
        """
        self._ensure_open()
        samples = self._coerce_samples(samples)
        self._pending.append(samples)
        self._pending_samples += samples.shape[0]
        self._total_samples += samples.shape[0]

        produced: List[StreamSegment] = []
        while self._pending_samples >= self._step_samples:
            produced.extend(self._run_pass(is_flush=False))
        return produced

    def flush(self) -> List[StreamSegment]:
        """Run a final pass over any remaining audio and emit final segments.

        Returns segments not already committed by earlier ``feed``
        passes, all marked ``is_final=True``. After ``flush`` the
        streamer is exhausted; further ``feed`` calls raise.
        """
        self._ensure_open()
        produced: List[StreamSegment] = []
        if self._pending_samples > 0 or self._window is not None:
            produced.extend(self._run_pass(is_flush=True))
        self._closed = True
        return produced

    def close(self) -> None:
        """Release the native context. Idempotent."""
        if not self._closed:
            self._closed = True
            self._ctx = None

    def __enter__(self) -> "WhisperStreamer":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("WhisperStreamer is closed; construct a new instance")

    def _coerce_samples(self, samples: np.ndarray) -> np.ndarray:
        """Validate shape/dtype and return a contiguous float32 view."""
        arr = np.asarray(samples)
        if arr.ndim != 1:
            raise ValueError(f"samples must be 1-D mono audio, got shape {arr.shape}")
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        return cast(np.ndarray, np.ascontiguousarray(arr))

    def _build_pass_audio(self) -> np.ndarray:
        """Concatenate ``_window`` + ``_pending`` into the next pass's input.

        Updates ``_window`` to hold the trailing ``length_samples`` of
        the result (so the next pass sees the same trailing context),
        and clears ``_pending``. Updates ``_window_start_samples`` to
        the absolute sample index of the first sample in the new
        ``_window``.
        """
        parts: List[np.ndarray] = []
        if self._window is not None:
            parts.append(self._window)
        if self._pending:
            parts.extend(self._pending)
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

        # Cap at length_samples so the encoder's per-call cost stays
        # bounded. Drop the oldest excess.
        if audio.shape[0] > self._length_samples:
            drop = audio.shape[0] - self._length_samples
            audio = audio[drop:]
            self._window_start_samples += drop
        elif audio.shape[0] < self._length_samples and self._window is not None:
            # No drop yet; window_start_samples stays the same.
            pass

        # The next call's window holds the trailing slice.
        self._window = audio
        self._pending = []
        self._pending_samples = 0
        return audio

    def _run_pass(self, *, is_flush: bool) -> List[StreamSegment]:
        """Run one ``whisper_full`` call and emit segments.

        ``is_flush`` flips every segment in this pass to final --
        there'll be no more passes that could revise them.
        """
        wh = self._wh

        audio = self._build_pass_audio()
        if audio.shape[0] == 0:
            return []

        params = wh.WhisperFullParams()
        params.n_threads = self.n_threads
        params.language = self.language
        params.translate = self.translate
        # Suppress upstream stdout chatter on every pass; callers who
        # want it can subclass and override _run_pass.
        params.print_progress = False
        params.print_realtime = False
        params.print_timestamps = False
        params.print_special = False
        # ``no_context`` resets decoder state between passes -- for
        # rolling-window streaming we feed the audio context via the
        # buffer overlap rather than the prior tokens, so this avoids
        # carrying a prior pass's hallucinations forward.
        params.no_context = True

        rc = self._ctx.full(audio, params)
        if rc != 0:
            raise RuntimeError(f"whisper_full failed: rc={rc}")

        # Pass-local timestamps are in centiseconds (10 ms units) per
        # whisper.cpp's API. Convert to seconds and shift by the
        # window's absolute start position so they're stream-relative.
        window_start_sec = self._window_start_samples / WHISPER_SAMPLE_RATE
        n = self._ctx.full_n_segments()
        segments: List[StreamSegment] = []
        # Anything entirely within the soon-to-be-discarded prefix is
        # final this pass; anything that touches the trailing edge is
        # provisional unless we're flushing.
        # ``trailing_edge_sec`` is the last second of audio; segments
        # whose t1 falls before ``trailing_edge_sec - step_sec`` are
        # outside the next pass's window and won't reappear.
        audio_end_sec = window_start_sec + audio.shape[0] / WHISPER_SAMPLE_RATE
        step_sec = self.step_ms / 1000.0
        for i in range(n):
            t0_cs = self._ctx.full_get_segment_t0(i)
            t1_cs = self._ctx.full_get_segment_t1(i)
            text = self._ctx.full_get_segment_text(i)
            t0 = window_start_sec + t0_cs / 100.0
            t1 = window_start_sec + t1_cs / 100.0
            # Drop segments we've already emitted as final on a prior
            # pass; whisper sometimes re-emits leading audio in the
            # window.
            if t1 <= self._final_until_sec + 1e-3:
                continue
            is_final = is_flush or (t1 + step_sec <= audio_end_sec)
            seg = StreamSegment(text=text.strip(), t0=t0, t1=t1, is_final=is_final)
            if not seg.text:
                continue
            segments.append(seg)
            if is_final:
                # Advance the watermark monotonically so future passes
                # don't re-emit this region.
                if seg.t1 > self._final_until_sec:
                    self._final_until_sec = seg.t1

        return segments


def transcribe_stream(
    audio_iter: Iterable[np.ndarray],
    *,
    model_path: str,
    step_ms: int = 3000,
    length_ms: int = 10000,
    language: Optional[str] = "en",
    translate: bool = False,
    n_threads: int = 4,
    verbose: bool = False,
) -> Iterator[StreamSegment]:
    """Iterator-shaped streaming transcription.

    Wraps :class:`WhisperStreamer` for the common case of a chunk
    iterator (audio file reader, microphone callback queue, ...). The
    streamer is constructed, fed each chunk in turn, and flushed on
    exhaustion. All segments -- both rolling and final -- pass
    through.

    Example:
        >>> from inferna.whisper.streaming import transcribe_stream
        >>> for seg in transcribe_stream(read_chunks("speech.wav"),
        ...                              model_path="models/ggml-base.en.bin"):
        ...     marker = "FINAL" if seg.is_final else "    ~"
        ...     print(f"{marker} [{seg.t0:.1f}-{seg.t1:.1f}] {seg.text}")
    """
    with WhisperStreamer(
        model_path,
        step_ms=step_ms,
        length_ms=length_ms,
        language=language,
        translate=translate,
        n_threads=n_threads,
        verbose=verbose,
    ) as streamer:
        for chunk in audio_iter:
            for seg in streamer.feed(chunk):
                yield seg
        for seg in streamer.flush():
            yield seg
