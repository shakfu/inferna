"""Buffer helpers for the whisper audio path -- stdlib only.

whisper's native ``full()`` binding takes an ``nb::ndarray<float, ndim<1>,
c_contig, device::cpu>``, which nanobind fills from *any* object exposing a
1-D C-contiguous float32 buffer. ``numpy.ndarray`` is one such object, but so
are :class:`array.array` and :class:`memoryview`, so nothing on this path
actually requires numpy.

These helpers normalise caller-supplied audio to that buffer shape without
importing numpy. numpy arrays still pass through zero-copy (a float32
ndarray already exposes exactly the buffer we want), so callers who have
numpy lose nothing.
"""

from __future__ import annotations

from array import array
from typing import Any, Iterable, List, Union

# Any object exposing a 1-D buffer of audio samples: numpy.ndarray,
# array.array, memoryview, or anything else implementing the buffer
# protocol. Deliberately not tied to numpy -- see the module docstring.
AudioSamples = Any

# What the helpers below hand back: always a 1-D, C-contiguous, float32
# buffer that the native binding accepts directly.
Float32Buffer = Union[memoryview, "array[float]"]


def as_float32(samples: AudioSamples) -> Float32Buffer:
    """Return ``samples`` as a 1-D contiguous float32 buffer.

    Float32 input is returned as a view (no copy). Other element types are
    converted, which costs a copy -- the same trade the previous
    ``astype(np.float32)`` made.
    """
    try:
        view = memoryview(samples)
    except TypeError as exc:
        raise TypeError(
            "samples must expose the buffer protocol (numpy.ndarray, "
            f"array.array('f'), memoryview, ...); got {type(samples).__name__}"
        ) from exc

    if view.ndim != 1:
        shape = tuple(view.shape) if view.shape is not None else ()
        raise ValueError(f"samples must be 1-D mono audio, got shape {shape}")

    if view.format == "f" and view.c_contiguous:
        return view

    # Non-float32 or strided: materialise a float32 copy. tolist() handles
    # both cases uniformly and is the only stdlib route that reads a
    # strided buffer correctly.
    return array("f", view.tolist())


def concat_float32(parts: Iterable[Float32Buffer]) -> "array[float]":
    """Concatenate float32 buffers into one contiguous ``array('f')``.

    Empty input yields an empty array, matching ``np.zeros(0, np.float32)``.
    """
    out = array("f")
    for part in parts:
        # cast("B") gives a raw byte view of the (contiguous) samples, so
        # frombytes is a single memcpy rather than an element-wise loop.
        buf = part if isinstance(part, memoryview) else memoryview(part)
        out.frombytes(buf.cast("B"))
    return out


def resample_linear(samples: AudioSamples, orig_sr: int, target_sr: int) -> "array[float]":
    """Resample by linear interpolation between neighbouring samples.

    Equivalent to the previous ``np.interp`` over ``np.linspace``. numpy is
    used as a pure speed-up when it happens to be installed -- long inputs
    are an O(n) Python loop otherwise -- but the result is identical either
    way, and the return type is always ``array('f')`` so callers never see
    the accelerator leak into the API.
    """
    src = as_float32(samples)
    n_src = len(src)

    if orig_sr == target_sr:
        return concat_float32([src])

    new_length = int(n_src / (orig_sr / target_sr))
    if n_src == 0 or new_length <= 0:
        return array("f")
    if new_length == 1:
        return array("f", [src[0]])

    try:
        import numpy as _np  # noqa: PLC0415 -- optional accelerator only
    except ImportError:
        pass
    else:
        old_idx = _np.arange(n_src)
        new_idx = _np.linspace(0, n_src - 1, new_length)
        resampled = _np.interp(new_idx, old_idx, _np.frombuffer(src, dtype=_np.float32))
        out = array("f")
        out.frombytes(resampled.astype(_np.float32).tobytes())
        return out

    out = array("f", bytes(4 * new_length))
    span = (n_src - 1) / (new_length - 1)
    for i in range(new_length):
        pos = i * span
        lo = int(pos)
        hi = lo + 1 if lo + 1 < n_src else n_src - 1
        frac = pos - lo
        out[i] = src[lo] * (1.0 - frac) + src[hi] * frac
    return out


def float32_from_values(values: List[float]) -> "array[float]":
    """Build a float32 buffer from a list of Python floats."""
    return array("f", values)
