Reproducible in whisper.cpp on Apple Metal 

For transparency: AI was used to help analyze and write up this issue. The defect itself was independently reproduced in two projects, [cyllama](https://github.com/shakfu/cyllama) and [inferna](https://github.com/shakfu/inferna), both of which consume llama.cpp's ggml via llama.cpp, whisper.cpp, and stable-diffusion.cpp.

**Environment:** Apple M1, macOS (Darwin 24.6), Metal backend; ggml as vendored in llama.cpp `b9528` (`GGML_METAL=ON`, static).

## Summary

The Metal `im2col` **kernel** and **dispatch geometry** are selected using different predicates:

* `ggml-metal-device.cpp` selects `kernel_im2col_ext` when `ne00 * ne01 > 1024` (`KW * IC` for 1-D convs).
* `ggml-metal-ops.cpp` uses `_ext` dispatch geometry only when `KH * KW > max_threads_per_threadgroup`.

For 1-D convolutions with large channel counts, these predicates can disagree, causing `kernel_im2col_ext` to run with the base kernel's grid layout and produce incorrect output.

## Impact

Whisper's encoder uses a 1-D convolution with:

```
KW=3, KH=1, IC=512
```

This yields:

```
KW * IC = 1536 > 1024   -> selects kernel_im2col_ext
KH * KW = 3             -> uses base dispatch geometry
```

On Metal, activations become corrupted and decoding fails:

* `whisper_full()` returns `rc=0`
* zero segments
* empty transcript

The same graph produces correct output on CPU. 2-D convolutions are unaffected because both selection sites use `KH * KW`.

## Reproduction

Build `whisper-cli` against a ggml revision containing the size-based `im2col` kernel selection, then run the same binary on each backend -- only the backend flag changes:

```sh
# Metal (GPU): default
whisper-cli -m models/ggml-base.en.bin -f samples/jfk.wav

# CPU: same binary, GPU disabled
whisper-cli -ng -m models/ggml-base.en.bin -f samples/jfk.wav
```

Expected:

> "And so my fellow Americans..."

Actual on Metal: 0 segments, empty transcript.
Actual on CPU (`-ng`): correct transcription.

Same binary, same graph -- flipping only the backend toggles the bug, which isolates the defect to the Metal `im2col` path.

## Root Cause

Pipeline selection:

```c
if (ne00 * ne01 <= 1024)
    kernel_im2col;
else
    kernel_im2col_ext;
```

Dispatch selection:

```c
if (KH * KW <= max_threads_per_threadgroup)
    base_geometry;
else
    ext_geometry;
```

These predicates are not equivalent for 1-D convolutions (`KW * IC` vs `KH * KW`), allowing the `_ext` kernel to run under the wrong dispatch geometry.

## Verification

The Metal shaders confirm that the two kernels interpret `tgpig[0]` differently:

```c
// kernel_im2col_ext
d   = tgpig[0] / CHW;
chw = tgpig[0] % CHW;

// kernel_im2col
iic = tgpig[0];
```

`kernel_im2col_ext` expects grid-x to be `quotient * CHW`, while the base dispatch provides only `IC`. For Whisper's `KW=3, IC=512`, channel indices alias (`iic / 3`), ~2/3 of channels are never addressed, and activations become garbage.

The `_ext` kernel itself appears correct: forcing the base kernel or making both sites use the same predicate restores correct output.

## Proposed Fix

Make dispatch selection follow the same criterion used for pipeline selection:

```diff
- if (KH * KW <= max_threads_per_threadgroup) {
+ if (ne00 * ne01 <= 1024) {
```

Preferably, derive the geometry directly from the selected pipeline so the predicates cannot diverge.

## Suggested Test

Add a Metal-vs-CPU `GGML_OP_IM2COL` test covering the mismatch case:

```
KW=3, KH=1, IC=512
```

where:

```
KW * IC > 1024
KH * KW <= max_threads_per_threadgroup
```

for both F16 and F32.

shell script attached to demonstrate issue from clean slate.

[repro_ggml_metal_im2col.sh](https://github.com/user-attachments/files/28698823/repro_ggml_metal_im2col.sh)