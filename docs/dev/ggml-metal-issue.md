# whisper-on-Metal silent zero-output: ggml `im2col_ext` mismatch

Status: resolved (fix in `CMakeLists.txt`, regression gate in
`tests/test_whisper_gpu_parity.py`).

Platform observed: macOS / Apple M1, Metal backend. Pins at time of
investigation: llama.cpp `b9528`, whisper.cpp `v1.8.6`, sd `master-672`.

## TL;DR

`_whisper_native` was compiled from whisper.cpp but linked against
**llama.cpp's** ggml-metal. llama and whisper each vendor their own ggml
snapshot (both stamped `0.13.1`, different upstream commits). On the Metal
(GPU) backend, whisper's encoder produced **zero segments with `rc=0`** -- a
clean success return and an empty transcript -- while the CPU backend was
correct.

Root cause (bisected to a single change): llama's
`ggml/src/ggml-metal/ggml-metal-device.cpp` routes large `im2col` (the
convolution front-end of whisper's encoder, `ne00*ne01 > 1024`) to
`kernel_im2col_ext`, which miscomputes for whisper's case. The base
`kernel_im2col` is correct in both snapshots. llama's own LLM inference never
uses `im2col`, so the regression is invisible upstream.

Fix: link `_whisper_native` against whisper's **own** ggml (its `whisper.a`
paired with its matching ggml-metal), exactly as `whisper-cli` does and as SD
already does for itself. Two ggml copies coexist fine in one process because
each extension's symbols are hidden (`-fvisibility=hidden`).

## Symptom

- `whisper_full` returns `rc=0`, `whisper_full_n_segments() == 0`, empty text.
- Only on the GPU/Metal backend. CPU (`use_gpu=False`) transcribes correctly.
- `whisper-cli` (the C++ binary built in whisper.cpp's own tree) transcribes
  correctly on Metal -- because it links whisper's own ggml.
- Invisible to the rest of the suite: the only tests that actually transcribe
  and assert text on Metal were `tests/test_whisper_streaming.py`; everything
  else either does not transcribe, asserts only metadata, or runs on CPU.

## What was ruled out (each tested, not assumed)

1. Not the version bump: cyllama (sibling project, llama `b9505` + same
   whisper) reproduces it identically. Reverting llama does not help.
2. Not a whisper.cpp bug: `whisper-cli` works on Metal with the same libs.
3. Not a ggml version/API skew: llama and whisper both vendor ggml `0.13.1`;
   public headers (`ggml.h`, `ggml-metal.h`, `ggml-cpu.h`) are identical;
   `GGML_MAX_NAME` is `64` in both; backend flags match.
4. Not multi-ggml symbol collision: reproduced with ONLY `_whisper_native.so`
   loaded (no llama/sd modules in the process).
5. Not a ggml leak from whisper's libs: `libwhisper.a` / `libcommon.a` carry
   zero ggml-metal objects.
6. Not the binding's `full()`: it passes a valid pointer + length and calls
   `whisper_full` synchronously; the CPU path through the same code is correct.
7. Not ABI/struct layout: rebuilding whisper.cpp **against llama's ggml
   headers** and linking llama's ggml (a fully self-consistent llama snapshot,
   no API mismatch) STILL produced zero segments. So the break is behavioral,
   not API.
8. Not the ~200-line `ggml-metal.metal` shader delta: see bisection below --
   the shaders are innocent.

## Bisection

Method: a minimal C++ probe (`whisper_init_from_file_with_params` ->
`whisper_full` on `tests/samples/jfk.wav`, print `n_segments`) linked against
`whisper.a` + a controllable ggml lib set. ggml was rebuilt from source with
individual files swapped between whisper's and llama's snapshots.

| Configuration | Result |
| --- | --- |
| whisper host + whisper shaders (baseline) | works (1 segment) |
| full llama ggml-metal | broken (0 segments) |
| whisper host + llama `ggml-metal.metal` (shaders) only | works -> shaders innocent |
| whisper host + llama `ggml-metal-ops.cpp` only | works -> ops innocent |
| whisper host + llama `ggml-metal-device.m` only | works (its only compute change is GLU F16, unused by whisper) |
| llama shaders + whisper host (inverse) | works -> confirms shaders innocent |
| full llama, revert ONLY the `im2col` kernel selection | fixed (1 segment) |

The differing ggml-metal files between the two snapshots were
`ggml-metal.metal`, `ggml-metal-ops.cpp`, `ggml-metal-device.m`, and
`ggml-metal-device.cpp`. Single-file swaps either worked or segfaulted (the
host C++/ObjC/shader pieces are coupled and must move together), so the
decisive test was reverting one logical change inside the otherwise-unmodified
full llama tree.

The culprit, in `ggml-metal-device.cpp` (pipeline selection for `im2col`):

```c
// whisper v1.8.6 -- always the base kernel:
snprintf(base, 256, "kernel_im2col_%s", ggml_type_name(op->type));

// llama b9528 -- size-based selection added:
GGML_TENSOR_LOCALS(int64_t, ne0, op->src[0], ne);
if (ne00*ne01 <= 1024) {
    snprintf(base, 256, "kernel_im2col_%s", ggml_type_name(op->type));
} else {
    snprintf(base, 256, "kernel_im2col_ext_%s", ggml_type_name(op->type));
}
```

Reverting only this block in the full llama tree (forcing the base kernel)
restores correct transcription.

### Mechanism (verified): kernel selection vs dispatch geometry disagree

The `_ext` kernel is not itself broken -- it is dispatched with the **wrong
threadgroup/grid geometry**, because two sites choose base-vs-ext by *different*
criteria:

- Kernel/pipeline selection, `ggml-metal-device.cpp`
  (`ggml_metal_library_get_pipeline_im2col`): `_ext` when `ne00*ne01 > 1024`,
  where `ne00,ne01` are `op->src[0]` (the conv kernel) dims = `KW * IC` for a
  1-D conv (`src[0]->ne = [KW, IC, OC]`).
- Dispatch geometry, `ggml-metal-ops.cpp` (`ggml_metal_op_im2col`): emits the
  `_ext` layout (`dispatch_threadgroups(quotient*CHW, OH, OW, n_threads,1,1)`)
  only when `KH*KW > max_threads_per_threadgroup`; otherwise the base layout
  (`dispatch_threadgroups(IC, OH, OW, ntptg0, KH, KW)`).

`KW*IC` (selection) and `KH*KW` (geometry) are different quantities, so they can
disagree. whisper's encoder has two 1-D convs:

- conv1 `[KW=3, IC=80]`: `ne00*ne01 = 240 <= 1024` -> base kernel; `KH*KW = 3`
  -> base geometry. Consistent -> correct.
- conv2 `[KW=3, IC=512]`: `ne00*ne01 = 1536 > 1024` -> **`_ext` kernel**;
  `KH*KW = 3 <= 1024` -> **base geometry**. Mismatch -> the `_ext` kernel runs
  under the base kernel's grid decomposition, mis-indexes, writes garbage.

Two independent fixes both restore correct output (each verified by rebuilding
ggml and re-running the probe), which is what pins the mismatch as the cause:

1. Force the base kernel in `device.cpp` (base kernel + base geometry).
2. Make `ops.cpp` use the same `ne00*ne01 <= 1024` criterion as `device.cpp`
   (so the `_ext` kernel gets `_ext` geometry).

The upstream-correct fix is (2): keep the optimization but dispatch matching
geometry (ideally derive geometry from the selected pipeline so the two cannot
drift). The bug is specific to **1-D convs with large input-channel count**
(`KW*IC > 1024` while `KH*KW <= max_threads`); 2-D convs use `KW*KH` on both
sides and stay consistent. Unnoticed upstream because LLM inference does not use
`im2col`; conv-heavy models (whisper, vision graphs) hit it.

Note on a red herring: llama's `ggml-metal-device.m` also widens a
`supports_op` check to accept `GGML_TYPE_F16`, but that case is `GGML_OP_GLU`
(LLM gated-activation), which whisper never uses. It is not involved.

## Why a single shared ggml cannot be used for whisper here

ggml (github.com/ggml-org/ggml) offers no ABI or behavioral stability
guarantee between commits, especially for backend kernels. Both llama.cpp and
whisper.cpp vendor independently-synced snapshots that drift. A newer,
nominally-superset ggml is not automatically usable by another consumer:
`im2col_ext` is a behavioral regression for whisper's workload that is not an
API change and that llama's own test surface never exercises (LLMs do not
convolve). Building whisper against llama's ggml does not help either -- the
defect is in llama's Metal kernel path, not in the ABI.

Approaches considered and rejected:

- Bump whisper.cpp (tried `v1.8.4` -> `v1.8.6`): no effect; the defect is in
  the *linked* ggml, not whisper's version.
- Build whisper against llama's ggml (`WHISPER_USE_SYSTEM_GGML=ON` +
  `find_package(ggml)` pointing at llama's build): configures and compiles, but
  still zero segments on Metal -- because it still runs llama's broken
  `im2col_ext` kernel.
- Disable ggml-metal optimizations via env (`GGML_METAL_FUSION_DISABLE`,
  `..._CONCURRENCY_DISABLE`, `..._GRAPH_OPTIMIZE_DISABLE`, `..._BF16_DISABLE`,
  `..._SHARED_BUFFERS_DISABLE`, `GGML_METAL_NO_RESIDENCY`): no combination
  helps; the selection is not gated by any of these.

## Fix

`CMakeLists.txt`: `_whisper_native` links whisper's own ggml stack from
`thirdparty/whisper.cpp/lib` (`libggml{,-base,-cpu,-blas,-metal}.a` +
`libwhisper.a`) instead of llama's `STATIC_LIBS` ggml. The whisper libs are
also removed from the global `STATIC_LIBS` so no single extension links two
ggml copies. Applied to both static link branches (macOS and Linux). The
whisper translation unit references zero llama symbols, verified by a clean
relink with no undefined symbols.

Validation:

- GPU transcription correct; `.so` rpath is whisper-only.
- whisper suite: 54 passed, 9 skipped (incl. the new gate).
- regression (chat + llama core + sd): 186 passed, 2 skipped.
- full `make test`: 1915 passed, 63 skipped.

## Known gap: dynamic-link (`WITH_DYLIB`) build

The `WITH_DYLIB` branch still links whisper against llama's ggml dylibs and
carries the SAME latent defect (documented with a WARNING in `CMakeLists.txt`).
A correct dynamic build must build and bundle whisper's own ggml dylibs and
rpath whisper to them -- a `scripts/manage.py` change. The default static build
is correct; dynamic is not yet fixed.

## Release gate

`tests/test_whisper_gpu_parity.py` transcribes on the GPU backend and asserts
the result is non-empty, contains the canonical JFK phrases, and matches CPU.
The empty-GPU-transcript is this bug's exact signature, so the gate fails on
any pin combination or build regression where the active ggml-metal
miscomputes whisper's encoder. It is benign on CPU-only hosts (GPU falls back
to CPU). If this gate is red, do not release with GPU enabled for whisper on
that platform/pin combination.

## Upstream report angle

Reportable to ggml/whisper.cpp as: `kernel_im2col_ext` (the large-`im2col`
Metal path, `ne00*ne01 > 1024`) produces incorrect output for whisper-style
convolution on Apple Metal in recent ggml; the base `kernel_im2col` is
correct. Repro: run whisper base.en on Metal with a ggml snapshot that selects
`im2col_ext`; encoder yields zero segments.

## Draft upstream issue

Target repo: `ggml-org/ggml` (the kernel lives in `src/ggml-metal/`).
Reproducible through whisper.cpp. Paste-ready below.

---

**Title:** ggml-metal: `kernel_im2col_ext` produces incorrect results for large
`im2col` on Apple Metal (whisper encoder yields empty output)

**Environment**
- Hardware: Apple M1 (MTLGPUFamilyApple7), macOS (Darwin 24.6), Metal backend.
- ggml: version reported `0.13.1`, the snapshot vendored in llama.cpp `b9528`.
  The same ggml version vendored in whisper.cpp `v1.8.6` does NOT exhibit it
  (it predates the change below).
- Build: `GGML_METAL=ON`, `GGML_METAL_EMBED_LIBRARY=ON`, `GGML_BLAS=ON`,
  static.

**Summary**

The Metal `im2col` kernel and its dispatch geometry are selected by two
*different* criteria that can disagree, and when they do, the `_ext` kernel runs
under the base kernel's threadgroup/grid layout and writes garbage:

- `ggml-metal-device.cpp` selects the **kernel**: `kernel_im2col_ext` when
  `ne00*ne01 > 1024` (`ne00,ne01` = `op->src[0]` dims = `KW * IC` for a 1-D
  conv).
- `ggml-metal-ops.cpp` selects the **dispatch geometry**: the `_ext` layout
  only when `KH*KW > max_threads_per_threadgroup`, otherwise the base layout.

`KW*IC` and `KH*KW` differ, so a 1-D conv with large input-channel count picks
the `_ext` *kernel* but the *base* geometry. The `_ext` kernel then mis-indexes.

With a real model this is silent: whisper.cpp's audio encoder does a 1-D conv
of shape `KW=3, IC=512` (`ne00*ne01 = 1536 > 1024`, but `KH*KW = 3`), so it hits
exactly this mismatch -> all-wrong activations -> decoder emits no speech ->
`whisper_full` returns `rc=0` with zero segments. The CPU backend is correct for
the identical graph. (whisper's first conv, `IC=80`, stays consistent and is
fine; only the large-`IC` conv breaks.) 2-D convs are unaffected because both
sites use `KW*KH`.

**Steps to reproduce**
1. Build whisper.cpp's `whisper-cli` (or any whisper integration) against a
   ggml at/after the commit that added the size-based `im2col` kernel selection
   in `ggml/src/ggml-metal/ggml-metal-device.cpp`.
2. Run `base.en` on `samples/jfk.wav` with the Metal backend (`use_gpu=true`).

**Expected:** transcript "And so my fellow Americans, ask not what your country
can do for you...".
**Actual (Metal):** `whisper_full` returns 0, `whisper_full_n_segments() == 0`,
empty transcript. **CPU backend:** correct.

**Root cause (bisected and verified)**

Two sites disagree on base-vs-`_ext`:

`ggml-metal-device.cpp` (`ggml_metal_library_get_pipeline_im2col`) -- selects
the kernel:

```c
GGML_TENSOR_LOCALS(int64_t, ne0, op->src[0], ne);   // ne00=KW, ne01=IC (1-D)
if (ne00*ne01 <= 1024) snprintf(base, ... "kernel_im2col_%s" ...);
else                   snprintf(base, ... "kernel_im2col_ext_%s" ...);
```

`ggml-metal-ops.cpp` (`ggml_metal_op_im2col`) -- selects the geometry:

```c
if (KH*KW <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
    // base layout
    ggml_metal_encoder_dispatch_threadgroups(enc, IC, OH, OW, ntptg0, KH, KW);
} else {
    // _ext layout
    ggml_metal_encoder_dispatch_threadgroups(enc, quotient*CHW, OH, OW, n_threads, 1, 1);
}
```

The selection key `ne00*ne01` (= `KW*IC`) and the geometry key `KH*KW` are
different quantities. For whisper's conv2 (`KW=3, IC=512, KH=1`):
`ne00*ne01 = 1536 > 1024` selects `kernel_im2col_ext`, but `KH*KW = 3 <= 1024`
emits the base geometry. The `_ext` kernel (which indexes via `tgpig[0]/CHW`,
`tgpig[0]%CHW`, expecting the `quotient*CHW` grid) then runs under the base grid
and mis-indexes -> garbage.

Confirmed by two independent fixes, each verified by rebuilding ggml and
re-running on Metal:

1. Force the base kernel in `device.cpp` -> base kernel + base geometry -> works.
2. Change `ops.cpp`'s geometry condition to the same `ne00*ne01 <= 1024` used by
   `device.cpp` -> `_ext` kernel + `_ext` geometry -> works.

So the `_ext` kernel is fine; the defect is the inconsistent selection. The
upstream-correct fix is (2): make the geometry follow the actually-selected
pipeline (or share one predicate) so the two cannot drift. The bug is specific
to 1-D convolutions with large input-channel count (`KW*IC > 1024` while
`KH*KW <= max_threads`); 2-D convs use `KW*KH` on both sides and are unaffected.
Unnoticed upstream because LLM inference does not use `im2col`.

Bisection notes: swapping only the shader file (`ggml-metal.metal`) or only
`ggml-metal-ops.cpp` between the working/broken snapshots does NOT reproduce it;
the trigger is the `device.cpp` `_ext` selection interacting with the
unchanged `ops.cpp` geometry predicate.

**Proposed fix**

Make the dispatch geometry follow the *selected kernel* so the two predicates
can never disagree.

Robust (recommended): have the geometry in `ggml_metal_op_im2col` key off which
pipeline was actually selected, rather than recomputing a separate condition.
e.g. expose the choice from the selection helper:

```c
// ggml-metal-device.{h,cpp}: report which variant was chosen
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col(
        ggml_metal_library_t lib, const ggml_tensor * op, bool * is_ext /*out*/);

// ggml-metal-ops.cpp:
bool is_ext = false;
auto pipeline = ggml_metal_library_get_pipeline_im2col(lib, op, &is_ext);
if (!is_ext) {
    ggml_metal_encoder_dispatch_threadgroups(enc, IC, OH, OW, ntptg0, KH, KW);
} else {
    const uint64_t n_threads = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (uint64_t) N);
    const int64_t  quotient  = N / n_threads + (N % n_threads > 0 ? 1 : 0);
    ggml_metal_encoder_dispatch_threadgroups(enc, quotient * CHW, OH, OW, n_threads, 1, 1);
}
```

Minimal (verified to fix it, but duplicates the literal `1024`): change the
geometry condition in `ggml_metal_op_im2col` to the same predicate the kernel
selection uses:

```c
-    if (KH*KW <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
+    if (ne00*ne01 <= 1024) {  // must match ggml_metal_library_get_pipeline_im2col
```

(`ne00,ne01` are already in scope via `GGML_TENSOR_LOCALS(..., op->src[0], ne)`.)

Open question for maintainers, which decides the canonical single predicate:
is `kernel_im2col_ext` intended as a *correctness fallback* for
`KH*KW > max_threads_per_threadgroup` (the original `ops.cpp` intent), or a
*performance path* for large channel counts (`ne00*ne01 > 1024`, the current
`device.cpp` intent)? Either way, both sites must use that one predicate.

**Suggested minimal check**

A ggml-level regression test comparing `GGML_OP_IM2COL` output on Metal vs CPU
for a 1-D conv that hits the mismatch -- `ne00*ne01 > 1024` (kernel selects
`_ext`) while `KH*KW <= max_threads_per_threadgroup` (geometry stays base),
e.g. `KW=3, IC=512, KH=1` (whisper's conv2), F16 and F32 -- would catch this
directly, independent of any model.

**Note (unrelated observation):** the same `device.cpp`/`device.m` delta also
widens a `supports_op` check to accept `GGML_TYPE_F16` for `GGML_OP_GLU`; that
is a separate, unrelated change and not involved in this bug.
