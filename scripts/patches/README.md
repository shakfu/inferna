# Patches

Local fixes to the vendored C++ dependencies, applied to the cloned source tree
before every build by `Builder._apply_source_patches()` in
`scripts/manage.py`.

Two globs are applied, in this order:

- `ggml-*.patch` — fixes to upstream ggml, applied to llama.cpp's and
  whisper.cpp's trees. Not applied to stable-diffusion.cpp: shared mode compiles
  llama.cpp's patched tree, and vendored mode compiles leejet's fork, whose
  layout they do not match.
- `<project>-*.patch` — fixes specific to one upstream, matched on the builder
  name (`llama.cpp-*`, `whisper.cpp-*`, `stable-diffusion.cpp-*`,
  `sqlite-vector-*`).

Each patch is applied with `git apply -p1`. An already-applied patch is
skipped. A patch that no longer applies fails the build with git's reason:
rebase it, or delete it if upstream merged an equivalent fix. `make reset` / `make remake` wipe the trees, so these run on every build.

The `.patch` files are the single source of truth and double as the upstream PR
payload; each carries its own rationale in a header above the diff.

## Applied

| Patch | Trees | What it fixes |
|-|-|-|
| `ggml-metal-pin-msl-version-set-lang.patch` | llama.cpp (v0.4.0+), whisper.cpp (v1.9.4+) | Metal shader compilation depending on the host process's SDK rather than the running OS |
| `stable-diffusion.cpp-diag-mask-inf-removed.patch` | sd.cpp (`master-1094-3bad892`, shared ggml at `b12359`) | compile failure: the MiniMax H3 audio VAE calls `ggml_diag_mask_inf`, which `b12359` removed |
| `stable-diffusion.cpp-graph-view-unexported.patch` | sd.cpp (`master-1094-3bad892`, shared ggml) | LNK2019 in dynamic Windows builds: the compat layer calls `ggml_graph_view`, which `ggml-base.dll` does not export |
| `sqlite-vector-inferna.patch` | sqlite-vector (`1.2.0`) | four build failures on manylinux and MSVC, each marked `inferna:` in the source |

In shared-ggml mode sd.cpp compiles llama.cpp's ggml tree in place
(`SD_GGML_SOURCE_DIR`), so it gets that tree's MSL pin. sd.cpp's vendored ggml
predates `ggml_metal_compile_options_set_lang()`, so `-set-lang` does not match
it. An `SD_USE_VENDORED_GGML=1` Metal build therefore ships sd.cpp without the
MSL pin.

`diag-mask-inf-removed` builds the causal mask as
`ggml_tri(ggml_fill(zeros, -inf), UPPER)`. Both forms write -inf where
`i0 > i1` and 0 elsewhere. The CPU and Metal `tri` kernels select rather than
multiply, so there is no `-inf * 0` NaN. Both ggml trees have `ggml_tri` and
`ggml_fill`, so the patch is unconditional. No test exercises this VAE.

`graph-view-unexported` builds the graph view in place from the public
`ggml_cgraph` fields, setting the same fields `ggml_graph_view` sets. Its only
caller, `sd_backend_graph_compute_with_eval_callback`, has no callers at this
pin, so the change affects linking only. Static builds and ELF/Mach-O shared
builds resolve the internal symbol.

`sqlite-vector-inferna.patch` replaces the inline source edits that
`SqliteVectorBuilder` used to make:

- glibc declares `strcasestr` only under `_GNU_SOURCE`, so manylinux (AlmaLinux 8) fails.
- MSVC has no `strcasecmp` / `strncasecmp`.
- MSVC has no `<cpuid.h>`; `__cpuidex` and `_xgetbv` come from `<intrin.h>`.
- MSVC has no `__builtin_popcount`, used in the AVX2 Hamming tail. Upstream
  guards the same loop in `distance-sse2.c`.

The inline edits matched source text and skipped silently when it moved. The
patch fails the build instead. `SqliteVectorBuilder` applies it to
`build/sqlite-vector` before staging into `thirdparty/sqlite-vector/`, which
the wheel compiles. `tests/test_sqlite_vector_msvc.py` fails on any staged
`__builtin_*` call outside a `__GNUC__` / `__clang__` guard.

`stable-diffusion.cpp-msvc-bigobj.patch` was dropped at `master-1094-3bad892`:
upstream adds `/bigobj` to the SD library target.

Two stable-diffusion.cpp patches were dropped at `master-898-2bb7294`, both
fixed upstream:

- `graph-cut-budget-clamp`: the budgeted plan it clamped is gone. Segmentation
  is now decided per run against live free VRAM
  ([#1905](https://github.com/leejet/stable-diffusion.cpp/pull/1905),
  [#1940](https://github.com/leejet/stable-diffusion.cpp/pull/1940)).
- `conditioner-compute-failure`: the LLM conditioner logs and returns an empty
  condition instead of asserting, and the pipeline turns it into a failed
  generation ([#1958](https://github.com/leejet/stable-diffusion.cpp/pull/1958),
  [#1973](https://github.com/leejet/stable-diffusion.cpp/pull/1973),
  [#2020](https://github.com/leejet/stable-diffusion.cpp/pull/2020)).

A patch that stopped matching used to be skipped silently, which is how the
v0.4.0 bump removed the MSL pin from the llama.cpp tree without failing a build
or a test. It now fails the build.

## Handled in the wrapper instead

**`alloc_params_buffer()` discards its return value** —
<https://github.com/leejet/stable-diffusion.cpp/issues/1367>. It returns `bool`
in `GGMLRunner` (`ggml_extend.hpp`), but the overrides in `DiffusionModel`,
`Conditioner`, `T5Embedder` and `LLM` declare `void`, and the call sites in
`stable-diffusion.cpp` never check. An allocation failure (e.g. CUDA OOM)
continues with unallocated tensors and produces garbage. `_sd_native.cpp` and
`stable_diffusion.py` validate `SDImage.is_valid` per image and raise
`RuntimeError` when every image is invalid.
