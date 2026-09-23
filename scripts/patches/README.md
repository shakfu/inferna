# Patches

Local fixes to the vendored C++ dependencies, applied to the cloned source tree
before every build by `GgmlBuilder._apply_source_patches()` in
`scripts/manage.py`.

Two globs are applied, in this order:

- `ggml-*.patch` — fixes to the ggml copy that all three upstreams vendor.
  Tried against every tree.
- `<project>-*.patch` — fixes specific to one upstream, matched on the builder
  name (`llama.cpp-*`, `whisper.cpp-*`, `stable-diffusion.cpp-*`).

Each patch is applied with `git apply -p1` and is idempotent and self-disabling:
already applied, or no longer applying (upstream merged an equivalent fix, or
refactored the context), are both logged and skipped rather than failing the
build. `make reset` / `make remake` wipe the trees, so these run on every build.

The `.patch` files are the single source of truth and double as the upstream PR
payload; each carries its own rationale in a header above the diff.

## Applied

| Patch | Trees | What it fixes |
|-|-|-|
| `ggml-metal-pin-msl-version-set-lang.patch` | llama.cpp (v0.4.0+), whisper.cpp (v1.9.4+) | Metal shader compilation depending on the host process's SDK rather than the running OS |
| `stable-diffusion.cpp-msvc-bigobj.patch` | sd.cpp | `C1128: number of sections exceeded object file format limit` on MSVC |

In shared-ggml mode sd.cpp compiles llama.cpp's ggml tree in place
(`SD_GGML_SOURCE_DIR`), so it gets that tree's MSL pin. sd.cpp's vendored ggml
predates `ggml_metal_compile_options_set_lang()`, so `-set-lang` does not match
it. An `SD_USE_VENDORED_GGML=1` Metal build therefore ships sd.cpp without the
MSL pin.

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

A patch that stops matching is skipped silently, by design -- which is how the
v0.4.0 bump removed the MSL pin from the llama.cpp tree without failing a build
or a test. When bumping a pin, check each patch still lands:

    for p in scripts/patches/ggml-*.patch; do
        git -C build/llama.cpp apply --reverse --check "$p" && echo "applied: $p"
    done

## Handled in the wrapper instead

**`alloc_params_buffer()` discards its return value** —
<https://github.com/leejet/stable-diffusion.cpp/issues/1367>. It returns `bool`
in `GGMLRunner` (`ggml_extend.hpp`), but the overrides in `DiffusionModel`,
`Conditioner`, `T5Embedder` and `LLM` declare `void`, and the call sites in
`stable-diffusion.cpp` never check. An allocation failure (e.g. CUDA OOM)
continues with unallocated tensors and produces garbage. `_sd_native.cpp` and
`stable_diffusion.py` validate `SDImage.is_valid` per image and raise
`RuntimeError` when every image is invalid.
