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
| `ggml-metal-pin-msl-version.patch` | whisper.cpp, sd.cpp (vendored ggml) | Metal shader compilation depending on the host process's SDK rather than the running OS |
| `ggml-metal-pin-msl-version-perkind.patch` | llama.cpp (v0.3.0 per-kind library layout) | Same fix, rebased onto the restructured `ggml_metal_library_compile_all` |
| `stable-diffusion.cpp-conditioner-compute-failure.patch` | sd.cpp | A failed text-encoder graph aborting the interpreter on `GGML_ASSERT` instead of raising |
| `stable-diffusion.cpp-graph-cut-budget-clamp.patch` | sd.cpp | `--max-vram` budgets ignoring VRAM already in use |
| `stable-diffusion.cpp-msvc-bigobj.patch` | sd.cpp | `C1128: number of sections exceeded object file format limit` on MSVC |

The two MSL-version patches are alternatives, not a pair: llama.cpp's ggml
restructured its Metal library loading in v0.3.0, so each tree matches exactly
one and skips the other.

## Handled in the wrapper instead

**`alloc_params_buffer()` discards its return value** —
<https://github.com/leejet/stable-diffusion.cpp/issues/1367>. It returns `bool`
in `GGMLRunner` (`ggml_extend.hpp`), but the overrides in `DiffusionModel`,
`Conditioner`, `T5Embedder` and `LLM` declare `void`, and the call sites in
`stable-diffusion.cpp` never check. An allocation failure (e.g. CUDA OOM)
continues with unallocated tensors and produces garbage. `_sd_native.cpp` and
`stable_diffusion.py` validate `SDImage.is_valid` per image and raise
`RuntimeError` when every image is invalid.
