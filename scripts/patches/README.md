# Patches

Proposed upstream patches for vendored C++ dependencies. These are not currently applied locally — inferna handles the affected error cases in its nanobind wrapper layer instead.

## stable-diffusion.cpp

**Target:** commit `545fac4` (tag `master-537-545fac4`)

**Upstream issue:** https://github.com/leejet/stable-diffusion.cpp/issues/1367

**Problem:** `alloc_params_buffer()` in `GGMLRunner` (ggml_extend.hpp) returns `bool`, but all wrapper classes in `DiffusionModel`, `Conditioner`, `T5Embedder`, and `LLM` declare their overrides as `void`, discarding the return value. The call sites in `stable-diffusion.cpp` also never check the result. When allocation fails (e.g. CUDA out of memory), execution silently continues with unallocated tensors, producing garbage output.

**Current inferna workaround:** The nanobind wrapper (`src/inferna/sd/_sd_native.cpp` + `src/inferna/sd/stable_diffusion.py`) validates each generated `SDImage.is_valid` and raises `RuntimeError` when all images have invalid data.
