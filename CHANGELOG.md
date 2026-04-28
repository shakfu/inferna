# CHANGELOG

All notable project-wide changes will be documented in this file. Note that each subproject has its own CHANGELOG.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Commons Changelog](https://common-changelog.org). This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Types of Changes

- Added: for new features.
- Changed: for changes in existing functionality.
- Deprecated: for soon-to-be removed features.
- Removed: for now removed features.
- Fixed: for any bug fixes.
- Security: in case of vulnerabilities.

---

## [Unreleased]

### Added

- `LlamaContext`, `WhisperContext`, `SDContext`, and `MtmdContext` now share a uniform lifecycle surface: each exposes `close()` and `is_valid`, and each enforces an internal `ensure_valid()` guard at every method that dereferences the native pointer. Calls after `close()` raise `RuntimeError` instead of crashing the interpreter.
- `inferna::BusyGuard` (`src/inferna/common/busy_lock.hpp`): a small RAII helper for the per-context Python `threading.Lock` used by the whisper and sd wrappers. Replaces the duplicated `try { gil_release ... } catch (...) { release; throw; } release;` pattern in `_whisper_native.cpp` and `_sd_native.cpp`.
- `Manager.send_reply()` (`_mongoose`): validated reply primitive that walks the mongoose connection list and rejects stale or fabricated connection ids before dereferencing. Replaces the unvalidated module-level `_mg.send_reply(uintptr_t, ...)` free function.
- CI: `build-cibw.yml` smoke job now runs a fast `pytest` subset (`test_generate`, `test_chat`, `test_batching`, `test_context`, `test_params`, `test_model` — 106 tests) against the installed wheel on linux/macos-arm/windows after the inference smoke test. Previously CI exercised only imports plus a 16-token completion.

### Changed

- `LlamaVocab.tokenize()` now honors llama.cpp's negative-return contract: if the initial buffer is too small, it resizes to the required capacity (`-rc`) and retries once. Previously it capped the buffer at `min(text.size()*2 + 100, n_vocab)` — a nonsensical cap that produced spurious tokenization failures on long inputs and small-vocab models.
- `LlamaVocab.token_to_piece()` now decodes piece bytes with `errors="replace"` (via `PyUnicode_DecodeUTF8`). Byte-level BPE pieces frequently include partial UTF-8 sequences; the previous strict decode could raise `UnicodeDecodeError` from inside the hot generation loop on multilingual models. Callers continue to receive `str`.
- `_llama_progress_cb` now returns `false` (abort the load) when the user-supplied progress callback raises. Previously it returned `true`, so a `KeyboardInterrupt` or any handler exception was silently ignored and loading continued. This honors llama.cpp's "return false to abort" contract.
- Log/progress/preview callbacks across `_llama_native` and `_sd_native` now acquire the GIL *before* checking `is_valid()` / `is_none()` on the global callback handle. The prior pattern raced with `set_*_callback`, which mutates the `nb::object` under the GIL and could corrupt the handle on the worker thread.
- `MtmdContext.get_output_embeddings()` now returns a flat row-major `list[float]` of size `n_tokens * n_embd` (matching `LlamaContext.get_logits()`) rather than `list[list[float]]`. Removes ~2.4M `PyFloat` allocations per call for typical CLIP shapes (`n_tokens=576, n_embd=4096`).
- `SDContext.__exit__` now calls `self.close()` for deterministic teardown of the native context (and its GPU buffers) at `with`-block exit. Previously it was a no-op and the context lingered until GC, which on macOS Metal accumulated working-set pressure across consecutive contexts.
- `LlamaModel.lora_adapter_init()` now raises `FileNotFoundError` (with a helpful message) when the LoRA path does not exist, instead of throwing `nb::python_error()` with no Python exception set.
- `set_preview_callback()` in `inferna.sd.stable_diffusion` now logs a warning when a user callback raises, matching the behavior of the progress and log callbacks. Previously the exception was silently swallowed.
- `MACOSX_DEPLOYMENT_TARGET` is now consistently `11.0` across `Makefile`, `scripts/manage.py` (both the module-level default and `WheelBuilder.get_min_osx_ver`), and `pyproject.toml`'s cibuildwheel macOS section. Previously the four sites carried `14.7` / `12.6` / `10.9`-or-`11.0` / `11.0`, producing wheels with mismatched `LC_BUILD_VERSION` minimums depending on which build path was taken.
- `Makefile` `wheel-<backend>` and `wheel-<backend>-dynamic` targets now invoke `scripts/manage.py build --all --deps-only` (with the matching backend env) before `uv build --wheel`. Without this, a local `make wheel-cuda` on a Metal-built tree would silently produce a wheel claiming CUDA support but linking Metal-built archives. cibuildwheel's `before-build` hook already handled this in CI; local make did not.
- `wheel-opencl-dynamic` now sets `SD_USE_VENDORED_GGML=0` to match every other dynamic wheel target. Previously OpenCL dynamic builds linked SD against its vendored ggml while llama.cpp shipped a separate ggml dylib — exactly the ABI mismatch the `GGML_MAX_NAME=128` plumbing was added to prevent.
- `MongooseConnection` (`inferna.llama.server.embedded`) now carries a reference to the owning `Manager`. Reply paths route through `manager.send_reply()`, which validates the connection id is live before dereferencing.
- README "Build from source with a specific backend" now describes the mandatory two-phase install flow (`scripts/manage.py build --deps-only` followed by `pip install . --no-build-isolation`) and notes that `pip install inferna --no-binary inferna` from sdist alone will not work because `thirdparty/*/lib/` is intentionally excluded.
- `make test` now runs `pytest -s --durations=50 --durations-min=1.0` to surface the 50 slowest tests (≥1s) on every run, making test-suite drift visible in local development.
- `pyproject.toml` `cmake.version` floor bumped from `>=3.21` to `>=3.26` to match `CMakeLists.txt`'s `cmake_minimum_required(VERSION 3.26...3.30)`. Previously scikit-build-core could pick up CMake 3.21 from the build env and pass it to a project that demands 3.26+, immediately failing configure with a confusing error.
- `pyproject.toml` `wheel.exclude` no longer contains `*.lib`. On Windows, delvewheel relocates DLLs into `inferna.libs/` and may copy companion `.lib` import libraries; stripping `*.lib` would silently delete them.
- `[tool.cibuildwheel.macos]` now declares its own `before-all` (running `manage.py build --all --deps-only`) and disables the global `before-build`, mirroring the existing Linux strategy. Previously macOS rebuilt all third-party C++ deps once per Python version (5x for cp310-cp314), and re-cloned the upstream repos each time.
- `Makefile` library-detection now uses a platform-conditional `SHLIB_EXT` (`dylib` on Darwin, `so` on Linux, `dll` elsewhere). `wheel-dynamic` and the `LIBLAMMA` definition both pick the right extension instead of hard-coding `libllama.dylib`.
- `CMakeLists.txt` skips backend `find_package(... REQUIRED)` calls (CUDAToolkit, Vulkan, hip, hipblas, rocblas, SYCL runtime, OpenCL) when `WITH_DYLIB=ON`. The pre-built ggml-* dylibs already have those backends linked; the `_llama_native` extension itself does not reference their symbols. Previously a developer building only the wrapper extension against a pre-built llama.cpp release still needed the full Vulkan SDK (etc.) installed.

### Removed

- Vestigial `nb::object params` field on `LlamaModelW` and `nb::object params_obj` field on `LlamaContextW`. Both were declared with "keep alive" comments but never assigned.
- Unused `VERSION := 0.1.20` variable from `Makefile`. It was never substituted into any rule and never updated by `make bump`, so it could only mislead.

### Fixed

- `LlamaContext` post-close crash surface: `n_ctx`, `n_ctx_seq`, `n_batch`, `n_ubatch`, `n_seq_max`, `pooling_type`, `encode`, `decode`, `set_n_threads`, `n_threads`, `n_threads_batch`, `set_embeddings_mode`, `set_causal_attn`, `install_cancel_callback`, `synchronize`, `get_state_size`, `kv_cache_clear`, all `memory_seq_*`, `get_logits`, `get_logits_ith`, `get_embeddings`, `get_embeddings_ith`, `get_perf_data`, `print_perf_data`, and `reset_perf_data` now raise `RuntimeError` if invoked after `close()` instead of dereferencing a null pointer.
- `WhisperContext` post-close crash surface: every native accessor (`n_vocab`, `n_text_ctx`, `n_audio_ctx`, all `model_*`, all `token_*`, `tokenize`, `encode`, `full`, all `full_get_*`, `print_timings`, `reset_timings`, etc.) now raises `RuntimeError` after `close()` instead of segfaulting on a null `whisper_context*`. Previously `close()` set the pointer to `nullptr` but no method gate enforced the lifecycle.
- `MtmdContext` lifecycle: `tokenize`, `encode_chunk`, `eval_chunks`, `get_output_embeddings`, and the `supports_*` / `audio_sample_rate` / `uses_*` properties now route through `ensure_valid()` instead of scattered `if (!s.ptr) throw` checks (or, for the prop accessors, silently returning `false`/`-1` on a closed context). Behavior is consistent with the other three context types.
- `manage.py` no longer silently uses a stale source checkout when the requested version changes. Each `build()` entry point (llama.cpp static + dynamic, whisper.cpp, stable-diffusion.cpp, sqlite-vector) calls a new `verify_checkout()` helper that compares `git rev-parse HEAD` against `git rev-parse <self.version>` and aborts with a clear "run `make reset`" message on mismatch. Previously the guard was `if not src_dir.exists(): self.setup()`, which meant a partial checkout from a prior run would silently take precedence over the new version pin.
- `_mg.send_reply()` (the unvalidated module-level reply primitive) accepted any `uintptr_t` and dereferenced it directly. A stale connection id (kept past request scope) or a fabricated value would trigger undefined behavior on freed memory. Replaced by `Manager.send_reply()` which walks `mgr.conns` and rejects pointers not in the live list (and pointers whose connections are mid-close).

## [0.1.1]

### Fixed

- Fixed `AttributeError: 'LlamaContext' object has no attribute 'params'` in `inferna chat` by storing the originally-constructed `LlamaContextParams` on the chat object and reusing it when creating a fresh context per generation.
- Fixed sqlite-vector extension lookup in editable installs: `SqliteVectorStore.EXTENSION_PATH` now searches every entry in the `inferna.rag` package `__path__`, so the `vector.{dylib,so,dll}` artifact is found whether it lives next to the source tree or in the scikit-build-core editable mirror under site-packages.

## [0.1.0]

### Added

- Created inferna, a nanobind rewrite of [cyllama](https://github.com/shakfu/cyllama) v0.2.14.
