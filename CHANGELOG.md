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

- `LlamaContext`, `WhisperContext`, and `SDContext` now share a uniform lifecycle surface: each exposes `close()` and `is_valid`. `LlamaContext` additionally has an internal `ensure_valid()` guard called by every method that dereferences its native pointer, so calls after `close()` raise `RuntimeError` instead of crashing the interpreter.
- `inferna::BusyGuard` (`src/inferna/common/busy_lock.hpp`): a small RAII helper for the per-context Python `threading.Lock` used by the whisper and sd wrappers. Replaces the duplicated `try { gil_release ... } catch (...) { release; throw; } release;` pattern in `_whisper_native.cpp` and `_sd_native.cpp`.

### Changed

- `LlamaModel.lora_adapter_init()` now raises `FileNotFoundError` (with a helpful message) when the LoRA path does not exist, instead of throwing `nb::python_error()` with no Python exception set.
- `set_preview_callback()` in `inferna.sd.stable_diffusion` now logs a warning when a user callback raises, matching the behavior of the progress and log callbacks. Previously the exception was silently swallowed.
- README "Build from source with a specific backend" now describes the mandatory two-phase install flow (`scripts/manage.py build --deps-only` followed by `pip install . --no-build-isolation`) and notes that `pip install inferna --no-binary inferna` from sdist alone will not work because `thirdparty/*/lib/` is intentionally excluded.

### Removed

- Vestigial `nb::object params` field on `LlamaModelW` and `nb::object params_obj` field on `LlamaContextW`. Both were declared with "keep alive" comments but never assigned.

### Fixed

- `LlamaContext` post-close crash surface: `n_ctx`, `n_ctx_seq`, `n_batch`, `n_ubatch`, `n_seq_max`, `pooling_type`, `encode`, `decode`, `set_n_threads`, `n_threads`, `n_threads_batch`, `set_embeddings_mode`, `set_causal_attn`, `install_cancel_callback`, `synchronize`, `get_state_size`, `kv_cache_clear`, all `memory_seq_*`, `get_logits`, `get_logits_ith`, `get_embeddings`, `get_embeddings_ith`, `get_perf_data`, `print_perf_data`, and `reset_perf_data` now raise `RuntimeError` if invoked after `close()` instead of dereferencing a null pointer.

## [0.1.1]

### Fixed

- Fixed `AttributeError: 'LlamaContext' object has no attribute 'params'` in `inferna chat` by storing the originally-constructed `LlamaContextParams` on the chat object and reusing it when creating a fresh context per generation.
- Fixed sqlite-vector extension lookup in editable installs: `SqliteVectorStore.EXTENSION_PATH` now searches every entry in the `inferna.rag` package `__path__`, so the `vector.{dylib,so,dll}` artifact is found whether it lives next to the source tree or in the scikit-build-core editable mirror under site-packages.

## [0.1.0]

### Added

- Created inferna, a nanobind rewrite of [cyllama](https://github.com/shakfu/cyllama) v0.2.14.
