# TODO

## Bugs

- [ ] **Ctrl-C does not interrupt `inferna.sd` generation** ([#8](https://github.com/shakfu/inferna/issues/8)) -- `generate_image()` is a single blocking C call with no abort path; the LLM cancellation work in `0.2.14` does not transfer (separate compute graph). Fix is gated on upstream PR [leejet/stable-diffusion.cpp#1124](https://github.com/leejet/stable-diffusion.cpp/pull/1124) which adds `sd_cancel_generation(sd_ctx, sd_cancel_mode_t)`. Once that merges and `--sd-version` bumps to a release containing it: extend `stable_diffusion.pxd` with the enum + extern, add `SDContext.cancel(mode)` and `SDContext.install_sigint_handler()` mirroring the LLM helpers (`src/inferna/api.py` `_SigintHandle`), wire into the inferna-desktop sidecar's `asyncio.CancelledError` path. Tests in `tests/test_sd_cancel.py` modeled on `tests/test_cancel.py`.

## Medium Priority

- [ ] Performance regression detection -- CI-integrated baseline capture/comparison to catch speed or memory regressions across commits

- [ ] Structured logging system (JSON output option, agent decision flow logging)

## Wheel / Packaging

- [ ] stable-diffusion.cpp uses compile-time `#ifdef SD_USE_CUDA` for backend selection instead of dynamic `ggml_backend_load_all()` like llama.cpp and whisper.cpp -- propose dynamic backend discovery upstream or patch locally for consistency

## Explore

- [ ] MCP server (`inferna/mcp/`): expose local inference (`complete`, `chat`, `embed`, `transcribe`, `generate_image`) as MCP tools and model listing as resources. Two transports: stdio entrypoint for subprocess clients (Claude Desktop), and Streamable-HTTP routes mounted on `EmbeddedServer` (`src/inferna/llama/server/embedded.pyx`) for Claude Code / remote clients. Reuse `agents/jsonrpc.py` framing and the high-level API in `src/inferna/api.py` -- no new heavy deps. (Client side already shipped: `LLM.add_mcp_server()` in `src/inferna/api.py:1378` wraps `agents/mcp.py` for non-agent callers.)

## CI / Workflows

### Medium Priority

- [ ] **Lightweight Python lint / type-check workflows** -- llama.cpp has `python-lint.yml`, `python-type-check.yml`, `python-check-requirements.yml`, `editorconfig.yml` using `runs-on: ubuntu-slim`, triggered only on `**/*.py` / config path changes, running in <1 min. Cheap pre-filter before the 40-minute wheel matrix. Add `.github/workflows/python-lint.yml` with `ruff check` and optionally `mypy` / `ty`

- [ ] **Composite actions for repeated toolchain setup** -- llama.cpp factors into `.github/actions/{windows-setup-cuda,linux-setup-vulkan,windows-setup-rocm,unarchive-tar,get-tag-name}/action.yml` and reuses them across `build-vulkan.yml`, `release.yml`, `build-cache.yml`. inferna duplicates the Vulkan-SDK pwsh install (~15 lines) and a version-reading Python snippet (`build-cibw.yml:234-239`, `build-gpu-wheels.yml:664-670`). Extract `.github/actions/setup-vulkan-windows` and `.github/actions/get-version`

- [ ] **Reusable `workflow_call` smoke-test** -- inferna's wheel-find + venv + import + inference block is duplicated across all three workflow files with minor variations. Wrap `scripts/run_wheel_test.py` in `.github/workflows/_smoke-test.yml` with `on: workflow_call` (inputs: `artifact-name`, `runs-on`, `run-inference`) and delete ~200 lines of duplication

### Wheel Coverage (additional backend variants)

Gap analysis vs. llama.cpp b8893 release assets. Ordered by effort/payoff.

- [ ] **Windows SYCL (Intel Arc + Xe)** -- Linux SYCL is shipped (`build_sycl` in `build-gpu-wheels-abi3.yml:319`, wheel name `inferna-sycl`). Windows SYCL still pending: follows the same pattern as windows-cuda/vulkan -- download prebuilt, synthesize `.lib` via existing `_generate_import_libs()`, `delvewheel --include ggml-sycl.dll` with `--no-dll` for `sycl[78].dll`, `pi_level_zero.dll`, `pi_opencl.dll`, `svml_dispmd.dll`, `libmmd.dll`, `libiomp5md.dll` (user-installed Intel oneAPI runtime). Build-time dep: Intel oneAPI DPC++ on the Windows runner -- use `oneapi-src/setup-oneapi` or similar. Needs `_release_asset_name()` + `_dylib_names` extended to recognize Windows SYCL assets

- [ ] **Windows HIP Radeon (AMD GPUs)** -- upstream ships `llama-b8893-bin-win-hip-radeon-x64.zip`. Same download+synthesize+delvewheel pattern; `--include ggml-hip.dll`, `--no-dll` for `amdhip64_6.dll`, `hipblas.dll`, `rocblas.dll`, `amd_comgr_*.dll` (user-installed AMD HIP SDK / Adrenalin runtime). Main obstacle: AMD HIP SDK Windows install on CI has no compact GitHub Action -- needs manual `Invoke-WebRequest` of AMD's installer (~2-3 GB) plus silent-install args, or a `choco install` package if one exists. Highest effort of the three Windows GPU gaps

- [ ] **Linux ROCm 7.2 prebuilt** -- upstream now ships `llama-b8893-bin-ubuntu-rocm-7.2-x64.tar.gz`. Current `build_rocm` job compiles ROCm 6.3 from source (20-40 min); switching to the prebuilt would cut CI time dramatically. Tradeoff: constrained to upstream's arch list (we currently target `gfx90a;gfx942;gfx1100` explicitly). Evaluate whether upstream's default architectures are acceptable before committing

- [ ] **ARM64 variants** -- growing relevance (Copilot+ PCs, Ampere/Graviton clouds, Apple Silicon KleidiAI). Currently commented out in `build-cibw.yml` for `ubuntu-24.04-arm` and `windows-11-arm`; upstream ships `ubuntu-arm64`, `ubuntu-vulkan-arm64`, `win-cpu-arm64`, `macos-arm64-kleidiai`. Needs its own wheel variant names and separate investigation of build-time toolchain availability on ARM runners

- [ ] **Linux OpenVINO (Intel CPU accelerator)** -- upstream ships `llama-b8893-bin-ubuntu-openvino-2026.0-x64.tar.gz` as a new backend. Would require inferna-side integration work (build flags, runtime loader, backend detection in `build_config.json`) on top of the wheel packaging. Lower priority until there's user demand

### Lower Priority

- [ ] **Separate SDK caches from `thirdparty/` build-artifact caches** -- llama.cpp's `build-cache.yml` caches SDK install dirs (`C:\Program Files\AMD\ROCm`, `./vulkan_sdk`, `./openvino_toolkit`) under keys scoped to `HIPSDK_INSTALLER_VERSION`, separate from compile caches. inferna's single `deps-<backend>-<linkmode>` key mixes SDK install with built artifacts, so a `manage.py` edit invalidates cached SDK binaries too. For `build-gpu-wheels.yml`: consider `--manylinux-image` with pre-built CUDA/ROCm image, or a container-volume trick to cache `/usr/local/cuda`

- [ ] **Monotonic `b<commit-count>` build-tag scheme** -- llama.cpp's `get-tag-name/action.yml` uses `fetch-depth: 0` + `git rev-list --count HEAD` to produce `b${BUILD_NUMBER}` on `master`, `${branch}-b${count}-${sha7}` on branches. Lets every CI-built wheel be distinguishable without bumping `pyproject.toml` version on every pre-release. Apply in `build-cibw.yml` / `build-gpu-wheels.yml` upload steps

## Wrapper Layer (from REVIEW)

Items distilled from a 2026-04 wrapper-code review. Two HIGH correctness bugs (`Speculative.is_compat` KV clobber, SD preview-callback UAF) and one latent `EmbeddedServer.start()` cleanup leak were fixed in-session; this list is what remained after independent re-verification (some original findings were dropped as either reversed or benign).

### Hardening — small bugs & ergonomic gaps

- [ ] **Add `BusyGuard` on `SDContext.close`** (or document "do not close while generating"). Currently `_sd_native.cpp:789-791` calls `free_sd_ctx` with no `busy_lock` — concurrent `generate_image` (GIL released) races against the free. `WhisperContextW::close` has the same shape; check both.

- [ ] **Make `WhisperContext.close` idempotent.** `_whisper_native.cpp:432-435` calls `s.ensure_valid()` before the null check, so a second close raises `RuntimeError` instead of being a no-op. Drop the `ensure_valid()` and gate cleanup on `s.ctx != nullptr`.

- [ ] **`PyErr_WriteUnraisable` in `_mongoose.cpp` HTTP handler.** `server/_mongoose.cpp:73-78` catches all Python handler exceptions and replies 500 with no logging. Surface them via `PyErr_WriteUnraisable` (or `PyErr_Print`) before the reply so handler bugs are visible instead of silently re-coded.

- [ ] **Document `Manager::send_reply` thread-affinity.** `server/_mongoose.cpp:114-131` walks `mgr.conns` while another thread may be inside GIL-released `poll`. `embedded.py` is single-threaded today but the constraint is undocumented. Add a docstring asserting "must be called from the same thread as `poll`", or guard with mongoose's wakeup primitive.

- [ ] **Replace per-token `LlamaBatch` alloc in `Speculative.draft`.** `_speculative.py:125, :144` constructs a fresh native batch each loop iteration; the project ships `BatchMemoryPool` in `_python_helpers.py` for exactly this. Pure perf, not correctness.

- [ ] **Better `from_numpy` error for non-2D/3D inputs.** `_sd_native.cpp:447-479` falls through to a bare nanobind cast error on 4D+ arrays. Add an explicit `if ndim not in (2, 3): raise ValueError(...)`.

- [ ] **Return `None` for failed slots in `generate_with_params`.** `_sd_native.cpp:825-844` wraps null-data results as stub `SDImage` objects mixed in with valid ones; only signal is a `warnings.warn`. Returning `None` for invalid slots makes `len(images) != batch` mean what it should.

- [ ] **Tidy `chat_apply_template` second-call buffer size.** `_llama_native.cpp:1000-1002` passes `required` (not `(int) buf.size()`) as the buffer size. Cosmetic — current behaviour is fine because the result is constructed with explicit length — but worth fixing.

### Coverage — bindings worth filling in

- [ ] **Bind `llama_set_adapters_lora` + `llama_set_adapter_cvec`.** Without these, `LlamaAdapterLora` is read-only — `lora_adapter_init` works but the loaded adapter cannot actually be applied to a context. (`_llama_native.cpp` near line 915.) Verified against `thirdparty/llama.cpp/include/llama.h`.

- [ ] **Bind the `whisper_*_with_state` family.** `_whisper_native.cpp:592-598` exposes `WhisperState` as a constructor-only stub. The whole point of `whisper_state` is concurrent decoding from one context; without methods (`whisper_full_with_state`, `whisper_encode_with_state`, etc.) the class is useless.

- [ ] **Bind `llama_state_*` save/load.** Only `llama_state_get_size` is bound (`_llama_native.cpp` near line 1188); `llama_state_get_data` / `_set_data` / `_load_file` / `_save_file` and the `_seq_*` and `_ext` variants are all unbound. Required for context checkpointing/resumption.

- [ ] **Bind whisper callbacks.** `whisper_full_params` exposes `new_segment_callback`, `progress_callback`, `encoder_begin_callback`, `abort_callback` — none are bound. SD module already has the analogous pattern; mirror it.

### Refactor — convention drift

- [ ] **Decide composition-vs-subclass for SD wrappers.** `SDImage` uses composition citing nanobind dealloc constraints (`stable_diffusion.py:156-158`); `SDContext`, `SDContextParams`, `SDSampleParams`, `SDImageGenParams` subclass the native types. Either the SDImage rationale applies to all of them or none. Pick one and apply uniformly.

- [ ] **Migrate enum exports to `nb::enum_` (or document why flat ints are required).** Currently every llama enum lives in three places — `llama.h`, `_llama_native_enums.cpp` flat exports, and `llama_cpp.py` re-aliases — so each new upstream enum needs three coordinated edits. The MTMD TU's `nb::enum_<MtmdInputChunkType>().value(...).export_values()` is the cleaner reference.

- [ ] **Add `tests/test_llama_native.py`.** Direct-surface coverage for symbols the integration tests skip: `LlamaModelKvOverride`/`TensorBuftOverride`, `GgmlBackend*` info, threadpool `attach`/`detach`, `chat_builtin_templates`, TTS helpers, `set_log_callback`.

### Pre-existing TU consolidation

- [ ] **Consolidate `LlamaBatch.set_batch` / `add_sequence` fill loops** in `src/inferna/llama/_llama_native.cpp`. Both methods duplicate the per-token `pos` / `seq_id` / `n_seq_id` / `logits` / `token` assignment; factor the inner loop into a single helper parameterized by starting offset and `seq_id`.

- [ ] **Validate ggml header consistency across translation units.** `_whisper_native.cpp` and `_sd_native.cpp` forward-declare ggml backend APIs while `_llama_native.cpp` includes the full ggml headers. Currently consistent, but check for cross-TU ABI drift on each upstream ggml header bump.

### Open observation (not yet a verified bug)

- [ ] **Investigate flaky `test_embedded_server_context_manager`.** One failure observed in a 1389-test run (`mg_listen` returned null on port 8097), passes cleanly on rerun. Root cause unverified — candidates include macOS TIME_WAIT residue, transient external interference, or an internal race across rapid `mg_mgr_init`/`mg_listen` cycles. Highest-value next move: log `errno`/`strerror(errno)` from the `_mongoose.cpp` listen path so the next flake produces a real signal instead of three guesses.

## RAG Scaling (see docs/dev/scaling_rag.md)

- [ ] Persistent quantization state in database metadata (quantize() exists but state is in-memory only)

- [ ] Metadata pre-filtering in vector search (filter by source, date, etc.)

- [ ] Async embedding generation (`embed_batch_async()`)

- [ ] Parallel document loading in DirectoryLoader

- [ ] Batch query processing in RAG pipeline

- [ ] Sharding for 1M+ vector workloads

