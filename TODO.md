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

- [x] ~~**Bind `llama_set_adapters_lora`**~~ — done. `LlamaContext.set_adapters_lora(adapters, scales)` bound; `LLM.load_lora` / `unload_lora` / `clear_loras` / `list_loras` exposed. `llama_set_adapter_cvec` (control vectors) still unbound — separate, lower priority.

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

## Wrapper Layer (REVIEW.md, 2026-05)

Items distilled from the second wrapper-code review (2026-05). Correctness fixes (UTF-8 streaming, Jinja `except`, `BatchMemoryPool`), the `LLM` god-class split into `ResponseCache` / `ChatTemplateRenderer` / `MCPFacade`, and the value-add features (LoRA, embeddings, structured outputs, function calling, logprobs, prompt cache / KV reuse, whisper streaming) all shipped in-session. This list is what remained.

### P1 — high impact

- [ ] **Continuous batching in `EmbeddedServer`.** Multi-day. Today's slot decode loop is sequential per request; a real per-slot continuous batching loop (one sampler per slot, ragged decode, true concurrent users) is the largest perf upgrade and the reason vLLM/llama-cpp-server outpace ad-hoc loops. Touches `src/inferna/llama/server/embedded.py`, `python.py:ServerSlot`, the streaming SSE path. Probably needs the `embedded.py` (~950 LOC) split first (see P3).

- [ ] **Vision in OpenAI-compat (`image_url` content parts).** Half-day. `integrations/openai_compat.py` does not handle `messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "..."}}]}]`. The MTMD context exists (`inferna.llama.mtmd`); plumb it: detect image content parts in `_create_completion`, encode via MTMD, build the multimodal prompt. Pair with a high-level `LLM.chat_with_images(messages, images)` convenience.

- [ ] **Expose KV-cache control on `LLM`.** Half-day. Native bindings exist (`memory_seq_rm`, `memory_seq_pos_max`, `memory_seq_cp`, `memory_seq_keep`, `memory_seq_add`); `LLM` only uses them internally for the prompt-cache layer. Public surface (`LLM.kv.drop`, `LLM.kv.copy`, `LLM.kv.pos_max`) unblocks user-built prompt caches, multi-conversation slot reuse, and the next round of agent-style work.

### P2 — useful features, half-day each

- [x] **Sampler parameters.** ~~Add `frequency_penalty: float = 0.0` and `presence_penalty: float = 0.0` to `GenerationConfig`...~~ Done, and expanded scope: `GenerationConfig` now exposes `frequency_penalty`, `presence_penalty`, `penalty_last_n`, `mirostat`/`mirostat_tau`/`mirostat_eta`, `typical_p`/`typical_min_keep`, `xtc_probability`/`xtc_threshold`, `dynatemp_range`/`dynatemp_exponent`, and `logit_bias`. Wired through `LLM._create_sampler` (`src/inferna/api.py`), `BatchGenerator.generate_batch` (`src/inferna/batching.py`), and `Chat.__init__` (`src/inferna/llama/chat.py`). See CHANGELOG.

- [ ] **Rebuild `cli.py` sampler chain.** `src/inferna/llama/cli.py` registers ~50 llama.cpp-compatible sampling flags (`--top-k`, `--top-p`, `--min-p`, `--repeat-penalty`, `--mirostat`/`--mirostat-ent`/`--mirostat-lr`, `--logit-bias`, `--temp`, `--seed`, …) and then ignores all of them at line 398 in favour of a hardcoded `self.sampler.add_greedy()` (the comment "start with greedy for simplicity" makes the gap explicit). Right fix: extract a shared `_internal/sampler_build.py::build_sampler(config, vocab)` helper that both `LLM._create_sampler` and the CLI consume, then replace the CLI's hardcoded greedy with a `build_sampler()` call driven from the parsed flags. Also decide `--logit-bias`'s string format (currently `type=str, default=""` and unparsed) — `id+bias,id+bias,...` per llama.cpp upstream, or JSON for OpenAI parity. Add CLI integration test per flag. Half-to-full day.

- [ ] **Speculative decoding ergonomics.** `_speculative.py` exists and is functional but agents/users have to hand-wire it. Add `LLM.enable_speculative(draft_model_path, n_draft=8)` convenience. Optional sibling-finder helper that resolves a smaller HF variant for a given main model. Print acceptance-rate stats on close so users can tune `n_draft`.

- [ ] **Whisper VAD convenience.** `WhisperVadParams` is bound but no Python helper for "transcribe-with-VAD-segmentation". Build `transcribe_with_vad(audio, ...)` that pre-segments via VAD and stitches segment outputs. Sits next to `WhisperStreamer` in `whisper/streaming.py` (or sibling `vad.py`).

- [ ] **`logprobs` in OpenAI-compat.** Now that `LLM(..., logprobs=True)` populates `Response.logprobs`, surface this through `integrations/openai_compat.py` so `client.chat.completions.create(logprobs=True, top_logprobs=K)` round-trips.

- [ ] **Streaming SSE chat completions in `EmbeddedServer` — verify wire format.** `openai_compat.py` (the client adapter) supports `stream=True`. The HTTP-level SSE encoding in `embedded.py` was flagged as "needs verification" during the REVIEW pass — confirm it matches OpenAI's `data: {chunk}\n\n` framing and add a regression test.

- [ ] **RoPE / YaRN scaling exposure.** `LlamaContextParams` carries the fields natively; expose on `GenerationConfig` (`rope_freq_base`, `rope_freq_scale`, `yarn_*`). Required for users running NTK-aware long-context configs.

### P3 — polish / cleanup

- [ ] **Split `embedded.py` (~950 LOC).** Mongoose binding + routing + slot management + chat completion all in one file. Same shape of refactor as the recent `LLM` split. Suggested seams: `_mongoose.py`, `routes/`, `slots.py`, `chat_completion.py`. High leverage on continuous batching and vision-in-OpenAI-compat.

- [ ] **`__all__` everywhere.** Public modules (`inferna/__init__.py`, `api.py`, `llama/llama_cpp.py`, `whisper/whisper_cpp.py`, `sd/stable_diffusion.py`) lack `__all__`. Add it so the public/private boundary is explicit and `from inferna import *` is well-defined.

- [ ] **`@overload` on `LLM.__call__`.** Returns `Response | Iterator[str]` depending on `stream=`; static checkers can't narrow without overloads. Add `@overload` for `stream=True` → `Iterator[str]` and `stream=False` → `Response`.

- [ ] **Lazy imports in `inferna/__init__.py` and `agents/__init__.py`.** Importing `inferna` today pulls MCP, jsonrpc, langchain integration, RAG. `from inferna import LLM` should not pay for those. Use PEP 562 module-level `__getattr__` to defer.

- [ ] **Centralize defaults.** Same numeric default appears in 3-4 places (`DEFAULT_MAX_TOKENS=512` in `defaults.py` vs `n_predict=32` in `simple()` etc.). Single source in `defaults.py`; CLI/api modules import constants.

- [ ] **Common exception base.** `ValueError` / `RuntimeError` / `ActionParseError` / `VectorStoreError` with no shared base. Add `InfernaError` so users can write a single `except` that catches "library errors only".

- [ ] **Generator return-type annotations.** Several `Iterator[str]` that are actually `Generator[str, None, None]`. Mostly cosmetic.

- [ ] **Frozen dataclasses.** `GenerationStats`, `Response` (immutable in practice), `ResponseCacheInfo` could be `frozen=True`. Surfaces accidental mutation. (`TokenLogprob`, `TopLogprob`, `StreamSegment` already frozen.)

- [ ] **Context managers on remaining classes.** `EmbeddedServer`, `RAG`, `WhisperContext`, `SDContext` lack `__enter__`/`__exit__`. Native cleanup works either way; explicit is better.

- [ ] **Centralize sampler / server log routing.** `Sampler.print_perf_data()` and `EmbeddedServer` access logs go to stdout. Pipe through `logging` so deployments capture them.

- [ ] **Naming: `ngl` vs `n_gpu_layers`.** `simple()` uses `ngl`; everything else uses `n_gpu_layers`. Pick one form per layer (long names in code, short flags only in CLI) and document.

- [ ] **Whisper wildcard import.** `whisper/whisper_cpp.py` does `from ._whisper_native import *` while `llama/llama_cpp.py` enumerates explicitly. Standardize on enumeration + `__all__`.

- [ ] **Thread-safety docs.** `WhisperContext` / `SDContext` say nothing about thread safety; one-line "not thread-safe; one context per thread" docstring matches what `LLM._busy_lock` documents.

### P4 — nice-to-have

- [ ] **Reranker auto-enable in RAG.** `rag/pipeline.py:117-150` plumbs `rerank` / `reranker` arguments; default is `False`. Consider auto-enabling when a reranker is supplied (currently silently no-op without `rerank=True`).

- [ ] **Streaming agent steps.** `ReActAgent` runs eagerly; no incremental "thought" stream. UX issue for long runs.

- [ ] **ReActAgent context compaction.** Hardcoded char cap rather than token-aware summarization. Fine as a starting point; flag it as a known limitation.

- [ ] **Model registry / `inferna.list_models()`.** Cache parsed GGUF metadata under `~/.cache/inferna/`. Quality-of-life.

- [ ] **Observability / OpenTelemetry.** Span generation around `_generate_stream`, structured server logs, sampler perf via logger. Cheap but not load-bearing.

- [ ] **Stable-diffusion gaps.** ControlNet (high-level wrappers); async `convert_model`; inpainting mask helpers.

- [ ] **Whisper language-detection confidence.** `WhisperContext.full_lang_id()` returns the id; pair with the score so callers can threshold.

- [ ] **Tighten GBNF whitespace rule.** The grammar's `space ::= | " " | "\n"{1,2} [ \t]{0,20}` allows up to 22 whitespace chars per separator. Sampling can drown in pretty-print whitespace, occasionally truncating mid-structure under `max_tokens`. Tighten upstream in `inferna.utils.json_schema_to_grammar` and confirm no regression in the structured-output tests. Workaround in `tests/test_function_calling.py` was a deterministic config (`temperature=0`, `max_tokens=256`) — see the test fixture's `_LIVE_CFG`.

## RAG Scaling (see docs/dev/scaling_rag.md)

- [ ] Persistent quantization state in database metadata (quantize() exists but state is in-memory only)

- [ ] Metadata pre-filtering in vector search (filter by source, date, etc.)

- [ ] Async embedding generation (`embed_batch_async()`)

- [ ] Parallel document loading in DirectoryLoader

- [ ] Batch query processing in RAG pipeline

- [ ] Sharding for 1M+ vector workloads

