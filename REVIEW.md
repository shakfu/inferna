# Comprehensive Review: Wrappers, Build System, and CI

## Scope

Three parallel reviews were conducted and consolidated here:

1. **Wrappers** — every nanobind TU (`_llama_native*.cpp`, `_whisper_native.cpp`, `_sd_native.cpp`, `_mongoose.cpp`), shared headers (`busy_lock.hpp`, `_llama_native.hpp`), and the public Python facades (`llama_cpp.py`, `whisper_cpp.py`, `stable_diffusion.py`).
2. **Build system** — `pyproject.toml`, `CMakeLists.txt`, `Makefile`, `scripts/manage.py`, `scripts/run_wheel_test.py`, `scripts/funcs*.sh`, `scripts/patches/`.
3. **CI workflows** — all five files under `.github/workflows/`.

Findings already addressed by the prior pass (LlamaContext `ensure_valid`, lora_adapter exception type, SD preview callback, vestigial params fields, busy-lock RAII guard, lifecycle close+is_valid on SD/Whisper) are not re-flagged. Each finding cites concrete code with file paths and line numbers as observed during review.

---

## Part A — Wrappers

### High severity

#### A1. `LlamaVocab.tokenize` caps the output buffer at `n_vocab` and never retries

**File:** `src/inferna/llama/_llama_native.cpp:727-738`

```cpp
int n_vocab = llama_vocab_n_tokens(s.ptr);
int max_tokens = std::min((int)(text.size() * 2 + 100), n_vocab);
std::vector<llama_token> tokens(max_tokens);
int n = llama_tokenize(s.ptr, text.c_str(), (int)text.size(),
                        tokens.data(), max_tokens, add_special, parse_special);
if (n < 0) {
    throw std::runtime_error(
        "Failed to tokenize: text=\"" + text + "\" n_tokens=" + std::to_string(n));
}
```

Two bugs in one expression:

- The `min(..., n_vocab)` cap is meaningless — vocabulary size is unrelated to the number of tokens any given input requires. For small-vocab models (~32k) and long inputs the cap throttles below the heuristic; it never reflects actual capacity.
- llama.cpp's contract for `llama_tokenize` returns the **negative required size** when the buffer is too small, so callers should resize and retry. This wrapper instead throws on every undersized buffer, surfacing as spurious tokenization failures on long inputs.

**Recommendation:** drop the `min` with `n_vocab`; on `n < 0`, resize to `-n` and call again before throwing. The retry pattern already exists in this file for `llama_adapter_meta_*` (~lines 774-808) and `llama_chat_apply_template` (~lines 933-940); use that as the model.

#### A2. `WhisperContext` exposes ~30 native methods that segfault if `close()` is called first

**File:** `src/inferna/whisper/_whisper_native.cpp:421-559`

The earlier pass added `close()` (line 421) and `is_valid` (line 424) but did **not** add `ensure_valid()` guards. Every accessor below — `n_vocab`, `n_text_ctx`, all `model_*`, all `token_*`, `tokenize`, `encode`, `full`, `full_*`, `print_timings` — passes `s.ctx` straight into whisper.cpp without a null check (verified at lines 427-458 and beyond). This is exactly the crash surface that was just fixed for `LlamaContext`.

**Recommendation:** add an `ensure_valid()` member to `WhisperContextW` (mirroring `LlamaContextW`) and call it at the top of every binding lambda that dereferences `ctx`. The fix is mechanical — same shape as what was done in `_llama_native.cpp:1067-1207`.

#### A3. Mongoose `send_reply` accepts an unverified `uintptr_t` connection pointer

**File:** `src/inferna/llama/server/_mongoose.cpp:118-124`

```cpp
mg_connection* c = reinterpret_cast<mg_connection*>(conn_id);
if (!c) return false;
inferna_mg_http_reply(c, ...);
```

`conn_id` is a Python integer. Python code can stash it past the lifetime of the request (mongoose closes/frees connections during its own poll cycle), or pass an arbitrary integer. Calling `mg_http_reply` on a freed/bogus pointer is undefined behavior; on a closed-but-not-yet-reaped connection it can write into freed memory.

The only guard is the null-check (which catches `0` only). Nothing validates that `c` is in `mgr->conns`, nothing checks `c->is_closing`, and nothing prevents user code from caching `_conn_id` for later use. In current `embedded.py` usage the dispatcher calls `send_reply` synchronously inside `_dispatch`, so the pointer is alive — but the API does not enforce that contract.

**Recommendation:** at minimum, walk the `mgr->conns` linked list and reject pointers not in it. Better: hand back an opaque per-request token (sequence number indexed into a Manager-side map) instead of a raw pointer.

### Medium severity

#### A4. `_llama_progress_cb` returns `true` on Python exception, breaking abort semantics

**File:** `src/inferna/llama/_llama_native.cpp:63-73`

```cpp
} catch (...) {
    return true;  // swallow callback errors and continue model loading
}
```

llama.cpp's progress callback contract is: **return `false` to abort loading**. Returning `true` from the catch makes a Python `KeyboardInterrupt` or any handler exception silently *not* abort the load. The user sees no error and loading continues for minutes.

**Recommendation:** return `false` to abort, or explicitly set the Python exception via `PyErr_*` and return `false`. Do not silently swallow.

#### A5. `_llama_log_cb` reads `g_log_cb` without the GIL held

**File:** `src/inferna/llama/_llama_native.cpp:286-294, 1310-1314`; same shape in `src/inferna/sd/_sd_native.cpp:271-304`.

```cpp
static nb::object g_log_cb;
extern "C" void _llama_log_cb(ggml_log_level level, const char* text, void*) {
    if (!g_log_cb.is_valid() || g_log_cb.is_none()) return;
    nb::gil_scoped_acquire gil;
    ...
}
```

`g_log_cb` is a global `nb::object`. The log callback fires from any ggml worker thread; `set_log_callback` mutates `g_log_cb` under the GIL. The `is_valid()/is_none()` check runs *before* `gil_scoped_acquire`, so a worker thread can race with a thread mutating the object — the `nb::object` internals are not thread-safe.

**Recommendation:** acquire the GIL first, then check `is_valid/is_none`.

#### A6. `LlamaVocab.token_to_piece` returns `std::string` that may carry invalid UTF-8

**File:** `src/inferna/llama/_llama_native.cpp:740-748`

```cpp
return std::string(buf, len);
```

nanobind's `std::string` → `str` conversion goes through `PyUnicode_DecodeUTF8` with strict errors. Byte-level BPE tokenizers regularly emit lone continuation bytes for individual pieces; the binding will raise `UnicodeDecodeError` from inside C++. The inline comment acknowledges the issue but doesn't fix it.

**Recommendation:** return `nb::bytes` and let Python decode with `errors='replace'`, or decode here with `errors='replace'` before returning.

#### A7. `MtmdContext` lacks the lifecycle contract that the other contexts now have

**File:** `src/inferna/llama/_llama_native_mtmd.cpp:80-90, 323-356`

No `close()`, no `is_valid`, no `ensure_valid()`. Several methods do `if (!s.ptr) throw std::runtime_error("Context not initialized")` at lines 343-356, 362, 395, 400, 416 — but the pattern is ad-hoc and inconsistent with the other three context types after the recent normalization.

**Recommendation:** add `close()` + `is_valid` + `ensure_valid()` to `MtmdContextW`, replacing the scattered `if (!s.ptr)` throws.

#### A8. `MtmdContext.get_output_embeddings` returns a `nb::list[nb::list[float]]`

**File:** `src/inferna/llama/_llama_native_mtmd.cpp:398-411`

For a CLIP projector with `n_tokens=576, n_embd=4096` this allocates ~2.4M `PyFloat` objects per call.

**Recommendation:** return a flat `std::vector<float>` plus a shape tuple (matches `LlamaContext.get_logits` at lines 1161-1168), or hand back a zero-copy `nb::ndarray<float>` view anchored on the parent context.

#### A9. `SDContext.__exit__` does not call `close()`

**File:** `src/inferna/sd/stable_diffusion.py:580-586`

```python
def __exit__(self, exc_type, exc_val, exc_tb):
    # Native dtor will free the underlying ctx when refcount drops to zero;
    # we don't have an explicit close, but mark as no longer usable.
    return None
```

The comment is now stale: `_n.SDContext` *does* expose `close()` (added in the previous review at `_sd_native.cpp:785-787`). A `with` block currently provides no determinism — the context is held until GC.

**Recommendation:** call `self.close()` (or the equivalent native method) in `__exit__`. Same applies symmetrically to `Upscaler.__exit__` (~line 718), once `Upscaler` is given its own `close()`.

### Low severity / style

- **A10.** Three independent reimplementations of `ggml_backend_load_all` bootstrap logic at `_llama_native.cpp:1330-1348`, `_whisper_native.cpp:576-598`, `_sd_native.cpp:995-1016`. Identical code, three copies. Factor into `src/inferna/common/backend_loader.hpp`.
- **A11.** Detokenize error message swaps `text` and `rc` at `_llama_native.cpp:757-759` (the label says `text="..."` but the value is `rc`). Cosmetic.
- **A12.** `LlamaSampler.add_*` chain methods (`_llama_native.cpp:1230-1290`) do not null-check the inner `llama_sampler_init_*` result before calling `llama_sampler_chain_add`. A malformed grammar that returns NULL from `init_grammar` silently no-ops the chain.
- **A13.** `LlamaModelKvOverride.key` setter (`_llama_native.cpp:672-677`) silently truncates oversized keys via `strncpy` instead of throwing.
- **A14.** Pattern inconsistency: `WhisperContext` exposes `_busy_lock` as a property; `LlamaContext` has no such accessor. Document the per-module rationale (currently only referenced from one of the three).

### Test gaps

- No regression test for "method on closed context" — needed for `LlamaContext` (the just-added `ensure_valid()` is currently unverified) and required once A2/A3 are fixed.
- No test exercising the tokenize retry path described in A1 — feed a 50KB string into `LlamaVocab.tokenize` for a small-vocab model.
- No test for progress-callback abort semantics (A4): supply `lambda p: False`, verify the load aborts; supply a callback that raises, verify documented behavior.
- No test for `MtmdContext` post-close behavior (A7), nor for `MtmdContext` use after parent `LlamaModel` is freed.
- No test driving `_mongoose.send_reply` with a stale or fake `conn_id` (A3). The public `MongooseConnection.send` shouldn't be callable past request scope, but the underlying `_mg.send_reply(int, ...)` will happily dereference garbage.

---

## Part B — Build system

### High severity

#### B1. `MACOSX_DEPLOYMENT_TARGET` disagrees across four sites

| Source | Value |
|---|---|
| `Makefile:4` | `14.7` |
| `scripts/manage.py:155` | `12.6` |
| `scripts/manage.py:2012-2014` (`WheelBuilder.get_min_osx_ver`) | `10.9`/`11.0` |
| `pyproject.toml:201` (cibuildwheel macos) | `11.0` |

A wheel produced via `make wheel` (deployment 14.7) versus one from cibuildwheel (11.0) will have a different `LC_BUILD_VERSION` minimum. Worse: dylibs built by `manage.py` (12.6) embedded into a wheel built with cibuildwheel's target (11.0) will carry references to symbols that exist only on 12.6+ — a wheel claiming 11.0 compatibility that will crash at load time on older macOS.

**Recommendation:** establish a single source of truth (likely `pyproject.toml`) and have the others read from it. At minimum align all four values, and add a CI assertion.

#### B2. `wheel-<backend>` Makefile targets do not rebuild `thirdparty/` for the matching backend

**File:** `Makefile`, e.g. `wheel-cuda` (~line 272)

```makefile
wheel-cuda:
    @GGML_CUDA=1 uv build --wheel
```

No dependency on `$(LIBLAMMA)`, no call into `manage.py build --all --deps-only`. The build picks up whatever was last built into `thirdparty/llama.cpp/lib` — likely Metal binaries on a developer's Mac — and silently produces a wheel that *claims* CUDA support but links Metal-built archives. The static `wheel:` target (~line 82) does depend on `$(LIBLAMMA)`, which makes the inconsistency easy to overlook.

**Recommendation:** make each `wheel-<backend>` target depend on the matching `build-<backend>`, or invoke `manage.py build --all --deps-only` with the backend env first. The cibuildwheel `before-build` hook handles this in CI; local `make` does not.

### Medium severity

#### B3. `Makefile:2` defines `VERSION := 0.1.20`, never used, never updated by `make bump`

**File:** `Makefile:2`; `pyproject.toml:3` is `0.1.1`; `manage.py do_bump` only edits `pyproject.toml` and `src/inferna/__init__.py`.

The variable is dead today, but if anyone later wires it into a release rule it will cut a wrong tag. **Recommendation:** delete it, or have `do_bump` keep it in sync.

#### B4. macOS cibuildwheel rebuilds all native deps once per Python version

**Files:** `pyproject.toml:172` (global `before-build`), `pyproject.toml:188` (Linux uses `before-all` — runs once), no macOS override (falls through to global).

Linux runs `manage.py build --all --deps-only --no-sd-examples` once; macOS runs it five times (cp310-cp314). Each invocation re-clones llama.cpp/whisper.cpp/sd.cpp because `manage.py setup()` (`scripts/manage.py:957-964`) does not handle pre-existing checkouts.

**Recommendation:** add a macOS-specific `[tool.cibuildwheel.macos] before-all` mirroring Linux.

#### B5. `manage.py setup()` is not idempotent on partial failure or version change

**File:** `scripts/manage.py:957-964` (Builder.setup), `:1170` (LlamaCppBuilder.build)

`setup()` clones only if `src_dir` doesn't exist. If a previous run cloned but failed mid-build, the directory exists in a partial state. Changing `--llama-version` between runs has no effect when an old checkout exists — the build silently uses the old commit.

**Recommendation:** in `setup()`, when `src_dir` exists, verify the checked-out ref against `self.version` and either `git fetch && git checkout` or fail loudly.

#### B6. `Makefile` `wheel-opencl-dynamic` omits `SD_USE_VENDORED_GGML=0`

**File:** `Makefile:306-307`

```makefile
wheel-opencl-dynamic:
    @GGML_OPENCL=1 WITH_DYLIB=1 uv build --wheel
```

Other dynamic wheel targets (e.g. `wheel-vulkan-dynamic` ~line 297) set `SD_USE_VENDORED_GGML=0`. Without it, stable-diffusion.cpp links its own vendored ggml while llama.cpp ships a separate ggml dylib — an ABI mismatch the `GGML_MAX_NAME=128` plumbing was added to prevent.

**Recommendation:** add `SD_USE_VENDORED_GGML=0` to `wheel-opencl-dynamic`. Audit `build-cpu-dynamic` and `wheel-cpu-dynamic` for the same.

#### B7. `cmake.version` floor in `pyproject.toml` is `>=3.21`, but CMakeLists.txt requires `3.26...3.30`

**File:** `pyproject.toml:41` vs `CMakeLists.txt:1`

scikit-build-core can pick up CMake 3.21 from the build env (or bootstrap one) and pass it to a project that demands ≥3.26 — immediate configure failure.

**Recommendation:** bump `pyproject.toml`'s `cmake.version` to `>=3.26`.

#### B8. `Makefile:90-94` `wheel-dynamic` is hard-coded to `libllama.dylib` (macOS-only)

```makefile
wheel-dynamic: $(LLAMACPP)/dynamic/libllama.dylib
```

On Linux the dependency would never be satisfied (`libllama.so`); on Windows `llama.dll`. The `WITH_DYLIB` block (lines 38-43) parameterizes `LIBLAMMA` per-platform but not this rule.

**Recommendation:** parameterize the dependency by platform, or drop the file-target dependency and call `manage.py` directly.

#### B9. `wheel.exclude = [..., "*.lib"]` is broader than needed

**File:** `pyproject.toml:45`

scikit-build-core matches `wheel.exclude` against installed wheel paths. Today CMake doesn't `install(FILES)` the import libs, so the exclusion is effectively a no-op. But on Windows when delvewheel relocates DLLs, any companion `.lib` siblings it copies into `inferna.libs/` will be silently stripped.

**Recommendation:** scope to known source-tree areas or trust CMake's install rules. Most of the entries (`*.cpp`, `*.h`, `*.hpp`, `*.a`) are already redundant since the source tree isn't installed; trimming the list reduces footguns.

#### B10. WITH_DYLIB mode still calls `find_package(Vulkan REQUIRED)` etc.

**File:** `CMakeLists.txt` (e.g. `find_package(hip REQUIRED)` ~line 526; Vulkan, CUDA elsewhere)

When `manage.py build --dynamic --vulkan` ships a pre-built llama.cpp release containing `libggml-vulkan`, the system Vulkan SDK is already linked into the dylib. The `_llama_native` extension itself doesn't reference Vulkan symbols. Yet the CMake configure still hard-`REQUIRED`s the system package, forcing an SDK install on dev machines that only need to build the extension wrapper.

**Recommendation:** in `WITH_DYLIB=ON` mode, skip system-package discovery for backends whose dylibs are being shipped as runtime deps.

### Low severity / style

- **B11.** `Makefile` `LLAMACPP_LIBS` (~line 344) and `MACOS_FRAMEWORKS` (~line 380) are dead variables — defined but referenced by no rule. Delete.
- **B12.** `make test-tts` / `test-cli` / `test-chat` (~lines 321, 326, 332) invoke `python -m inferna.tts` / `inferna.cli` / `inferna.chat`, but these modules live at `inferna.llama.tts` / `.cli` / `.chat`. Either the targets are broken or they rely on an undocumented re-export. Switch to the proper CLI entry point: `python -m inferna cli ...`.
- **B13.** `STABLE_BUILD` flag in `manage.py:141-153` has identical pinned versions in both branches — the flag is a no-op today. Either remove the dead branching or maintain a "bleeding-edge" set in the `else` arm.
- **B14.** `scripts/funcs.sh` and `scripts/funcs_py.sh` reference `./bin/llama-cli` / `./bin/llama-server`; the actual binaries live at `thirdparty/llama.cpp/bin/...` (compare `Makefile:315`). The helpers are broken from a fresh checkout. Delete or fix the paths.
- **B15.** `scripts/patches/README.md` documents an unapplied patch and still mentions Cython (the migration to nanobind is documented elsewhere). Update wording or drop the directory.
- **B16.** `manage.py:2786-2814` carries a comment about generated Cython `.cpp` files plus an empty `cython_cpp_files: list = []` and a no-op `for cpp in cython_cpp_files` loop. Cython is gone — delete.
- **B17.** `manage.py:1085-1088` declares `extra_libs = ["llama", "llama-common", "mtmd"]`. CMake links `LIB_LLAMA` and `LIB_MTMD` but never `LIB_LLAMA_COMMON`. The extra archive is built and copied but unused. Either drop from `extra_libs` or wire it into CMake (only needed if `_llama_native*.cpp` calls into `common_*` symbols — verify before deleting).
- **B18.** `pyproject.toml:188` (Linux `before-all`) hard-codes `cp310` for Python interpreter discovery. Reasonable workaround for manylinux2014's old system Python, but a future bump to manylinux_2_28 (modern Python included) leaves dead pinning. Add a TODO.

---

## Part C — CI workflows

### High severity

#### C1. `build-gpu-wheels.yml` (1165 lines) and `build-gpu-wheels-abi3.yml` (1195 lines) are near-identical

The only meaningful per-job differences are the name suffix (`abi3`), cache key prefix, artifact name prefix, two extra env vars (`SKBUILD_CMAKE_DEFINE="INFERNA_ABI3=ON"`, `SKBUILD_WHEEL_PY_API=cp312`), and `CIBW_BUILD: "cp312-*"`. Manylinux images, repo URLs, CUDA versions, compiler arches (`75`), exclude lists, repair commands — all duplicated.

The duplication has already produced drift: `build-gpu-wheels-abi3.yml:114` pins `cibuildwheel==3.4.1` for the CUDA Linux job while every other job in that same file (lines 227, 343, 451, 568, 656, 733) uses `==3.4.0`.

**Recommendation:** factor into a reusable `workflow_call` workflow with `abi3: bool` and `python_matrix: string` inputs, or add a `wheel_kind: [stable, abi3]` matrix dimension. Eliminates ~1100 duplicated lines.

#### C2. cibuildwheel pin diverges from `pyproject.toml`'s declared CIBW config

**Files:** `pyproject.toml:164-214` carries `[tool.cibuildwheel]` config but does not pin a cibuildwheel version. Workflows pin per-step. Every GPU job overrides `before-build` and `before-all` (e.g. `build-gpu-wheels.yml:144` `CIBW_BEFORE_BUILD_LINUX` re-implements the rename + `write_build_config`; the CPU `build-cibw.yml:80-86` re-declares `CIBW_ENVIRONMENT_LINUX`).

Any change to `[tool.cibuildwheel.linux]` is silently ignored by every GPU build.

**Recommendation:** pick a single source of truth — either keep all CIBW config in workflows and gut the pyproject section, or move backend-agnostic defaults to pyproject and have workflows extend (not replace) via `CIBW_ENVIRONMENT_PASS_LINUX` plus per-job overrides only.

#### C3. No CI runs the actual test suite

Per repo policy ("Always run `make test` after each major change"), but none of the five workflows runs `make test` or `pytest tests/`. The `[tool.cibuildwheel]` section at `pyproject.toml:175-176` declares a `test-command` running pytest — but every GPU job sets `CIBW_TEST_COMMAND: ""` (e.g. `build-gpu-wheels.yml:165, 280, 387, 493, 575, 661, 747`), disabling it. The CPU `build-cibw.yml` only does smoke imports plus a 16-token `complete()`.

1150+ tests never run in CI on built wheels. The pyproject test config is effectively dead.

**Recommendation:** at minimum, run `pytest` against the cp312 cpu-linux wheel of every workflow run. Even a 5-minute subset is meaningful.

### Medium severity

#### C4. cibuildwheel version drift inside a single workflow

**File:** `build-gpu-wheels-abi3.yml:114` pins `cibuildwheel==3.4.1`; lines 227, 343, 451, 568, 656, 733 use `==3.4.0`. Almost certainly an accidental partial bump.

**Recommendation:** lift the cibuildwheel version into the workflow `env:` block (or a job-level matrix value) so it cannot drift.

#### C5. `build-new-wheels.yml` is a half-finished staging clone

The header comment (lines 6-8) describes it as "staging for variants that get promoted to `build-gpu-wheels.yml`". It builds only one job (CUDA 13.1 Windows, lines 44-136) using `cibuildwheel==3.4.0`, single-Python (`CIBW_BUILD: "cp312-*"`, line 33), and `Jimver/cuda-toolkit@v0.2.35` (line 63) — production uses `@v0.2.21`. The smoke-test matrix (lines 146-153) is single-entry, so the matrix abstraction is overhead.

**Recommendation:** either promote CUDA 13.1 Windows into `build-gpu-wheels.yml` and delete the staging file, or document an end-state and align action versions with production now.

#### C6. CUDA Toolkit action version split (production vs staging)

`build-gpu-wheels.yml:534` and `build-gpu-wheels-abi3.yml:548` pin `Jimver/cuda-toolkit@v0.2.21`. `build-new-wheels.yml:63` pins `@v0.2.35`. If the older pin is intentional (CUDA 12.4 compatibility), document it; otherwise unify.

#### C7. GPU wheels never run an actual inference smoke test

**File:** `build-gpu-wheels.yml:766-1015`

Inference is exercised only for SYCL (lines 995-1015, with `continue-on-error: true`). CUDA, ROCm, and Vulkan jobs do imports only, with the (correct) reasoning that hosted runners lack GPUs. But every ggml build registers the CPU backend too — so a `complete()` call with `n_gpu_layers=0` would catch wheels that import but cannot construct an `LLM`. CPU `build-cibw.yml:216-236` already does this.

**Recommendation:** add a CPU-fallback inference smoke test (with `n_gpu_layers=0`) to every GPU job.

#### C8. Cache-key composition omits `pyproject.toml` for the stable GPU workflow

`build-gpu-wheels.yml:99, 210, 324, 430, 531, 617, 700` hash only `scripts/manage.py` and the workflow file. The abi3 variant correctly includes `pyproject.toml` (`build-gpu-wheels-abi3.yml:106` and similar). A change to `pyproject.toml` affecting build deps does not invalidate cached `thirdparty/` on the stable workflow.

**Recommendation:** add `pyproject.toml` to all `hashFiles(...)` cache-key calls in `build-gpu-wheels.yml`.

#### C9. `gh release create` failure mode swallows everything via `2>/dev/null || true`

**Files:** `build-cibw.yml:286-289`, `build-cibw-abi3.yml:346-349`, `build-gpu-wheels.yml:1129-1136`

The shell pattern hides "release already exists" but also hides auth errors and rate limits. CI passes silently when the release wasn't actually uploaded.

**Recommendation:** `gh release view "${TAG}" >/dev/null 2>&1 || gh release create "${TAG}" --prerelease ...` to distinguish "exists" from "failed".

#### C10. Workflows mutate `pyproject.toml` mid-build to rename the package

Every GPU job runs `sed` against `pyproject.toml` to substitute `name = "inferna"` → `name = "inferna-cuda12"` etc. (e.g. `build-gpu-wheels.yml:145, 259, 369, 479, 562, 649, 732`). The regex relies on exact whitespace; reformatting the TOML breaks it silently. The workflow is also the de-facto source of truth for the published package name, but a developer reading `pyproject.toml` cannot tell that.

**Recommendation:** introduce a `scripts/manage.py rename_package <variant>` helper to centralize the substitution, validate it, and make the variant set version-controllable.

#### C11. Triggers are `workflow_dispatch`-only

All five workflows have only manual triggers. There is no CI on push, no CI on PR, no automatic build on tag. Combined with C3, this means a contributor's PR gets zero automated wheel build and zero test run; a release tag does not auto-build wheels. This is a deliberate cost-control decision (GPU/Windows runner minutes are expensive), but a slim PR-triggered linux-cpu-cp312 build would catch the common breakage cheaply.

**Recommendation:** add a `pull_request:` trigger to a minimal subset of `build-cibw.yml` (linux-intel + cp312 + smoke), gated by paths-filter so only build-affecting changes run it.

### Low severity / style

- **C12.** `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24` typing is inconsistent: `: true` (YAML bool) in `build-cibw.yml:20`, `build-cibw-abi3.yml:27`, `build-gpu-wheels-abi3.yml:79` vs `: "true"` (string) in `build-gpu-wheels.yml:72`. Pick one.
- **C13.** `build-cibw.yml:107-114` smoke-tests on `linux-intel`, `macos-arm`, `windows-intel` but builds for `macos-intel` (line 38) too. The macOS Intel wheel is uploaded but never installed in CI. Add `macos-15-intel` to the smoke matrix.
- **C14.** Inline `python -c "..."` heredocs duplicated across files (`build-cibw.yml:178-214`, `build-cibw-abi3.yml:243-273`, `build-gpu-wheels.yml:892-938`; same pattern for version-extraction). Move to `scripts/ci_smoke.py` and `scripts/ci_version.py` so changes propagate atomically.
- **C15.** No PyPI publishing. Releases only land on GitHub. Positive from a security standpoint (no `PYPI_API_TOKEN`), but worth flagging if PyPI is intended. If added, prefer trusted publishing via `pypa/gh-action-pypi-publish@release/v1`.
- **C16.** Secrets surface is otherwise clean: only `secrets.GITHUB_TOKEN` is used (`build-cibw.yml:279`, `build-cibw-abi3.yml:339`, `build-gpu-wheels.yml:1122`); no `pull_request_target`, no `pwn-request` patterns. No findings here.

### Workflow inventory

| Workflow | Purpose | Triggers | Python matrix | Backends |
|---|---|---|---|---|
| `build-cibw.yml` | CPU/Metal wheels (full Python matrix) | `workflow_dispatch` | cp310-cp314 (via pyproject) | linux x86_64 (CPU), windows x86_64 (CPU), macos-15-intel (CPU), macos-arm (Metal) |
| `build-cibw-abi3.yml` | CPU/Metal abi3 wheels (single wheel per platform) | `workflow_dispatch` | cp312 build, smoke on cp312/3.13/3.14 | same 4 platforms |
| `build-gpu-wheels.yml` | GPU wheels, full Python matrix | `workflow_dispatch` (per-backend boolean inputs) | cp310-cp314 | CUDA 12.4 (Linux+Windows), ROCm 6.3 (Linux), SYCL (Linux), Vulkan (Linux+Windows+macOS-Intel) |
| `build-gpu-wheels-abi3.yml` | GPU wheels, abi3 variant | `workflow_dispatch` (same inputs) | cp312 only | same as build-gpu-wheels |
| `build-new-wheels.yml` | Staging for new GPU variants | `workflow_dispatch` | cp312 only | CUDA 13.1 Windows only |

Two parallel axes: **{stable, abi3}** × **{cpu/metal, gpu, staging}**. The matrix is intentional but implemented by file copy, hence the duplication described in C1.

---

## Cross-cutting themes

1. **Configuration drift is the dominant class of bug.** macOS deployment target (B1), project version (B3), cibuildwheel version (C2/C4), CUDA toolkit action (C6), `SD_USE_VENDORED_GGML` (B6) — all the same shape: a value defined in multiple places with no single source of truth and no CI assertion forcing them to agree.

2. **Lifecycle contracts are partial.** The previous review normalized `close()` + `is_valid` across `LlamaContext`, `SDContext`, `WhisperContext`. But `ensure_valid()` enforcement was only added to `LlamaContext` (A2/A3 are the gaps), and `MtmdContext` was missed entirely (A7).

3. **CI-side test coverage is thinner than the local suite.** The repo runs 1367 tests locally; CI runs `import` + a 16-token completion (C3). Catching regressions before they land requires running pytest in CI, which the existing pyproject config already declares but every GPU job disables.

4. **Duplication is treated as a copy-paste problem rather than a parameterization opportunity.** Three independent `ggml_backend_load_all` reimplementations (A10), two near-identical 1165-line GPU workflow files (C1), and three copies of CI heredoc Python (C14). All factor cleanly given a small effort.

## Recommended priority order

If this list is fixed in waves, the natural order is:

1. **A2** (whisper crash surface) — same shape as the just-fixed Llama issue, cheap to fix.
2. **A1** (tokenize buffer cap) — silent data corruption / spurious failures.
3. **A3** (mongoose pointer validation) — UB on a public API surface.
4. **B1** (deployment target alignment) — distribution-correctness bug.
5. **C3** (run pytest in CI) — high-leverage; turns 1367 dormant tests into a regression net.
6. **C1** (factor abi3/stable into one workflow) — eliminates the largest source of CI drift.
7. **A4-A9** (callback/lifecycle/perf items) — cluster of smaller fixes.
8. **B2-B10** (Makefile/manage.py drift) — cluster.
9. **C2-C11** (CI hygiene) — cluster.
10. Style items last.
