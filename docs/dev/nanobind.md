# Cython → nanobind migration: final analysis

The migration is complete. All three upstream wrappers (llama.cpp, whisper.cpp,
stable-diffusion.cpp) plus the embedded mongoose server now use nanobind. No
Cython remains in the build. Final test state: **1,472 passing, 31 skipped,
0 failed.**

This document is the post-mortem. For the in-flight handoff state during the
migration, see git history (`git show 9abbb99:NANOBIND.md` for the
mid-migration handoff that this replaces).

## Final architecture

Every native module now follows the same pattern:

```
src/inferna/<area>/
    _<area>_native.cpp        # nanobind C++ TU — thin binding layer
    _<area>_native_*.cpp      # optional companion TUs for sub-areas
    <area>_cpp.py             # public Python facade
                              # (or stable_diffusion.py / embedded.py)
```

| Area      | Native TU                          | Public Python module        | Companion TUs |
|-----------|------------------------------------|-----------------------------|---------------|
| llama     | `_llama_native.cpp`                | `llama_cpp.py`              | `_llama_native_tts.cpp`, `_llama_native_mtmd.cpp`, `_llama_native_enums.cpp` |
| whisper   | `_whisper_native.cpp`              | `whisper_cpp.py`            | —             |
| sd        | `_sd_native.cpp`                   | `stable_diffusion.py`       | `stb_impl.cpp` |
| embedded  | `_mongoose.cpp`                    | `embedded.py`               | `mongoose.c`, `mongoose_wrapper.c` |

Companion TUs share state via a small header (`_llama_native.hpp`) declaring
`inferna::unwrap_model(handle)` / `unwrap_ctx(handle)` so multimodal/tts code
can recover the underlying llama.cpp pointers without dragging full struct
layouts across translation units.

Pure-Python helpers extracted from the old `.pyx`:
- `_python_helpers.py` — memory pools, download API (HF + Docker), n-gram cache
- `_speculative.py` — speculative decoder (was 264-line `.pxi`, now 178-line `.py`)

## Pros

**1. Headers replace `.pxd` files — no more shadow API.**
The biggest win. Previously every llama.cpp/whisper.cpp/sd.cpp struct, enum,
and function signature had to be re-declared in a `.pxd` file (~3.4K lines
total). Every upstream change risked silent struct-layout drift that Cython
couldn't detect. nanobind reads the actual C/C++ headers — there's exactly one
source of truth. This is worth a lot for a project that pins to specific
upstream commits and bumps them often.

**2. Real C++ ergonomics.**
`std::vector<float>`, `std::optional<std::string>`, `std::function`,
`nb::bytes`, lambda closures, RAII destructors — all native. Compare to the
Cython memoryview / `cdef extern` / `<bytes>data` / function-pointer-to-Python-callable
bridge gymnastics, which were genuinely tortured in places (see the old
`log_callback` / `progress_callback` patterns). The nanobind versions are
shorter and easier to follow.

**3. One language per file.**
Before: a `.pyx` is part-Python part-Cython part-C, with Cython-level types
(`cdef`, `cppbool`), C-level types (`uint32_t`), Python defaults, three
flavors of string handling, and four `cimport` namespaces. After: one `.cpp`
is just C++, one `.py` is just Python. Easier to onboard, easier to grep,
easier for tooling.

**4. Build toolchain shrank.**
Removed: `find_package(Cython)`, `include(UseCython)`, `cython_transpile`,
the entire 115-line `add_cython_extension` wrapper function,
`CYTHON_INCLUDE_DIRS`, two cmake helper modules, the `cython` and
`cython-cmake` packages from `[build-system].requires`, the `cython` dev dep.
`pyproject.toml` is shorter; CMake is shorter.

**5. Cleaner module layout.**
The new `_*_native.cpp` + `*.py` shim pattern (now applied uniformly to
llama, whisper, sd) cleanly separates "thin C++ binding layer" from "Pythonic
public surface." Adding Python-only methods, defaults, or convenience
wrappers is now trivial — you just edit a `.py` file. With Cython this would
have meant editing the `.pyx`, recompiling, and dealing with `cdef class`
subclassing constraints.

**6. Pure-Python helpers escaped the `.pyx` jail.**
~600 lines of pure-Python code (`download_model`, `NgramCache`, memory pools,
etc.) were sitting in `llama_cpp.pyx` for no reason other than convenience.
They now live in `_python_helpers.py` where they can be linted, type-checked,
debugged, and edited without invoking the C++ build. This was a long-standing
smell and the migration forced the cleanup.

**7. Some classes simplified into pure Python.**
`speculative.pxi` was 264 lines of `cdef`-decorated Python that only used
Cython for fast pointer access. As pure Python on top of the new bindings
(`_speculative.py`) it's 178 lines, easier to follow, and the perf cost is
irrelevant compared to the underlying `llama_decode` call.

**8. Compile speed is roughly comparable.**
Subjectively: full clean rebuild times feel similar. nanobind compiles fewer
template instantiations than pybind11 but more than Cython (Cython generates
straight-line C). Not benchmarked rigorously.

**9. Slightly smaller wheels.**
Measured: nanobind wheel **11.6 MB** vs Cython equivalent **12.3 MB** —
roughly 6% smaller. nanobind's runtime is leaner than Cython's per-extension
boilerplate (Cython embeds a fair amount of generated trampoline code per
`.pyx`); with multiple extensions in the same wheel that adds up. Modest but
consistent with nanobind's design goal of being the leanest of the C++
binding libraries.

## Cons

**1. The `.pxi` constants problem.**
Cython auto-exposed every `cpdef enum` value from `cimport`ed `.pxd` files as
a module attribute — callers got `cy.LLAMA_VOCAB_TYPE_BPE` for free. nanobind
requires explicit registration. Result: a hand-maintained 200-line
`_llama_native_enums.cpp` that mirrors 109 enum values from llama.h/ggml.h,
plus another 100-line block in `llama_cpp.py` re-exporting them. Every time
upstream adds an enum value, both files need an edit. This is exactly the
"shadow API" problem (pro #1) sneaking back in for constants.

**2. Test surface mismatches were silent.**
Cython's automatic `OverflowError` on `uint32_t` assignment, automatic
None-rejection by typed parameters, automatic `FileNotFoundError` raising,
and so on — all came "for free" with `cdef`. nanobind requires explicit
`PyErr_SetString(PyExc_OverflowError, ...)` plumbing per call site. The 6
test failures we hit at the end were all this category: behaviors Cython
provided implicitly that nanobind requires explicit code for. Easy fixes
individually, but you have to find them all (and in larger codebases you'd
miss some without good test coverage).

**3. Lost the "free" Python-side composition.**
Cython's `cdef class` could be subclassed in Python with `__class__`
reassignment to add methods to native-returning factory results (e.g.,
`LlamaModel.get_vocab()` returning a custom subclass). nanobind enforces
matching `tp_dealloc` which makes this fail with "deallocator differs." We
had to use the composition pattern (`.native` handle + wrapper class) for
`SDImage` and similar cases. Doable but more code than the Cython subclass
trick.

**4. Cross-TU access requires plumbing.**
In Cython, `include "mtmd.pxi"` gave the included file direct access to all
`cdef class` types in the parent. With nanobind, separate `.cpp` files in the
same module need a header (`_llama_native.hpp`) declaring
`inferna::unwrap_model(handle)` etc. Three new files of glue (header + impl +
per-TU includes) for what was textual inclusion before.

**5. More files, same total LOC (or slightly more).**
Before: `llama_cpp.pyx` (4318) + 5 `.pxd` files (~2300) + 3 `.pxi` (~975) =
~7600 lines, 9 files.
After: `_llama_native.cpp` (~1500) + `_llama_native_tts.cpp` (~75) +
`_llama_native_mtmd.cpp` (~340) + `_llama_native_enums.cpp` (~200) +
`_llama_native.hpp` (20) + `llama_cpp.py` (~280) + `_python_helpers.py`
(~470) + `_speculative.py` (~180) = ~3060 lines, 8 files.

Counted naïvely it's a *win* (~7600 → ~3060), but the comparison is unfair
because the `.pxd` files were declarations of upstream APIs that nanobind now
reads directly from headers, and the helpers/speculative/memory-pool code
mostly stayed the same size. The C++ binding layer itself is roughly
comparable in size to the Cython equivalent — maybe 10–20% leaner.

**6. Cython gave you a debugger; nanobind gives you gdb.**
You could set breakpoints in `.pyx` files with the right tooling,
line-profile them, get coverage. The new C++ TUs are opaque to Python tooling
— the only way to debug a binding bug is to read the C++. In practice this
hasn't bitten us yet, but it's a real ergonomic loss.

**7. LSP/IDE story is rougher.**
Most of the build cycle, the LSP fires diagnostics like "nanobind/nanobind.h
not found" and "use of undeclared identifier 'nb'" because the editor doesn't
know about the cmake-managed include paths. Cython files at least had
Cython-aware editors that understood `cdef`/`cimport`. Working in a hot
LSP-error environment is mildly demoralizing even when the build is green.

**8. The migration itself was a footgun.**
The `whisper_cpp.cpp` near-loss (an entry in `.gitignore` from the Cython era
silently swallowed the new nanobind source for an entire commit) is exactly
the kind of trap migrations create. Caught before it cost more than a day,
but it almost erased a 1,376-line file. Any half-converted state in the
build is a cliff.

**9. Module name mismatch is now possible.**
With `_*_native.cpp` + `*_cpp.py` shim pattern, the import path and the
actual extension module are decoupled. This is fine when working — and in
fact is what enables progressive migration — but it's a layer of indirection
a maintainer has to know about to debug import issues. The old "what you
import is what you wrote" property is gone.

## Net read

For **this project** — a thin wrapper around three rapidly-evolving
upstreams, with a heavy test suite already in place — the migration was
worth it. The big win (#1: no more shadow API) directly addresses inferna's
biggest historical pain (Cython `.pxd` files going stale on llama.cpp bumps).
The build toolchain simplification (#4) and the Python-helper cleanup (#6)
are real quality-of-life improvements.

The downsides are real but mostly one-time costs. The enum-export duplication
(#1 con) is the only ongoing tax, and it's small.

For **a different project** — say, one that depends on Cython-specific
features like memoryviews, fused types, or `nogil` extensively, or one with
sparse test coverage — the calculus could go the other way. The "Cython
gives you behaviors for free that nanobind makes you write" cost (#2 con)
compounds in proportion to how many edge cases your code has.

## Conventions established (for future maintenance)

These patterns proved out across all four converted modules — apply them when
extending the bindings:

- **Build glue**: `find_package(nanobind CONFIG REQUIRED)` is located by
  running `python -c "import nanobind; print(nanobind.cmake_dir())"` at
  configure time and prepending to `CMAKE_PREFIX_PATH`.
- **ABI3**: when `INFERNA_ABI3=ON`, pass `STABLE_ABI` to
  `nanobind_add_module`.
- **ggml-backend.h**: in TUs that include `whisper.h` or
  `stable-diffusion.h`, forward-declare `ggml_backend_load*` rather than
  including the header — the vendored copies in whisper.cpp/sd.cpp/llama.cpp
  differ enough to trigger redefinition errors. Direct `#include
  "ggml-backend.h"` is fine in the llama TU since it only sees llama.cpp's
  copy.
- **Composition over subclassing for value-returning native classes**:
  nanobind enforces matching `tp_dealloc` between a class and its Python
  subclass, so `__class__` reassignment fails with *"deallocator differs"*.
  Pattern used in `sd/stable_diffusion.py::SDImage`: hold a `_native` handle,
  unwrap at every boundary that calls into native code, wrap return values
  back in the Python class.
- **Subclassing of param/data classes is fine** (no native-returning
  factory), used freely for `SDContextParams`, `SDImageGenParams`,
  `SDContext`, etc.
- **`std::optional<std::string>` for nullable C-string fields** — accepts
  `None` natively from Python and owns the backing storage (so the `const
  char*` in the C struct stays valid for the wrapper's lifetime).
- **Property boilerplate macros** (used in sd + llama):
  - `PARAM_VAL(WrapperT, T, FIELD, NAME)` — POD scalar pass-through.
  - `PARAM_PATH(WrapperT, FIELD, OWNED, NAME)` — owning string-backed
    `const char*`.
- **None-acceptance for `nb::object` setters/callbacks** requires
  `nb::arg("name").none()` qualifier — without it nanobind rejects `None`.
- **`nb::kw_only()` placement** is strict in nanobind 2.x: must precede the
  args that should be keyword-only, not follow them.
- **`_busy_lock` thread-safety guard pattern** preserved 1:1 in whisper +
  sd: a Python `threading.Lock` instance held in the wrapper, exposed
  read-only, acquired non-blocking around any GIL-releasing native call.
  Tests poke it directly (`acquire(blocking=False)` / `release()`) to
  simulate concurrent use.
- **GIL handling**: `nb::gil_scoped_release` around long native calls
  (`whisper_full`, `generate_image`, `llama_decode/encode`, `mg_mgr_poll`).
  `nb::gil_scoped_acquire` inside C callbacks invoked from worker threads.
- **Cross-TU pointer access**: companion `.cpp` files in the same nanobind
  module use a shared `.hpp` declaring `unwrap_*(nb::handle)` functions
  (defined in the primary TU) rather than duplicating struct layouts.
- **Explicit error mapping**: when porting Cython code that relies on
  automatic `OverflowError` / `TypeError` / `FileNotFoundError` from `cdef`
  type coercion, set the Python exception explicitly with
  `PyErr_SetString(PyExc_<Type>, ...)` followed by `throw nb::python_error()`.
- **Explicit enum constants**: nanobind doesn't auto-export C enum values.
  Either bind individually as module attributes (the bulk approach used in
  `_llama_native_enums.cpp`) or use `nb::enum_<>` for a typed enum class
  surface (used for `MtmdInputChunkType`).

## Pitfalls hit during the migration (avoid re-discovering)

- **`std::bad_cast` at module import** — caused by registering an unbound C
  enum as a constructor default. Use `int` with explicit cast inside.
- **`__class__` reassignment fails with "deallocator differs"** — nanobind
  subclasses can't be the target of `__class__` swap from a base instance.
  Fix by composition (hold `_native` handle).
- **`incompatible function arguments ... NoneType`** — nanobind rejects
  `None` for `nb::object` parameters by default. Add `"name"_a.none()`.
- **`Variadic nb::kwargs are implicitly keyword-only; any nb::kw_only()
  annotation must be positioned to reflect that!`** — `nb::kw_only()` must
  *precede* the args it applies to.
- **Default values for hires/cache substructs** — `sd_img_gen_params_init`
  doesn't init nested substructs; e.g. `hires.upscaler` stays at `NONE`
  not `LATENT`. Tests assert the documented defaults — set them explicitly
  in the wrapper ctor. Same caution applies to any C "init" function on
  the llama side.
- **Header conflicts** — including both whisper.h and llama.h in one TU
  fails with redefinition. Split into separate TUs (one `.cpp` per
  backend); forward-declare cross-cutting symbols where needed.
- **Vendored ggml symbols not exported in static lib** — e.g.
  `sd_hires_params_init` is declared in the SD header but not in
  `libstable-diffusion.a`. Inline the field assignments instead.
- **`.gitignore` from the Cython era** — entries like
  `src/inferna/whisper/whisper_cpp.cpp` (originally to ignore Cython-generated
  output) silently swallow new hand-written nanobind sources of the same
  name. Audit and remove all such entries before starting a Cython→nanobind
  rename.
- **`-std=c++17` applied to `.c` files** — `target_compile_options(target
  PRIVATE ${CXX_COMPILE_OPTIONS})` on a target with mixed `.c`/`.cpp`
  sources fails with "invalid argument '-std=c++17' not allowed with 'C'".
  Wrap C++-specific flags in `$<COMPILE_LANGUAGE:CXX>:...>` generator
  expressions.
- **Mismatched extension-module name vs source filename** — the `.so`
  filename and the import path are determined by the `NB_MODULE(name, m)`
  macro, not the `.cpp` filename or the CMake target name. To rename the
  produced module, all three must change together.

## File inventory at end of migration

```
Removed:
  src/inferna/llama/llama_cpp.pyx                     (4318 LOC)
  src/inferna/llama/{llama,ggml,gguf,mtmd,tts_helpers}.pxd  (~2300 LOC)
  src/inferna/llama/{mtmd,speculative,tts_helpers}.pxi      (~975 LOC)
  src/inferna/whisper/whisper_cpp.pyx                 (deleted)
  src/inferna/whisper/whisper.pxd                     (deleted)
  src/inferna/sd/stable_diffusion.pyx                 (deleted)
  src/inferna/sd/stable_diffusion.pxd                 (deleted)
  src/inferna/llama/server/embedded.pyx               (deleted)
  src/inferna/llama/server/mongoose.pxd               (deleted)
  scripts/cmake/{FindCython,UseCython}.cmake          (deleted)

Added:
  src/inferna/llama/_llama_native.cpp                 (~1500 LOC)
  src/inferna/llama/_llama_native_tts.cpp             (~75 LOC)
  src/inferna/llama/_llama_native_mtmd.cpp            (~340 LOC)
  src/inferna/llama/_llama_native_enums.cpp           (~200 LOC)
  src/inferna/llama/_llama_native.hpp                 (20 LOC)
  src/inferna/llama/llama_cpp.py                      (~280 LOC, public facade)
  src/inferna/llama/_python_helpers.py                (~470 LOC)
  src/inferna/llama/_speculative.py                   (~180 LOC)
  src/inferna/whisper/_whisper_native.cpp             (~640 LOC)
  src/inferna/whisper/whisper_cpp.py                  (~30 LOC, public facade)
  src/inferna/sd/_sd_native.cpp                       (existing)
  src/inferna/sd/stable_diffusion.py                  (existing, public facade)
  src/inferna/llama/server/_mongoose.cpp              (existing)
  src/inferna/llama/server/embedded.py                (existing, public facade)

Build glue changes:
  CMakeLists.txt           — dropped Cython entirely; all extensions use
                             nanobind_add_module
  pyproject.toml           — [build-system].requires now ["scikit-build-core",
                             "nanobind>=2.12.0"]; cython dropped from dev deps
  .gitignore               — stale Cython-generated paths removed
```

## Build / test

```bash
# Clean build
uv sync --reinstall-package inferna

# Full test suite
uv run pytest tests/
# → 1472 passed, 31 skipped
```
