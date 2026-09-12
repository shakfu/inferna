# abi3 wheels

Every published inferna wheel is abi3: built against the CPython stable ABI and tagged `cp312-abi3-<plat>`. One wheel per platform and backend imports on CPython 3.12 and later. `requires-python` is `>=3.12`. cyllama uses the same scheme from its `0.3.0`; see its `docs/dev/abi3.md`.

## Enabling

abi3 is a CMake option, `INFERNA_ABI3`, default `OFF`. With it on, each `nanobind_add_module` call gets `STABLE_ABI`, and `find_package(Python)` adds `Development.SABIModule`.

The wheel tag is set separately, through scikit-build-core's `wheel.py-api`. Both settings are needed: the CMake option alone yields `.abi3.so` modules in a `cp312-cp312` wheel.

| Context | How |
|---|---|
| Local wheel | `make wheel-abi3`, or `make wheel-<backend>-dynamic-abi3` |
| Local editable | `make dev-abi3` |
| pip | `pip install . --config-settings=cmake.define.INFERNA_ABI3=ON --config-settings=wheel.py-api=cp312` |
| CI | `SKBUILD_CMAKE_DEFINE="INFERNA_ABI3=ON" SKBUILD_WHEEL_PY_API=cp312` with `CIBW_BUILD="cp312-*"` |

Release wheels come from `build-cibw-abi3.yml` (CPU/Metal) and `build-gpu-wheels-abi3.yml` (GPU variants). `wheel.py-api` is not set in `[tool.scikit-build]`: that would tag every wheel abi3, including ones compiled without `STABLE_ABI`.

## Releasing

Pushing a bare-semver tag runs both `build-cibw-abi3.yml` (CPU/Metal) and `build-gpu-wheels-abi3.yml` (GPU), and each attaches its wheels to the GitHub release for that tag:

```
git tag 0.3.2 && git push origin 0.3.2
```

The release body is the `## [<tag>]` section of `CHANGELOG.md`, extracted by `scripts/release_notes.py`; with no such section it falls back to `## [Unreleased]`, then to GitHub's generated notes. This publishes to GitHub only -- PyPI uploads stay manual.

The release is always created as a **prerelease**. Promoting it is a separate manual decision:

```
gh release edit 0.3.2 --prerelease=false
```

That is what makes partial state acceptable. Uploads are additive (`gh release upload --clobber`): a release carrying five of six GPU backends is a normal intermediate state, and re-running the one leg that failed tops it up. Neither `publish` job is gated on its whole matrix succeeding, and neither refuses to run against an existing release.

Both workflows publish to the same tag via `softprops/action-gh-release`, which creates the release once and updates it thereafter, so either order works and either workflow can be re-run.

A tag that disagrees with `version` in `pyproject.toml` is a warning annotation, not a failure: the tag names the release, pyproject names the wheel files.

To start a tag over, the release has to go too -- it outlives its tag ref and keeps its assets:

```
gh release delete 0.3.2 --cleanup-tag
git tag -d 0.3.2
```

`workflow_dispatch` against a tag repeats the whole thing, including republishing. Against a branch it builds and smoke tests only.

## Why 3.12

nanobind supports the stable ABI only on 3.12+. It defines `Py_LIMITED_API=0x030C0000`. On an older interpreter it drops `STABLE_ABI` silently and builds a per-version extension (`nanobind-config.cmake`, the check under "Stable ABI builds require CPython >= 3.12"). It also errors if scikit-build-core requests a stable-ABI version below 3.12.

## Verification

`build-cibw-abi3.yml` runs two checks after the build:

- It asserts each native module has an `.abi3.` suffix, which catches the silent fallback above.

- It installs the one wheel on 3.12, 3.13, and 3.14 on each platform and runs an import and inference smoke test. cibuildwheel itself tests only on cp312.

## Limits

- The 3.12 floor is fixed for the life of the wheel tag. C API added in 3.13+ is unusable unless the floor is raised.

- Free-threaded interpreters (`cp313t`, `cp314t`) cannot load abi3 wheels. They would need their own per-version wheels.
