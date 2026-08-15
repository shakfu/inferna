#!/usr/bin/env python3
"""Audit a built wheel's external runtime dependencies against the project's
wheel-repair exclude lists.

Catches drift between what the GPU build actually links against and what
``WHEEL_REPAIR_EXCLUDES_*`` in ``scripts/manage.py`` tells auditwheel /
delocate to leave external. If oneAPI / CUDA / ROCm bumps a soname and we
forget to update the exclude list, auditwheel either bloats the wheel by
bundling vendor runtimes or fails outright; running this auditor on the
built wheel in CI flags the mismatch before a release.

Linux: walks every ELF ``*.so*`` in the wheel, parses ``DT_NEEDED`` via
``llvm-readelf``/``readelf``. An entry is considered acceptable if (a)
it carries the auditwheel hash suffix ``-<8hex>.so`` (so the lib is
bundled into ``<pkg>.libs/`` with its NEEDED rewritten), (b) it's on
the manylinux baseline (libc/libm/libpthread/libdl/librt/libstdc++/
libgcc_s/ld-linux), or (c) it matches a pattern in
``WHEEL_REPAIR_EXCLUDES_LINUX[<backend>]`` (fnmatch globs honored — the
SYCL list uses ``libmkl_*.so*``).

macOS: walks every Mach-O ``*.dylib``/``*.so`` via ``otool -L``. An
install_name is acceptable if it points into ``<pkg>/.dylibs/`` (where
delocate bundles), lives under ``/usr/lib/`` or ``/System/`` (OS-supplied),
or contains a substring from ``WHEEL_REPAIR_EXCLUDES_DARWIN[<backend>]``
(delocate matches by substring, not glob).

Backend defaults to whatever appears between the first ``_`` and the
``-<version>`` in the wheel filename (``cyllama_sycl-...`` -> ``sycl``;
``inferna-0.1.6-...`` -> no backend, treated as CPU/base build).

Exit codes:
    0 — no unexpected NEEDED entries
    1 — at least one unexpected NEEDED entry was found
    2 — usage / tooling error (e.g. no readelf available)

Example::

    python scripts/audit_wheel.py dist/cyllama_sycl-0.3.0-*.whl
    python scripts/audit_wheel.py dist/inferna_cuda-*.whl --backend cuda
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# Hoisted constants in manage.py; importing here is the single source of truth.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import manage  # noqa: E402

# manylinux_2_28+ allows these to remain external without an explicit
# `--exclude`; auditwheel's policy file does the same. Anything outside
# this set must be either bundled (hash-suffixed in NEEDED) or matched by
# the backend's exclude list — otherwise the wheel would bundle it.
MANYLINUX_BASELINE = frozenset({
    "libc.so.6",
    "libm.so.6",
    "libpthread.so.0",
    "libdl.so.2",
    "librt.so.1",
    "libutil.so.1",
    "libresolv.so.2",
    "libstdc++.so.6",
    "libgcc_s.so.1",
    "ld-linux-x86-64.so.2",
    "ld-linux-aarch64.so.1",
})

# auditwheel renames bundled libs to "<orig>-<8 hex chars>.so..." and
# rewrites the parent's DT_NEEDED to the new name. A NEEDED entry matching
# this pattern is therefore evidence of bundling, not an external dep.
BUNDLED_HASH_RE = re.compile(r"-[0-9a-f]{8}\.so")


# ---------------------------------------------------------------------------
# Tool discovery


def _find_readelf() -> str:
    """Return a path to a working readelf binary.

    Tries llvm-readelf first (ships with Homebrew/Xcode/Clang and is
    portable across hosts), then falls back to GNU readelf.
    """
    for cand in (
        "llvm-readelf",
        "readelf",
        "/opt/homebrew/opt/llvm/bin/llvm-readelf",
        "/usr/local/opt/llvm/bin/llvm-readelf",
    ):
        if cand.startswith("/"):
            if Path(cand).is_file():
                return cand
        else:
            found = shutil.which(cand)
            if found:
                return found
    print("ERROR: no readelf found (install LLVM or binutils).", file=sys.stderr)
    sys.exit(2)


def _find_otool() -> str:
    found = shutil.which("otool")
    if not found:
        print("ERROR: otool not available; macOS wheel audit requires Xcode CLI tools.", file=sys.stderr)
        sys.exit(2)
    return found


# ---------------------------------------------------------------------------
# Wheel walking


def _extract_wheel(wheel: Path, into: Path) -> None:
    with zipfile.ZipFile(wheel) as zf:
        zf.extractall(into)


# The wheel's distribution tag is not always the key used by the exclude
# lists in manage.py: the published variants are versioned (`cuda12`,
# `cuda13`) or named for the vendor SDK rather than the ggml backend
# (`rocm` builds the `hip` backend). Anything not listed here is assumed to
# already be an exclude-list key. Keep in sync with ALLOWED_VARIANTS in
# scripts/ci_rename_package.py.
_WHEEL_TAG_TO_BACKEND: dict[str, str] = {
    "cuda12": "cuda",
    "cuda13": "cuda",
    "rocm": "hip",
}


class UnknownBackendError(ValueError):
    """The wheel carries a backend tag that maps to no exclude list."""


def _detect_backend(wheel_name: str) -> str:
    """Pull the backend tag out of the wheel filename.

    Convention (matches both inferna and cyllama): ``<pkg>[_<backend>]-<version>-...``.
    Returns ``""`` for plain CPU/base wheels (no backend suffix).

    Raises UnknownBackendError when a backend tag is present but maps to no
    exclude list. Falling back to ``""`` there would audit a GPU wheel against
    the CPU list, which allows nothing -- every legitimately runtime-supplied
    driver library then reports as an unexpected dependency, blaming the wheel
    for what is really a gap in this mapping.
    """
    # The tag may contain digits (`cuda12`), so it cannot be `[a-z]+`.
    m = re.match(r"[A-Za-z0-9]+(?:_([a-z][a-z0-9]*))?-\d", wheel_name)
    if not m:
        return ""
    tag = m.group(1) or ""
    if not tag:
        return ""
    backend = _WHEEL_TAG_TO_BACKEND.get(tag, tag)
    if backend not in manage.WHEEL_REPAIR_EXCLUDES_LINUX:
        raise UnknownBackendError(
            f"wheel {wheel_name!r} has backend tag {tag!r}, which maps to no entry in "
            f"manage.py:WHEEL_REPAIR_EXCLUDES_LINUX (tried {backend!r}). Add it to "
            f"_WHEEL_TAG_TO_BACKEND in this script, or pass --backend explicitly."
        )
    return backend


def _wheel_platform(wheel_name: str) -> str:
    """Classify the wheel as linux / darwin / windows via its tag."""
    if "manylinux" in wheel_name or "linux_" in wheel_name:
        return "linux"
    if "macosx" in wheel_name:
        return "darwin"
    if "win_" in wheel_name or "win32" in wheel_name:
        return "windows"
    return ""


# ---------------------------------------------------------------------------
# Per-platform audit


_NEEDED_RE = re.compile(r"\(NEEDED\)\s+Shared library:\s+\[(.+?)\]")


def _so_needed(path: Path, readelf: str) -> set[str]:
    """Return the DT_NEEDED set for an ELF file, or empty if it's not ELF."""
    try:
        out = subprocess.run(
            [readelf, "-d", str(path)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    except OSError:
        return set()
    return set(_NEEDED_RE.findall(out))


def _audit_linux(wheel: Path, backend: str, root: Path) -> list[tuple[str, set[str]]]:
    """Return [(needed_lib, set_of_so_filenames)] for every unexpected entry."""
    readelf = _find_readelf()
    excludes = manage.WHEEL_REPAIR_EXCLUDES_LINUX.get(backend, [])

    # Collect NEEDED → which .so files referenced it. Multiple references
    # to the same lib are merged; the per-.so list is printed for context.
    refs: dict[str, set[str]] = {}
    for so in sorted(root.rglob("*.so*")):
        if not so.is_file() or so.is_symlink():
            continue
        for lib in _so_needed(so, readelf):
            refs.setdefault(lib, set()).add(str(so.relative_to(root)))

    unexpected: list[tuple[str, set[str]]] = []
    for lib in sorted(refs):
        if lib in MANYLINUX_BASELINE:
            continue
        if BUNDLED_HASH_RE.search(lib):
            # Bundled into <pkg>.libs/ — auditwheel rewrote the NEEDED.
            continue
        if any(fnmatch.fnmatchcase(lib, pat) for pat in excludes):
            continue
        unexpected.append((lib, refs[lib]))
    return unexpected


_DYLIB_LINE_RE = re.compile(r"^\s+(\S+)\s+\(compatibility version", re.M)


def _dylib_install_names(path: Path, otool: str) -> set[str]:
    try:
        out = subprocess.run(
            [otool, "-L", str(path)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    except OSError:
        return set()
    return set(_DYLIB_LINE_RE.findall(out))


def _audit_darwin(wheel: Path, backend: str, root: Path) -> list[tuple[str, set[str]]]:
    otool = _find_otool()
    excludes = manage.WHEEL_REPAIR_EXCLUDES_DARWIN.get(backend, manage.WHEEL_REPAIR_DARWIN_BASE)

    refs: dict[str, set[str]] = {}
    for path in sorted(list(root.rglob("*.dylib")) + list(root.rglob("*.so"))):
        if not path.is_file() or path.is_symlink():
            continue
        for name in _dylib_install_names(path, otool):
            refs.setdefault(name, set()).add(str(path.relative_to(root)))

    unexpected: list[tuple[str, set[str]]] = []
    for name in sorted(refs):
        # delocate bundles into <pkg>/.dylibs/ and rewrites install_name to
        # @loader_path/../.dylibs/<name>. Anything routed through @loader_path,
        # @rpath, or @executable_path is bundled-or-relative — not external.
        if name.startswith(("@loader_path", "@rpath", "@executable_path")):
            continue
        # OS-supplied dylibs that delocate (and Apple) consider always-present.
        if name.startswith(("/usr/lib/", "/System/")):
            continue
        # delocate matches --exclude by substring against the install_name.
        if any(pat in name for pat in excludes):
            continue
        unexpected.append((name, refs[name]))
    return unexpected


# ---------------------------------------------------------------------------
# Entry point


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("wheel", type=Path, help="Path to the .whl file to audit.")
    p.add_argument(
        "--backend",
        default=None,
        help="Backend name (cuda/hip/sycl/vulkan/opencl/cpu/metal). "
        "Inferred from the wheel filename if omitted.",
    )
    p.add_argument(
        "--platform",
        choices=["linux", "darwin", "windows"],
        default=None,
        help="Platform classification. Inferred from the wheel filename if omitted.",
    )
    args = p.parse_args(argv)

    wheel: Path = args.wheel
    if not wheel.is_file():
        print(f"ERROR: wheel not found: {wheel}", file=sys.stderr)
        return 2

    try:
        backend = args.backend if args.backend is not None else _detect_backend(wheel.name)
    except UnknownBackendError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    plat = args.platform or _wheel_platform(wheel.name)
    if not plat:
        print(f"ERROR: cannot classify platform from filename: {wheel.name}", file=sys.stderr)
        return 2

    print(f"wheel:    {wheel.name}")
    print(f"backend:  {backend or '(none)'}")
    print(f"platform: {plat}")
    print()

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _extract_wheel(wheel, root)

        if plat == "linux":
            unexpected = _audit_linux(wheel, backend, root)
            exclude_field = "WHEEL_REPAIR_EXCLUDES_LINUX"
        elif plat == "darwin":
            unexpected = _audit_darwin(wheel, backend, root)
            exclude_field = "WHEEL_REPAIR_EXCLUDES_DARWIN"
        else:
            # Windows would need pefile / dumpbin to read PE imports; the
            # delvewheel exclude list is short enough that a misconfiguration
            # is usually caught by the build itself. Treat as a no-op.
            print(f"Windows audit not implemented; nothing to check for {wheel.name}.")
            return 0

    if not unexpected:
        print(f"OK: every external dep is bundled, on the platform baseline, or matched by the {backend!r} exclude list.")
        return 0

    print(f"FAIL: {len(unexpected)} unexpected external dep(s):")
    for lib, sites in unexpected:
        print(f"  {lib}")
        for s in sorted(sites):
            print(f"      <- {s}")
    print()
    print(
        f"These are neither bundled nor matched by manage.py:{exclude_field}[{backend!r}]. "
        "Either add them to that list (if they should remain runtime-supplied) or "
        "investigate why the wheel was built against them."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
