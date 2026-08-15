"""Tests for `scripts/audit_wheel.py`'s backend detection.

The audit step runs on every built GPU wheel and decides which external
dependencies are allowed to remain runtime-supplied by looking up
`manage.WHEEL_REPAIR_EXCLUDES_LINUX[<backend>]`. The backend is inferred from
the wheel filename, and that filename does not always spell the backend the way
the exclude lists key it: published variants are versioned (`inferna_cuda12`)
or named for the vendor SDK rather than the ggml backend (`inferna_rocm` builds
the `hip` backend).

Getting this wrong is not a no-op. An unresolved tag used to fall back to `""`,
whose exclude list is empty, so every driver library the wheel legitimately
expects at runtime was reported as an unexpected dependency and the audit
failed -- blaming the wheel for a gap in the mapping.
"""

import fnmatch
import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(SCRIPTS))
audit_wheel = _load("audit_wheel")
manage = _load("manage")


WHEEL_SUFFIX = "-0.1.10-cp312-abi3-manylinux_2_35_x86_64.whl"


@pytest.mark.parametrize(
    "dist,expected",
    [
        ("inferna", ""),
        ("inferna_cuda12", "cuda"),
        ("inferna_cuda13", "cuda"),
        ("inferna_rocm", "hip"),
        ("inferna_hip", "hip"),
        ("inferna_vulkan", "vulkan"),
        ("inferna_sycl", "sycl"),
        ("inferna_opencl", "opencl"),
    ],
)
def test_detect_backend(dist, expected):
    assert audit_wheel._detect_backend(dist + WHEEL_SUFFIX) == expected


def test_every_published_variant_resolves():
    """Each name ci_rename_package.py allows must map to a real exclude list."""
    ci_rename = _load("ci_rename_package")

    for variant in sorted(ci_rename.ALLOWED_VARIANTS):
        dist = variant.replace("-", "_")
        backend = audit_wheel._detect_backend(dist + WHEEL_SUFFIX)
        assert backend in manage.WHEEL_REPAIR_EXCLUDES_LINUX, (
            f"{variant} detected as {backend!r}, which has no exclude list"
        )


def test_unknown_backend_tag_raises():
    """An unmapped tag must fail loudly, not silently audit as a CPU wheel."""
    with pytest.raises(audit_wheel.UnknownBackendError):
        audit_wheel._detect_backend("inferna_cuda99" + WHEEL_SUFFIX)


@pytest.mark.parametrize(
    "dist,needed",
    [
        # Exactly what the CUDA and ROCm audits reported as "unexpected" while
        # the backend was misdetected as "" (run 31865880138).
        (
            "inferna_cuda12",
            [
                "libcublas.so.12",
                "libcublasLt.so.12",
                "libcuda.so.1",
                "libcudart.so.12",
                "libgomp.so.1",
            ],
        ),
        (
            "inferna_rocm",
            [
                "libamdhip64.so.6",
                "libgomp.so.1",
                "libhipblas.so.2",
                "librocblas.so.4",
            ],
        ),
    ],
)
def test_driver_libs_are_excluded_for_their_backend(dist, needed):
    """The libs those wheels link against must clear their own exclude list."""
    backend = audit_wheel._detect_backend(dist + WHEEL_SUFFIX)
    excludes = manage.WHEEL_REPAIR_EXCLUDES_LINUX[backend]

    unmatched = [
        lib
        for lib in needed
        if not any(fnmatch.fnmatchcase(lib, pat) for pat in excludes)
    ]
    assert not unmatched, f"{dist} ({backend}) would still fail the audit on: {unmatched}"


def test_cpu_wheel_allows_nothing():
    """The empty-backend list stays empty -- the fallback that caused the bug."""
    assert manage.WHEEL_REPAIR_EXCLUDES_LINUX[""] == []


def test_exclude_list_key_sets_agree():
    """_detect_backend validates against the Linux dict but is used for both.

    The two dicts are keyed the same today; if they ever diverge, a darwin-only
    backend would be rejected as unknown.
    """
    assert set(manage.WHEEL_REPAIR_EXCLUDES_LINUX) == set(manage.WHEEL_REPAIR_EXCLUDES_DARWIN)
