"""Drift checks between stable-diffusion.h and the nanobind bindings.

A stable-diffusion.cpp bump that adds a function or an enum value fails here
until the binding wraps it or ``KNOWN_UNWRAPPED`` records why it does not.
The header is the vendored copy that the extension was compiled against.
"""

import re
from pathlib import Path

import pytest

from inferna.sd import _sd_native as N

ROOT = Path(__file__).resolve().parent.parent
HEADER = ROOT / "thirdparty" / "stable-diffusion.cpp" / "include" / "stable-diffusion.h"
NATIVE_SOURCES = sorted((ROOT / "src" / "inferna" / "sd").rglob("*.[ch]pp"))

pytestmark = pytest.mark.skipif(not HEADER.exists(), reason="stable-diffusion.cpp headers not built")

# stable-diffusion.h functions the bindings do not call, with the reason.
# Delete an entry when the function is wrapped.
KNOWN_UNWRAPPED = {
    "sd_cache_params_init": "called by sd_img_gen_params_init, which is wrapped",
    "sd_hires_params_init": "called by sd_img_gen_params_init, which is wrapped",
    "convert": "convert_with_components covers it; convert calls it with no components and n_threads=0",
    "sd_set_backend_eval_callback": (
        "fires per graph node on ggml worker threads with raw ggml_tensor*; "
        "imatrix collection, its practical use, is wrapped"
    ),
}


def _strip_c(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", re.sub(r"//[^\n]*", "", text), flags=re.S)
    return re.sub(r"^\s*#.*$", "", text, flags=re.M)


def _api_functions() -> set:
    """``SD_API`` function names in stable-diffusion.h."""
    names = set()
    for decl in re.findall(r"SD_API\b[^;]*;", _strip_c(HEADER.read_text()), re.S):
        m = re.search(r"\b(\w+)\s*\(", decl)
        if m:
            names.add(m.group(1))
    return names


def _enums() -> dict:
    out = {}
    for m in re.finditer(r"enum\s+(\w+)\s*\{(.*?)\}", _strip_c(HEADER.read_text()), re.S):
        values, val = {}, -1
        for item in (x.strip() for x in m.group(2).split(",")):
            if not item:
                continue
            key, _, expr = (x.strip() for x in item.partition("="))
            val = eval(expr, {}, dict(values)) if expr else val + 1  # noqa: S307 -- trusted header
            values[key] = val
        out[m.group(1)] = values
    return out


@pytest.fixture(scope="module")
def native_source() -> str:
    return "".join(_strip_c(p.read_text(errors="ignore")) for p in NATIVE_SOURCES)


def test_every_sd_api_function_is_wrapped(native_source):
    unwrapped = {n for n in _api_functions() if not re.search(rf"\b{n}\b", native_source)}
    new = sorted(unwrapped - KNOWN_UNWRAPPED.keys())
    assert not new, f"stable-diffusion.h functions with no binding (wrap them, or add to KNOWN_UNWRAPPED): {new}"


def test_known_unwrapped_is_current(native_source):
    api = _api_functions()
    gone = sorted(n for n in KNOWN_UNWRAPPED if n not in api)
    assert not gone, f"KNOWN_UNWRAPPED names no longer in stable-diffusion.h: {gone}"
    wrapped = sorted(n for n in KNOWN_UNWRAPPED if re.search(rf"\b{n}\b", native_source))
    assert not wrapped, f"now wrapped; remove from KNOWN_UNWRAPPED: {wrapped}"


@pytest.mark.parametrize("name", sorted(_enums()) if HEADER.exists() else [])
def test_enum_matches_header(name):
    values = _enums()[name]
    missing = sorted(k for k in values if k not in N.ENUMS)
    assert not missing, f"{name}: not in _sd_native.ENUMS: {missing}"
    wrong = {k: (N.ENUMS[k], v) for k, v in values.items() if N.ENUMS[k] != v}
    assert not wrong, f"{name}: (exported, header) values differ: {wrong}"
