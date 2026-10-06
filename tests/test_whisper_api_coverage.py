"""Drift checks between whisper.h and the nanobind bindings.

A whisper.cpp bump that adds a function or an enum value fails here until the
binding wraps it or ``KNOWN_UNWRAPPED`` records why it does not. The header
is the vendored copy that the extension was compiled against.
"""

import re
from pathlib import Path

import pytest

from inferna.whisper import _whisper_native as N

ROOT = Path(__file__).resolve().parent.parent
HEADER = ROOT / "thirdparty" / "whisper.cpp" / "include" / "whisper.h"
NATIVE_SOURCES = sorted((ROOT / "src" / "inferna" / "whisper").rglob("*.[ch]pp"))

pytestmark = pytest.mark.skipif(not HEADER.exists(), reason="whisper.cpp headers not built")

_INIT = "alternative loader; WhisperContext loads from a file path"

# Non-deprecated whisper.h functions the bindings do not call, with the reason.
# Delete an entry when the function is wrapped.
KNOWN_UNWRAPPED = {
    # FFI helpers for callers that cannot pass structs by value
    "whisper_context_default_params_by_ref": "params are wrapped by value",
    "whisper_full_default_params_by_ref": "params are wrapped by value",
    "whisper_free_context_params": "frees a *_by_ref result",
    "whisper_free_params": "frees a *_by_ref result",
    "whisper_bench_memcpy": "benchmark helper",
    "whisper_bench_memcpy_str": "benchmark helper",
    "whisper_bench_ggml_mul_mat": "benchmark helper",
    "whisper_bench_ggml_mul_mat_str": "benchmark helper",
    "whisper_ctx_init_openvino_encoder": "OpenVINO backend is not built",
    "whisper_ctx_init_openvino_encoder_with_state": "OpenVINO backend is not built",
    "whisper_init_from_buffer_with_params": _INIT,
    "whisper_init_from_buffer_with_params_no_state": _INIT,
    "whisper_init_from_file_with_params_no_state": _INIT,
    "whisper_init_with_params": _INIT,
    "whisper_init_with_params_no_state": _INIT,
    "whisper_vad_init_with_params": "alternative loader; WhisperVadContext loads from a file path",
}

# header enum -> (exported class, enumerator prefix the class drops)
ENUMS = {
    "whisper_alignment_heads_preset": ("WhisperAheadsPreset", "WHISPER_AHEADS_"),
    "whisper_gretype": ("WhisperGretype", "WHISPER_GRETYPE_"),
    "whisper_sampling_strategy": ("WhisperSamplingStrategy", "WHISPER_SAMPLING_"),
}


def _strip_c(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", re.sub(r"//[^\n]*", "", text), flags=re.S)
    return re.sub(r"^\s*#.*$", "", text, flags=re.M)


def _api_functions() -> set:
    """Non-deprecated ``WHISPER_API`` function names in whisper.h."""
    names = set()
    for decl in re.findall(r"(?:WHISPER_DEPRECATED\s*\(\s*)?WHISPER_API\b[^;]*;", _strip_c(HEADER.read_text()), re.S):
        if "DEPRECATED" in decl:
            continue
        m = re.search(r"\b(whisper_\w+)\s*\(", decl)
        if m:
            names.add(m.group(1))
    return names


def _enums() -> dict:
    out = {}
    for m in re.finditer(r"enum\s+(whisper_\w*)\s*\{(.*?)\}", _strip_c(HEADER.read_text()), re.S):
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


def test_every_whisper_api_function_is_wrapped(native_source):
    unwrapped = {n for n in _api_functions() if not re.search(rf"\b{n}\b", native_source)}
    new = sorted(unwrapped - KNOWN_UNWRAPPED.keys())
    assert not new, f"whisper.h functions with no binding (wrap them, or add to KNOWN_UNWRAPPED): {new}"


def test_known_unwrapped_is_current(native_source):
    api = _api_functions()
    gone = sorted(n for n in KNOWN_UNWRAPPED if n not in api)
    assert not gone, f"KNOWN_UNWRAPPED names no longer in whisper.h: {gone}"
    wrapped = sorted(n for n in KNOWN_UNWRAPPED if re.search(rf"\b{n}\b", native_source))
    assert not wrapped, f"now wrapped; remove from KNOWN_UNWRAPPED: {wrapped}"


def test_every_enum_is_mapped():
    assert sorted(_enums()) == sorted(ENUMS)


@pytest.mark.parametrize("name", sorted(ENUMS))
def test_enum_matches_header(name):
    cls_name, prefix = ENUMS[name]
    cls = getattr(N, cls_name)
    header = {k.removeprefix(prefix): v for k, v in _enums()[name].items()}
    exported = {k: getattr(cls, k) for k in header if hasattr(cls, k)}
    assert exported == header
