"""Drift checks between llama.h and the nanobind bindings.

A llama.cpp bump that adds a function or an enum value fails here until the
binding wraps it or ``KNOWN_UNWRAPPED`` records why it does not. The header
is the vendored copy that the extension was compiled against.
"""

import re
from pathlib import Path

import pytest

import inferna.llama.llama_cpp as cy
from inferna.llama import _llama_native as N

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = ROOT / "thirdparty" / "llama.cpp" / "include"
NATIVE_SOURCES = sorted((ROOT / "src" / "inferna").rglob("*.[ch]pp"))

pytestmark = pytest.mark.skipif(not (INCLUDE / "llama.h").exists(), reason="llama.cpp headers not built")

# Non-deprecated llama.h functions the bindings do not call, with the reason.
# Delete an entry when the function is wrapped.
KNOWN_UNWRAPPED = {
    # llama-context.cpp implements each as its _ext variant with flags=0, which is wrapped
    "llama_state_seq_get_size": "covered by llama_state_seq_get_size_ext",
    "llama_state_seq_get_data": "covered by llama_state_seq_get_data_ext",
    "llama_state_seq_set_data": "covered by llama_state_seq_set_data_ext",
}

# ggml enums the bindings export in full; others (ggml_op, ...) are exported selectively.
GGML_ENUMS = ("ggml_type", "ggml_status", "ggml_prec", "ggml_log_level", "ggml_tri_type")


def _strip_c(text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", re.sub(r"//[^\n]*", "", text), flags=re.S)


def _header(name: str) -> str:
    return _strip_c((INCLUDE / name).read_text())


def _api_functions() -> set:
    """Non-deprecated ``LLAMA_API`` function names in llama.h."""
    names = set()
    for decl in re.findall(r"(?:DEPRECATED\s*\(\s*)?LLAMA_API\b[^;]*;", _header("llama.h"), re.S):
        if "DEPRECATED" in decl:
            continue
        m = re.search(r"\b(llama_\w+)\s*\(", decl)
        if m:
            names.add(m.group(1))
    return names


def _enums(header: str, prefix: str) -> dict:
    """Map each ``enum <prefix>*`` in ``header`` to {enumerator: value}."""
    text = _header(header)
    out = {}
    for m in re.finditer(rf"enum\s+({prefix}\w*)\s*\{{(.*?)\}}", text, re.S):
        values, val = {}, -1
        for item in (x.strip() for x in m.group(2).split(",")):
            if not item:
                continue
            key, _, expr = (x.strip() for x in item.partition("="))
            # expressions may name earlier enumerators or GGML_* defines, e.g. GGML_ROPE_TYPE_NEOX
            val = eval(expr, {}, {**vars(N), **values}) if expr else val + 1  # noqa: S307 -- trusted header
            values[key] = val
        out[m.group(1)] = values
    return out


@pytest.fixture(scope="module")
def native_source() -> str:
    return "".join(_strip_c(p.read_text(errors="ignore")) for p in NATIVE_SOURCES)


def test_every_llama_api_function_is_wrapped(native_source):
    unwrapped = {n for n in _api_functions() if not re.search(rf"\b{n}\b", native_source)}
    new = sorted(unwrapped - KNOWN_UNWRAPPED.keys())
    assert not new, f"llama.h functions with no binding (wrap them, or add to KNOWN_UNWRAPPED): {new}"


def test_known_unwrapped_is_current(native_source):
    api = _api_functions()
    gone = sorted(n for n in KNOWN_UNWRAPPED if n not in api)
    assert not gone, f"KNOWN_UNWRAPPED names no longer in llama.h: {gone}"
    wrapped = sorted(n for n in KNOWN_UNWRAPPED if re.search(rf"\b{n}\b", native_source))
    assert not wrapped, f"now wrapped; remove from KNOWN_UNWRAPPED: {wrapped}"


def _enum_cases():
    cases = list(_enums("llama.h", "llama_").items())
    ggml = _enums("ggml.h", "ggml_")
    cases += [(name, ggml[name]) for name in GGML_ENUMS]
    return cases


@pytest.mark.parametrize("name,values", _enum_cases(), ids=[c[0] for c in _enum_cases()])
def test_enum_matches_header(name, values):
    missing = sorted(k for k in values if not hasattr(N, k))
    assert not missing, f"{name}: not exported by _llama_native_enums.cpp: {missing}"
    wrong = {k: (getattr(N, k), v) for k, v in values.items() if getattr(N, k) != v}
    assert not wrong, f"{name}: (exported, header) values differ: {wrong}"


def test_facade_reexports_native():
    """``llama_cpp`` lists its re-exports by hand; nothing native may be left out."""
    chunk_types = set(N.MtmdInputChunkType.__members__)
    missing = sorted(k for k in dir(N) if not k.startswith("_") and k not in chunk_types and not hasattr(cy, k))
    assert not missing, f"native names missing from inferna.llama.llama_cpp: {missing}"


def test_llama_version_format():
    assert re.fullmatch(r"\d+\.\d+\.\d+(-\w+)?", cy.llama_version())
