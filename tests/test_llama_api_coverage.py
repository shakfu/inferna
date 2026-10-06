"""Drift checks between the llama.cpp headers and the nanobind bindings.

Covers ``llama.h``, ``mtmd.h``, ``mtmd-helper.h`` and ``gguf.h``. A llama.cpp
bump that adds a function, an enum value or a field of a struct the bindings
use fails here until the binding uses it or ``KNOWN_UNWRAPPED`` /
``KNOWN_UNBOUND_FIELDS`` records why not. A name counts as bound when it
appears in the C++ sources under ``src/inferna/llama`` outside comments and
string literals; the check is by name, not signature. The headers are the
vendored copies the extension was compiled against.
"""

import re
from pathlib import Path

import pytest

import inferna.llama.llama_cpp as cy
from inferna.llama import _llama_native as N

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = ROOT / "thirdparty" / "llama.cpp" / "include"
NATIVE_SOURCES = sorted((ROOT / "src" / "inferna" / "llama").rglob("*.[ch]pp"))

pytestmark = pytest.mark.skipif(not (INCLUDE / "llama.h").exists(), reason="llama.cpp headers not built")

# header -> (export macro, symbol prefix)
HEADERS = {
    "llama.h": ("LLAMA_API", "llama_"),
    "mtmd.h": ("MTMD_API", "mtmd_"),
    "mtmd-helper.h": ("MTMD_API", "mtmd_"),
    "gguf.h": ("GGML_API", "gguf_"),
}

_MTMD_BATCH = "multi-chunk mmproj batch encoding; not bound yet"
_MTMD_GEN_AUDIO = "audio generation pipeline; not bound yet"
_MTMD_VIDEO = "video input; not bound yet"
_MTMD_CHUNK = "chunk copy and serialization; not bound yet"
_MTMD_IMAGE_TOKENS = "image-token introspection; MtmdInputChunk does not expose mtmd_image_tokens"
_GGUF_RAW = "raw pointer access; typed getters/setters are bound"
_GGUF_WRITE = "tensor writing; GGUFContext edits metadata only"

# Non-deprecated functions the bindings do not call, with the reason.
# Delete an entry when the function is wrapped.
KNOWN_UNWRAPPED = {
    "llama.h": {
        # llama-context.cpp implements each as its _ext variant with flags=0, which is wrapped
        "llama_state_seq_get_size": "covered by llama_state_seq_get_size_ext",
        "llama_state_seq_get_data": "covered by llama_state_seq_get_data_ext",
        "llama_state_seq_set_data": "covered by llama_state_seq_set_data_ext",
        "llama_load_mode_from_str": "reimplemented over llama_load_mode_name; llama.cpp aborts on an unknown name",
        "llama_batch_get_one": "reimplemented as an owned LlamaBatch; llama.cpp's borrows the token buffer",
    },
    "mtmd.h": {
        "mtmd_batch_init": _MTMD_BATCH,
        "mtmd_batch_free": _MTMD_BATCH,
        "mtmd_batch_add_chunk": _MTMD_BATCH,
        "mtmd_batch_encode": _MTMD_BATCH,
        "mtmd_batch_get_output_embd": _MTMD_BATCH,
        "mtmd_bitmap_init_lazy": _MTMD_VIDEO,
        "mtmd_bitmap_set_mergeable": _MTMD_VIDEO,
        "mtmd_gen_audio_get_info": _MTMD_GEN_AUDIO,
        "mtmd_gen_audio_process": _MTMD_GEN_AUDIO,
        "mtmd_gen_inp_default": _MTMD_GEN_AUDIO,
        "mtmd_input_chunk_copy": _MTMD_CHUNK,
        "mtmd_input_chunk_get_placeholder": _MTMD_CHUNK,
        "mtmd_input_chunk_save": _MTMD_CHUNK,
        "mtmd_input_chunk_load": _MTMD_CHUNK,
        "mtmd_input_chunk_get_tokens_image": _MTMD_IMAGE_TOKENS,
        "mtmd_image_tokens_get_n_tokens": _MTMD_IMAGE_TOKENS,
        "mtmd_image_tokens_get_id": _MTMD_IMAGE_TOKENS,
        "mtmd_image_tokens_get_n_pos": _MTMD_IMAGE_TOKENS,
        "mtmd_image_tokens_get_decoder_pos": _MTMD_IMAGE_TOKENS,
        "mtmd_get_memory_usage": "C++ only (returns std::map), marked unstable upstream",
        "mtmd_get_cap_from_file": "not bound yet",
        "mtmd_tokenize_from_parts": "not bound yet; mtmd_tokenize is",
        "mtmd_log_set": "not bound yet; logging goes through llama_log_set",
        "mtmd_test_create_input_chunks": "upstream test helper",
    },
    "mtmd-helper.h": {
        **{
            f"mtmd_helper_gen_audio_{n}": _MTMD_GEN_AUDIO
            for n in ("init", "free", "reset", "set_input", "step_prompt", "step_gen", "get_output")
        },
        **{
            f"mtmd_helper_video_{n}": _MTMD_VIDEO
            for n in ("init", "init_from_buf", "init_params_default", "free", "get_info", "read_next")
        },
        "mtmd_helper_support_video": _MTMD_VIDEO,
        "mtmd_helper_decode_image_chunk": "eval_chunks covers decoding; mtmd_helper_eval_chunks is bound",
        "mtmd_helper_eval_chunk_single": "eval_chunks covers decoding; mtmd_helper_eval_chunks is bound",
        "mtmd_helper_image_get_decoder_pos": _MTMD_IMAGE_TOKENS,
        "mtmd_helper_log_set": "not bound yet; logging goes through llama_log_set",
    },
    "gguf.h": {
        "gguf_init_from_buffer": "not bound yet; GGUFContext loads from a file",
        "gguf_init_from_callback": "not bound yet; GGUFContext loads from a file",
        "gguf_get_val_data": _GGUF_RAW,
        "gguf_get_arr_data": _GGUF_RAW,
        "gguf_set_arr_data": _GGUF_RAW,
        "gguf_get_meta_data": _GGUF_RAW,
        "gguf_set_kv": "not bound yet",
        "gguf_get_tensor_ne": _GGUF_WRITE,
        "gguf_add_tensor": _GGUF_WRITE,
        "gguf_set_tensor_type": _GGUF_WRITE,
        "gguf_set_tensor_data": _GGUF_WRITE,
        "gguf_write_to_file_ptr": "takes a FILE*; gguf_write_to_file is bound",
    },
}

# Fields of structs the bindings use that they leave unset, as "struct.field".
KNOWN_UNBOUND_FIELDS = {
    "llama_context_params.defrag_thold": "deprecated upstream",
    "llama_context_params.cb_eval": "graph-eval callback; fires per node on ggml worker threads",
    "llama_context_params.cb_eval_user_data": "graph-eval callback; fires per node on ggml worker threads",
    "llama_context_params.abort_callback": "cancellation uses llama_set_abort_callback",
    "llama_context_params.abort_callback_data": "cancellation uses llama_set_abort_callback",
    "llama_context_params.ctx_other": "pointer to a second context for MTP drafting; lifetime not designed",
    "llama_model_params.devices": "NULL-terminated device list; not bound yet, split_mode and main_gpu are",
    "llama_model_quantize_params.tt_overrides": "pointer array; not bound yet",
    "llama_model_quantize_params.prune_layers": "pointer array; not bound yet",
    **{
        f"llama_sampler_i.{f}": "optional; Python samplers leave it NULL"
        for f in (
            "backend_init",
            "backend_accept",
            "backend_apply",
            "backend_set_input",
            "backend_reset",
            "copy_state",
        )
    },
    "mtmd_context_params.image_marker": "deprecated upstream; media_marker is bound",
    "mtmd_context_params.cb_eval": "graph-eval callback; fires per node on ggml worker threads",
    "mtmd_context_params.cb_eval_user_data": "graph-eval callback; fires per node on ggml worker threads",
    "mtmd_context_params.batch_max_tokens": "used by the mtmd batch API, which is not bound",
}

# Enumerators of mtmd.h / mtmd-helper.h / gguf.h the bindings do not use.
# *_COUNT sentinels are skipped.
KNOWN_UNBOUND_ENUMERATORS = {
    **{
        f"MTMD_GEN_{n}": _MTMD_GEN_AUDIO
        for n in (
            "AUDIO_TYPE_NONE",
            "AUDIO_TYPE_POCKETTTS",
            "AUDIO_TYPE_QWEN3TTS",
            "PROCESS_TYPE_GEN_CODE",
            "PROCESS_TYPE_GEN_WAV",
        )
    },
    "MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM": _MTMD_GEN_AUDIO,
    "MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV": _MTMD_GEN_AUDIO,
}

# ggml enums the bindings export in full; others (ggml_op, ...) are exported selectively.
GGML_ENUMS = ("ggml_type", "ggml_status", "ggml_prec", "ggml_log_level", "ggml_tri_type")


def _strip_c(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", re.sub(r"//[^\n]*", "", text), flags=re.S)
    return re.sub(r"^[ \t]*#[^\n]*", "", text, flags=re.M)


def _header(name: str) -> str:
    return _strip_c((INCLUDE / name).read_text())


def _api_functions(header: str) -> set:
    """Non-deprecated exported function names in ``header``."""
    macro, prefix = HEADERS[header]
    names = set()
    for decl in re.findall(rf"(?:DEPRECATED\s*\(\s*)?{macro}\b[^;]*;", _header(header), re.S):
        if "DEPRECATED" in decl:
            continue
        m = re.search(rf"\b({prefix}\w+)\s*\(", decl)
        if m:
            names.add(m.group(1))
    return names


def _member_name(decl: str):
    m = re.search(r"\(\s*\*\s*(\w+)\s*\)", decl)  # function-pointer member
    if m:
        return m.group(1)
    m = re.search(r"(\w+)\s*(\[[^\]]*\])?\s*$", decl.strip())
    return m.group(1) if m else None


def _struct_fields(header: str) -> dict:
    """Map each ``struct <prefix>* { ... }`` in ``header`` to its field names."""
    text = _header(header)
    out = {}
    for m in re.finditer(rf"\bstruct\s+({HEADERS[header][1]}\w+)\s*\{{", text):
        depth, i = 1, m.end()
        while depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        body = re.sub(r"\bunion\s*\{|\}", "", text[m.end() : i - 1])  # flatten unions
        out[m.group(1)] = {n for d in body.split(";") if d.strip() and (n := _member_name(d))}
    return out


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
    text = "".join(_strip_c(p.read_text(errors="ignore")) for p in NATIVE_SOURCES)
    # a name inside an error message is not a binding
    return re.sub(r'"(?:[^"\\\n]|\\.)*"', '""', text)


def _used(name: str, source: str) -> bool:
    return re.search(rf"\b{name}\b", source) is not None


@pytest.mark.parametrize("header", list(HEADERS))
def test_every_api_function_is_wrapped(native_source, header):
    unwrapped = {n for n in _api_functions(header) if not _used(n, native_source)}
    new = sorted(unwrapped - KNOWN_UNWRAPPED[header].keys())
    assert not new, f"{header} functions with no binding (wrap them, or add to KNOWN_UNWRAPPED): {new}"


@pytest.mark.parametrize("header", list(HEADERS))
def test_known_unwrapped_is_current(native_source, header):
    api = _api_functions(header)
    gone = sorted(n for n in KNOWN_UNWRAPPED[header] if n not in api)
    assert not gone, f"KNOWN_UNWRAPPED names no longer in {header}: {gone}"
    wrapped = sorted(n for n in KNOWN_UNWRAPPED[header] if _used(n, native_source))
    assert not wrapped, f"now wrapped; remove from KNOWN_UNWRAPPED: {wrapped}"


def _used_struct_cases():
    return [(h, s) for h in HEADERS for s in sorted(_struct_fields(h))]


@pytest.mark.parametrize("header,struct", _used_struct_cases(), ids=[c[1] for c in _used_struct_cases()])
def test_struct_fields_are_bound(native_source, header, struct):
    """Every field of a struct the bindings use is set, read, or listed with a reason."""
    if not _used(struct, native_source):
        pytest.skip("struct not used by the bindings")
    fields = _struct_fields(header)[struct]
    missing = sorted(
        f for f in fields if not _used(f, native_source) and f"{struct}.{f}" not in KNOWN_UNBOUND_FIELDS
    )
    assert not missing, f"{struct} fields with no binding (bind them, or add to KNOWN_UNBOUND_FIELDS): {missing}"


def test_known_unbound_fields_are_current(native_source):
    fields = {f"{s}.{f}" for h in HEADERS for s, fs in _struct_fields(h).items() for f in fs}
    gone = sorted(k for k in KNOWN_UNBOUND_FIELDS if k not in fields)
    assert not gone, f"KNOWN_UNBOUND_FIELDS names no longer in the headers: {gone}"
    bound = sorted(k for k in KNOWN_UNBOUND_FIELDS if _used(k.split(".")[1], native_source))
    assert not bound, f"now bound; remove from KNOWN_UNBOUND_FIELDS: {bound}"


def _enum_cases():
    cases = list(_enums("llama.h", "llama_").items())
    ggml = _enums("ggml.h", "ggml_")
    cases += [(name, ggml[name]) for name in GGML_ENUMS]
    cases += list(_enums("gguf.h", "gguf_").items())
    return cases


@pytest.mark.parametrize("name,values", _enum_cases(), ids=[c[0] for c in _enum_cases()])
def test_enum_matches_header(name, values):
    missing = sorted(k for k in values if not hasattr(N, k))
    assert not missing, f"{name}: not exported by _llama_native_enums.cpp: {missing}"
    wrong = {k: (getattr(N, k), v) for k, v in values.items() if getattr(N, k) != v}
    assert not wrong, f"{name}: (exported, header) values differ: {wrong}"


def _enumerators(header: str) -> set:
    names = {k for values in _enums(header, HEADERS[header][1]).values() for k in values}
    return {k for k in names if not k.endswith("_COUNT")}


@pytest.mark.parametrize("header", ["mtmd.h", "mtmd-helper.h", "gguf.h"])
def test_enumerators_are_used(native_source, header):
    missing = sorted(
        k for k in _enumerators(header) if not _used(k, native_source) and k not in KNOWN_UNBOUND_ENUMERATORS
    )
    assert not missing, f"{header} enumerators with no binding (bind them, or add to KNOWN_UNBOUND_ENUMERATORS): {missing}"


def test_known_unbound_enumerators_are_current(native_source):
    names = set().union(*(_enumerators(h) for h in ("mtmd.h", "mtmd-helper.h", "gguf.h")))
    gone = sorted(k for k in KNOWN_UNBOUND_ENUMERATORS if k not in names)
    assert not gone, f"KNOWN_UNBOUND_ENUMERATORS names no longer in the headers: {gone}"
    bound = sorted(k for k in KNOWN_UNBOUND_ENUMERATORS if _used(k, native_source))
    assert not bound, f"now bound; remove from KNOWN_UNBOUND_ENUMERATORS: {bound}"


def test_header_parse_is_not_empty():
    # the regexes would otherwise pass vacuously after a header reformat
    assert len(_api_functions("llama.h")) > 200
    assert "llama_decode" in _api_functions("llama.h")
    assert "mtmd_tokenize" in _api_functions("mtmd.h")
    assert "gguf_init_from_file" in _api_functions("gguf.h")
    assert "n_batch" in _struct_fields("llama.h")["llama_context_params"]


def test_facade_reexports_native():
    """``llama_cpp`` lists its re-exports by hand; nothing native may be left out."""
    chunk_types = set(N.MtmdInputChunkType.__members__)
    missing = sorted(k for k in dir(N) if not k.startswith("_") and k not in chunk_types and not hasattr(cy, k))
    assert not missing, f"native names missing from inferna.llama.llama_cpp: {missing}"


def test_llama_version_format():
    assert re.fullmatch(r"\d+\.\d+\.\d+(-\w+)?", cy.llama_version())
