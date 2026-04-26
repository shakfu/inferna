// Cross-TU helpers for the _llama_native nanobind extension.
//
// The wrapper structs (LlamaModelW, LlamaContextW, ...) are defined in
// _llama_native.cpp. Companion TUs (mtmd, tts, etc.) only need to recover the
// underlying llama.cpp pointers from a Python handle, which is cleaner than
// dragging the full struct layouts across translation units.

#pragma once

#include <nanobind/nanobind.h>

struct llama_model;
struct llama_context;

namespace inferna {
    // Unwrap a Python LlamaModel / LlamaContext object back to the native
    // llama.cpp pointer. Throws on type mismatch or null pointer.
    llama_model*   unwrap_model(nanobind::handle h);
    llama_context* unwrap_ctx  (nanobind::handle h);
}
