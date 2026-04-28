// nanobind bindings for llama.cpp. Produces the `_llama_native` extension
// module; the public Python surface lives in `inferna.llama.llama_cpp`,
// which imports from this module and adds pure-Python helpers.
//
// Companion TUs (linked into the same extension): _llama_native_mtmd.cpp
// (multimodal), _llama_native_tts.cpp (TTS helpers), _llama_native_enums.cpp
// (integer enum constant exports). Cross-TU pointer access is done via
// the helpers declared in _llama_native.hpp.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "llama.h"
#include "ggml-backend.h"  // ok here — this TU only sees llama.cpp's vendored copy
#include "ggml-cpu.h"
#include "gguf.h"

#include "_llama_native.hpp"

namespace nb = nanobind;
using namespace nb::literals;

// =============================================================================
// Macros to reduce boilerplate. Reused across the parameter wrappers.
//
//   PARAM_VAL : trivially-bindable POD field (int, float, bool, enum-as-int).
//   PARAM_PATH: const char* field backed by an owning std::optional<std::string>.
// =============================================================================

#define PARAM_VAL(WrapperT, T, FIELD, NAME)                            \
    .def_prop_rw(NAME,                                                 \
        [](WrapperT& s) { return (T) s.p.FIELD; },                     \
        [](WrapperT& s, T v) { s.p.FIELD = (decltype(s.p.FIELD)) v; })

#define PARAM_PATH(WrapperT, FIELD, OWNED, NAME)                                    \
    .def_prop_rw(NAME,                                                              \
        [](WrapperT& s) -> nb::object {                                             \
            if (!s.p.FIELD) return nb::none();                                      \
            return nb::cast(std::string(s.p.FIELD));                                \
        },                                                                          \
        [](WrapperT& s, std::optional<std::string> v) {                             \
            if (!v || v->empty()) { s.p.FIELD = nullptr; s.OWNED.reset(); }         \
            else { s.OWNED = std::move(*v); s.p.FIELD = s.OWNED->c_str(); }         \
        })

// =============================================================================
// Progress-callback bridge (used by LlamaModelParams)
// =============================================================================

// Cross-FFI: llama.cpp invokes this from native code (with the GIL released
// during model loading), passing back the Python callable we stashed in
// progress_callback_user_data.
extern "C" bool _llama_progress_cb(float progress, void* user_data) {
    if (!user_data) return true;
    nb::gil_scoped_acquire gil;
    try {
        nb::object cb = nb::borrow(reinterpret_cast<PyObject*>(user_data));
        nb::object result = cb(progress);
        return nb::cast<bool>(result);
    } catch (...) {
        return true;  // swallow callback errors and continue model loading
    }
}

// =============================================================================
// LlamaModelParams
// =============================================================================

struct LlamaModelParamsW {
    llama_model_params p;
    nb::object progress_callback_obj;     // owns the Python callable lifetime
    std::vector<float> tensor_split_owned;

    LlamaModelParamsW() : p(llama_model_default_params()) {}
};

// =============================================================================
// LlamaContextParams
// =============================================================================

struct LlamaContextParamsW {
    llama_context_params p;
    LlamaContextParamsW() : p(llama_context_default_params()) {}
};

// =============================================================================
// LlamaModelQuantizeParams
// =============================================================================

struct LlamaModelQuantizeParamsW {
    llama_model_quantize_params p;
    LlamaModelQuantizeParamsW() : p(llama_model_quantize_default_params()) {}
};

// =============================================================================
// LlamaSamplerChainParams
// =============================================================================

struct LlamaSamplerChainParamsW {
    llama_sampler_chain_params p;
    LlamaSamplerChainParamsW() : p(llama_sampler_chain_default_params()) {}
};

// =============================================================================
// LlamaChatMessage  (role/content are const char*, must own their backing)
// =============================================================================

struct LlamaChatMessageW {
    llama_chat_message p{};
    std::string role_s;
    std::string content_s;

    LlamaChatMessageW(const std::string& role, const std::string& content)
        : role_s(role), content_s(content)
    {
        p.role = role_s.c_str();
        p.content = content_s.c_str();
    }
};

// =============================================================================
// LlamaTokenData (POD)
// =============================================================================

struct LlamaTokenDataW {
    llama_token_data p{};
};

// =============================================================================
// LlamaLogitBias (POD)
// =============================================================================

struct LlamaLogitBiasW {
    llama_logit_bias p{};
};

// =============================================================================
// LlamaModelKvOverride (flattened tagged union — accessed as tag + 4 vals)
// =============================================================================

struct LlamaModelKvOverrideW {
    llama_model_kv_override p{};
};

// =============================================================================
// LlamaModelTensorBuftOverride
// =============================================================================

struct LlamaModelTensorBuftOverrideW {
    llama_model_tensor_buft_override p{};
    std::optional<std::string> pattern_s;
};

// =============================================================================
// LlamaVocab — non-owning view of a llama_vocab*
// (vocab pointer is owned by the parent LlamaModel.)
// =============================================================================

struct LlamaVocabW {
    const llama_vocab* ptr = nullptr;
    nb::object parent;  // keep parent LlamaModel alive
};

// =============================================================================
// LlamaAdapterLora — non-owning by default (model owns); set owner=true for
// explicit free().
// =============================================================================

struct LlamaAdapterLoraW {
    llama_adapter_lora* ptr = nullptr;
    bool owner = false;
    ~LlamaAdapterLoraW() {
        if (owner && ptr) { llama_adapter_lora_free(ptr); ptr = nullptr; }
    }
    LlamaAdapterLoraW() = default;
    LlamaAdapterLoraW(const LlamaAdapterLoraW&) = delete;
    LlamaAdapterLoraW& operator=(const LlamaAdapterLoraW&) = delete;
};

// =============================================================================
// LlamaModel — owns llama_model*
// =============================================================================

struct LlamaModelW {
    llama_model* ptr = nullptr;
    bool owner = true;
    std::string path_model;
    bool verbose = true;

    // Cached metadata (populated on init).
    int      cached_n_embd      = -1;
    int      cached_n_embd_inp  = -1;
    int      cached_n_layer     = -1;
    int      cached_n_head      = -1;
    int      cached_n_head_kv   = -1;
    int      cached_n_ctx_train = -1;
    uint64_t cached_n_params    = 0;
    uint64_t cached_size        = 0;
    bool     cache_initialized  = false;

    LlamaModelW(const std::string& path, std::optional<LlamaModelParamsW*> p_opt, bool verbose_)
        : path_model(path), verbose(verbose_)
    {
        // Surface clear typed errors before handing the path to llama.cpp.
        nb::module_ validation = nb::module_::import_("inferna.utils.validation");
        validation.attr("validate_gguf_file")(path, "kind"_a = "GGUF model");

        LlamaModelParamsW default_params;
        LlamaModelParamsW* lp = p_opt && *p_opt ? *p_opt : &default_params;

        ptr = llama_model_load_from_file(path.c_str(), lp->p);
        if (!ptr) {
            throw std::invalid_argument(
                "Failed to load model from file: " + path + ". "
                "The file passed format checks but llama.cpp could not load it. "
                "Possible causes: unsupported GGUF version or quantization, "
                "insufficient memory, an aborted progress callback, or a corrupt "
                "tensor section. Run with verbose=True to see detailed errors "
                "from llama.cpp.");
        }
        initialize_cache();
    }

    ~LlamaModelW() {
        if (ptr && owner) { llama_model_free(ptr); ptr = nullptr; }
    }
    LlamaModelW(const LlamaModelW&) = delete;
    LlamaModelW& operator=(const LlamaModelW&) = delete;

    void initialize_cache() {
        if (!ptr || cache_initialized) return;
        cached_n_embd      = llama_model_n_embd(ptr);
        cached_n_embd_inp  = llama_model_n_embd_inp(ptr);
        cached_n_layer     = llama_model_n_layer(ptr);
        cached_n_head      = llama_model_n_head(ptr);
        cached_n_head_kv   = llama_model_n_head_kv(ptr);
        cached_n_ctx_train = llama_model_n_ctx_train(ptr);
        cached_n_params    = llama_model_n_params(ptr);
        cached_size        = llama_model_size(ptr);
        cache_initialized  = true;
    }
};

// =============================================================================
// LlamaBatch — owns llama_batch (allocated by llama_batch_init).
// =============================================================================

struct LlamaBatchW {
    llama_batch p{};
    int n_tokens_capacity = 0;
    int embd = 0;
    int n_seq_max = 1;
    bool owner = true;

    LlamaBatchW(int n_tokens_, int embd_, int n_seq_max_)
        : n_tokens_capacity(n_tokens_), embd(embd_), n_seq_max(n_seq_max_)
    {
        p = llama_batch_init(n_tokens_, embd_, n_seq_max_);
    }
    LlamaBatchW() = default;  // for from_instance fallback paths
    ~LlamaBatchW() { if (owner) llama_batch_free(p); }
    LlamaBatchW(const LlamaBatchW&) = delete;
    LlamaBatchW& operator=(const LlamaBatchW&) = delete;
};

// =============================================================================
// LlamaContext — owns llama_context*
// =============================================================================

// Polled by ggml during decode/encode; aborts the in-flight op when the
// underlying flag is set. See LlamaContext.install_cancel_callback().
extern "C" bool _cancel_flag_callback(void* user_data) {
    return user_data && *static_cast<bool*>(user_data);
}

// Module-level logging callback bridge.
static nb::object g_log_cb;
extern "C" void _llama_log_cb(ggml_log_level level, const char* text, void* /*user_data*/) {
    if (!g_log_cb.is_valid() || g_log_cb.is_none()) return;
    nb::gil_scoped_acquire gil;
    try {
        g_log_cb((int)level, text ? std::string(text) : std::string());
    } catch (...) {}
}
extern "C" void _llama_no_log_cb(ggml_log_level, const char*, void*) {}

struct LlamaContextW {
    llama_context* ptr = nullptr;
    bool owner = true;
    nb::object model_obj;   // keep parent model alive
    bool verbose = true;
    int  n_tokens = 0;
    bool cancel_flag = false;

    // Throw a normal Python exception if the context has been closed,
    // instead of letting llama.cpp dereference a null pointer.
    void ensure_valid() const {
        if (!ptr) throw std::runtime_error(
            "LlamaContext has been closed and is no longer usable");
    }

    LlamaContextW(nb::object model_o, std::optional<LlamaContextParamsW*> p_opt, bool verbose_)
        : model_obj(std::move(model_o)), verbose(verbose_)
    {
        LlamaModelW* model = nullptr;
        try { model = nb::cast<LlamaModelW*>(model_obj); }
        catch (...) {
            throw std::invalid_argument(
                "model must be LlamaModel, got an incompatible type");
        }
        if (!model->ptr) {
            throw std::invalid_argument(
                "model has been freed or is invalid (NULL pointer)");
        }

        LlamaContextParamsW default_params;
        LlamaContextParamsW* cp = p_opt && *p_opt ? *p_opt : &default_params;

        // Mirror the upstream Python sanity check: refuse pathological n_ctx
        // values that would crash the allocator instead of returning NULL.
        // Computed in uint64_t with overflow detection so the check is
        // portable to MSVC (which lacks __int128).
        long long n_ctx_eff = cp->p.n_ctx ? cp->p.n_ctx : model->cached_n_ctx_train;
        if (n_ctx_eff > 0) {
            auto mul_ov = [](uint64_t a, uint64_t b, uint64_t& out) -> bool {
#if defined(__GNUC__) || defined(__clang__)
                return __builtin_mul_overflow(a, b, &out);
#else
                out = a * b;
                return a != 0 && out / a != b;
#endif
            };
            uint64_t est = 4;
            uint64_t tmp = 0;
            bool overflow = false;
            overflow |= mul_ov(est, (uint64_t) model->cached_n_layer, tmp); est = tmp;
            overflow |= mul_ov(est, (uint64_t) n_ctx_eff,             tmp); est = tmp;
            overflow |= mul_ov(est, (uint64_t) model->cached_n_embd,  tmp); est = tmp;
            const uint64_t cap = 100ULL << 40;  // 100 TiB
            if (overflow || est > cap) {
                std::string size_str = overflow
                    ? std::string("more than 16 EiB")
                    : (std::to_string(est >> 30) + " GiB");
                throw std::runtime_error(
                    "Refusing to create llama_context: requested n_ctx=" +
                    std::to_string(n_ctx_eff) + " would need an estimated ~" +
                    size_str + " of KV cache "
                    "(model n_layer=" + std::to_string(model->cached_n_layer) +
                    ", n_embd=" + std::to_string(model->cached_n_embd) + "), "
                    "which exceeds the 100 TiB sanity cap. Lower n_ctx or use "
                    "a smaller model. (This check exists because llama.cpp's "
                    "allocator can segfault rather than return NULL on "
                    "extreme OOM.)");
            }
        }

        ptr = llama_init_from_model(model->ptr, cp->p);
        if (!ptr) {
            throw std::runtime_error(
                "Failed to create llama_context (model=" + model->path_model +
                ", requested n_ctx=" + std::to_string(cp->p.n_ctx) +
                ", model n_ctx_train=" + std::to_string(model->cached_n_ctx_train) +
                "). Common causes: requested context size too large for "
                "available memory (OOM), n_ctx exceeds what the model "
                "supports, or invalid context parameters. Try lowering "
                "n_ctx or n_batch.");
        }
    }

    ~LlamaContextW() {
        if (ptr && owner) { llama_free(ptr); ptr = nullptr; }
    }
    LlamaContextW(const LlamaContextW&) = delete;
    LlamaContextW& operator=(const LlamaContextW&) = delete;
};

// =============================================================================
// LlamaSampler — owns llama_sampler*
// =============================================================================

struct LlamaSamplerW {
    llama_sampler* ptr = nullptr;
    bool owner = true;

    explicit LlamaSamplerW(std::optional<LlamaSamplerChainParamsW*> p_opt) {
        LlamaSamplerChainParamsW default_params;
        LlamaSamplerChainParamsW* cp = p_opt && *p_opt ? *p_opt : &default_params;
        ptr = llama_sampler_chain_init(cp->p);
        if (!ptr) throw std::runtime_error("Failed to init Sampler");
    }
    LlamaSamplerW() = default;  // used by clone()
    ~LlamaSamplerW() {
        if (ptr && owner) { llama_sampler_free(ptr); ptr = nullptr; }
    }
    LlamaSamplerW(const LlamaSamplerW&) = delete;
    LlamaSamplerW& operator=(const LlamaSamplerW&) = delete;
};

// =============================================================================
// GGUFContext
// =============================================================================

struct GGUFContextW {
    gguf_context* ptr = nullptr;
    bool owner = false;
    GGUFContextW() = default;
    ~GGUFContextW() {
        if (ptr && owner) { gguf_free(ptr); ptr = nullptr; }
    }
    GGUFContextW(const GGUFContextW&) = delete;
    GGUFContextW& operator=(const GGUFContextW&) = delete;
};

// =============================================================================
// Ggml* wrappers
// =============================================================================

struct GgmlBackendDeviceW {
    ggml_backend_dev_t ptr = nullptr;
    // Devices are owned by the registry; never freed by us.
};

struct GgmlBackendW {
    ggml_backend_t ptr = nullptr;
    bool owner = false;
    ~GgmlBackendW() {
        if (ptr && owner) { ggml_backend_free(ptr); ptr = nullptr; }
    }
    GgmlBackendW() = default;
    GgmlBackendW(const GgmlBackendW&) = delete;
    GgmlBackendW& operator=(const GgmlBackendW&) = delete;
};

struct GgmlTensorW {
    ggml_tensor* ptr = nullptr;
    // Non-owning view; tensor lifetime is owned by its parent ggml_context.
};

struct GgmlThreadPoolParamsW {
    ggml_threadpool_params p{};
    explicit GgmlThreadPoolParamsW(int n_threads)
        : p(ggml_threadpool_params_default(n_threads)) {}
    GgmlThreadPoolParamsW() = default;
};

struct GgmlThreadPoolW {
    ggml_threadpool* ptr = nullptr;
    bool owner = false;
    GgmlThreadPoolW() = default;
    explicit GgmlThreadPoolW(GgmlThreadPoolParamsW& params) {
        ptr = ggml_threadpool_new(&params.p);
        if (!ptr) throw std::bad_alloc();
        owner = true;
    }
    ~GgmlThreadPoolW() {
        if (ptr && owner) { ggml_threadpool_free(ptr); ptr = nullptr; }
    }
    GgmlThreadPoolW(const GgmlThreadPoolW&) = delete;
    GgmlThreadPoolW& operator=(const GgmlThreadPoolW&) = delete;
};

// =============================================================================
// Cross-TU pointer unwrap (declared in _llama_native.hpp)
// =============================================================================

namespace inferna {
    llama_model* unwrap_model(nb::handle h) {
        LlamaModelW* m = nb::cast<LlamaModelW*>(h);
        if (!m || !m->ptr)
            throw std::invalid_argument("LlamaModel is null or freed");
        return m->ptr;
    }
    llama_context* unwrap_ctx(nb::handle h) {
        LlamaContextW* c = nb::cast<LlamaContextW*>(h);
        if (!c || !c->ptr)
            throw std::invalid_argument("LlamaContext is null or freed");
        return c->ptr;
    }
}

// =============================================================================
// Sub-module registrars (defined in companion TUs)
// =============================================================================

void register_tts(nb::module_& m);
void register_mtmd(nb::module_& m);
void register_enums(nb::module_& m);

// =============================================================================
// NB_MODULE
// =============================================================================

NB_MODULE(_llama_native, m) {
    m.doc() = "nanobind native bindings for llama.cpp (inferna).";

    // -------------------------------------------------------------------------
    // LlamaModelParams
    // -------------------------------------------------------------------------
    nb::class_<LlamaModelParamsW>(m, "LlamaModelParams")
        .def(nb::init<>())
        PARAM_VAL(LlamaModelParamsW, int,  n_gpu_layers, "n_gpu_layers")
        PARAM_VAL(LlamaModelParamsW, int,  split_mode,   "split_mode")
        PARAM_VAL(LlamaModelParamsW, int,  main_gpu,     "main_gpu")
        PARAM_VAL(LlamaModelParamsW, bool, vocab_only,    "vocab_only")
        PARAM_VAL(LlamaModelParamsW, bool, use_mmap,      "use_mmap")
        PARAM_VAL(LlamaModelParamsW, bool, use_direct_io, "use_direct_io")
        PARAM_VAL(LlamaModelParamsW, bool, use_mlock,     "use_mlock")
        PARAM_VAL(LlamaModelParamsW, bool, check_tensors, "check_tensors")
        PARAM_VAL(LlamaModelParamsW, bool, use_extra_bufts, "use_extra_bufts")
        PARAM_VAL(LlamaModelParamsW, bool, no_host,       "no_host")
        PARAM_VAL(LlamaModelParamsW, bool, no_alloc,      "no_alloc")
        // tensor_split: list[float]; we own a vector backing the C pointer.
        .def_prop_rw("tensor_split",
            [](LlamaModelParamsW& s) -> std::vector<float> {
                std::vector<float> out;
                if (!s.p.tensor_split) return out;
                size_t n = llama_max_devices();
                out.assign(s.p.tensor_split, s.p.tensor_split + n);
                return out;
            },
            [](LlamaModelParamsW& s, std::optional<std::vector<float>> values) {
                size_t max_devices = llama_max_devices();
                if (!values || values->empty()) {
                    s.tensor_split_owned.clear();
                    s.p.tensor_split = nullptr;
                    return;
                }
                if (values->size() > max_devices) {
                    throw std::invalid_argument(
                        "tensor_split has " + std::to_string(values->size()) +
                        " elements but max devices is " + std::to_string(max_devices));
                }
                s.tensor_split_owned.assign(max_devices, 0.0f);
                for (size_t i = 0; i < values->size(); ++i)
                    s.tensor_split_owned[i] = (*values)[i];
                s.p.tensor_split = s.tensor_split_owned.data();
            })
        .def_prop_rw("progress_callback",
            [](LlamaModelParamsW& s) -> nb::object {
                if (!s.progress_callback_obj || s.progress_callback_obj.is_none())
                    return nb::none();
                return s.progress_callback_obj;
            },
            [](LlamaModelParamsW& s, nb::object cb) {
                if (cb.is_none()) {
                    s.progress_callback_obj = nb::object();
                    s.p.progress_callback = nullptr;
                    s.p.progress_callback_user_data = nullptr;
                } else {
                    s.progress_callback_obj = cb;
                    s.p.progress_callback = _llama_progress_cb;
                    // We pass the Python-object pointer directly — nb::object
                    // above keeps the refcount alive for the params lifetime.
                    s.p.progress_callback_user_data = cb.ptr();
                }
            }, nb::arg("cb").none());

    // -------------------------------------------------------------------------
    // LlamaContextParams
    // -------------------------------------------------------------------------
    nb::class_<LlamaContextParamsW>(m, "LlamaContextParams")
        .def(nb::init<>())
        // n_ctx setter rejects negative values up-front (they would silently
        // become a huge unsigned via the C cast otherwise).
        .def_prop_rw("n_ctx",
            [](LlamaContextParamsW& s){ return s.p.n_ctx; },
            [](LlamaContextParamsW& s, int64_t v){
                if (v < 0)
                    throw std::invalid_argument(
                        "n_ctx must be >= 0 (use 0 to inherit the model's training "
                        "context length), got " + std::to_string(v));
                if (v > (int64_t) UINT32_MAX) {
                    PyErr_SetString(PyExc_OverflowError,
                        "n_ctx exceeds uint32_t maximum");
                    throw nb::python_error();
                }
                s.p.n_ctx = (uint32_t) v;
            })
        PARAM_VAL(LlamaContextParamsW, uint32_t, n_batch,         "n_batch")
        PARAM_VAL(LlamaContextParamsW, uint32_t, n_ubatch,        "n_ubatch")
        PARAM_VAL(LlamaContextParamsW, uint32_t, n_seq_max,       "n_seq_max")
        PARAM_VAL(LlamaContextParamsW, uint32_t, n_threads,       "n_threads")
        PARAM_VAL(LlamaContextParamsW, uint32_t, n_threads_batch, "n_threads_batch")
        PARAM_VAL(LlamaContextParamsW, int,  rope_scaling_type, "rope_scaling_type")
        PARAM_VAL(LlamaContextParamsW, int,  pooling_type,      "pooling_type")
        PARAM_VAL(LlamaContextParamsW, int,  attention_type,    "attention_type")
        PARAM_VAL(LlamaContextParamsW, int,  flash_attn_type,   "flash_attn_type")
        PARAM_VAL(LlamaContextParamsW, float, rope_freq_base,   "rope_freq_base")
        PARAM_VAL(LlamaContextParamsW, float, rope_freq_scale,  "rope_freq_scale")
        PARAM_VAL(LlamaContextParamsW, float, yarn_ext_factor,  "yarn_ext_factor")
        PARAM_VAL(LlamaContextParamsW, float, yarn_attn_factor, "yarn_attn_factor")
        PARAM_VAL(LlamaContextParamsW, float, yarn_beta_fast,   "yarn_beta_fast")
        PARAM_VAL(LlamaContextParamsW, float, yarn_beta_slow,   "yarn_beta_slow")
        PARAM_VAL(LlamaContextParamsW, uint32_t, yarn_orig_ctx, "yarn_orig_ctx")
        PARAM_VAL(LlamaContextParamsW, int,  type_k,            "type_k")
        PARAM_VAL(LlamaContextParamsW, int,  type_v,            "type_v")
        PARAM_VAL(LlamaContextParamsW, bool, embeddings,        "embeddings")
        PARAM_VAL(LlamaContextParamsW, bool, offload_kqv,       "offload_kqv")
        PARAM_VAL(LlamaContextParamsW, bool, no_perf,           "no_perf")
        PARAM_VAL(LlamaContextParamsW, bool, op_offload,        "op_offload")
        PARAM_VAL(LlamaContextParamsW, bool, swa_full,          "swa_full")
        PARAM_VAL(LlamaContextParamsW, bool, kv_unified,        "kv_unified");

    // -------------------------------------------------------------------------
    // LlamaModelQuantizeParams
    // -------------------------------------------------------------------------
    nb::class_<LlamaModelQuantizeParamsW>(m, "LlamaModelQuantizeParams")
        .def(nb::init<>())
        PARAM_VAL(LlamaModelQuantizeParamsW, int,  nthread, "nthread")
        PARAM_VAL(LlamaModelQuantizeParamsW, int,  ftype,   "ftype")
        PARAM_VAL(LlamaModelQuantizeParamsW, int,  output_tensor_type,    "output_tensor_type")
        PARAM_VAL(LlamaModelQuantizeParamsW, int,  token_embedding_type,  "token_embedding_type")
        PARAM_VAL(LlamaModelQuantizeParamsW, bool, allow_requantize,        "allow_requantize")
        PARAM_VAL(LlamaModelQuantizeParamsW, bool, quantize_output_tensor,  "quantize_output_tensor")
        PARAM_VAL(LlamaModelQuantizeParamsW, bool, only_copy, "only_copy")
        PARAM_VAL(LlamaModelQuantizeParamsW, bool, pure,      "pure")
        PARAM_VAL(LlamaModelQuantizeParamsW, bool, keep_split, "keep_split")
        PARAM_VAL(LlamaModelQuantizeParamsW, bool, dry_run,    "dry_run");

    // -------------------------------------------------------------------------
    // LlamaSamplerChainParams
    // -------------------------------------------------------------------------
    nb::class_<LlamaSamplerChainParamsW>(m, "LlamaSamplerChainParams")
        .def(nb::init<>())
        PARAM_VAL(LlamaSamplerChainParamsW, bool, no_perf, "no_perf");

    // -------------------------------------------------------------------------
    // LlamaChatMessage
    // -------------------------------------------------------------------------
    nb::class_<LlamaChatMessageW>(m, "LlamaChatMessage")
        .def(nb::init<const std::string&, const std::string&>(),
             "role"_a, "content"_a)
        .def_prop_ro("role",    [](LlamaChatMessageW& s){ return s.role_s; })
        .def_prop_ro("content", [](LlamaChatMessageW& s){ return s.content_s; });

    // -------------------------------------------------------------------------
    // LlamaTokenData
    // -------------------------------------------------------------------------
    nb::class_<LlamaTokenDataW>(m, "LlamaTokenData")
        .def(nb::init<>())
        PARAM_VAL(LlamaTokenDataW, int,   id,    "id")
        PARAM_VAL(LlamaTokenDataW, float, logit, "logit")
        PARAM_VAL(LlamaTokenDataW, float, p,     "p");

    // -------------------------------------------------------------------------
    // LlamaLogitBias
    // -------------------------------------------------------------------------
    nb::class_<LlamaLogitBiasW>(m, "LlamaLogitBias")
        .def(nb::init<>())
        PARAM_VAL(LlamaLogitBiasW, int,   token, "token")
        PARAM_VAL(LlamaLogitBiasW, float, bias,  "bias");

    // -------------------------------------------------------------------------
    // LlamaModelKvOverride — fields exposed as a flat tag+value surface.
    // -------------------------------------------------------------------------
    nb::class_<LlamaModelKvOverrideW>(m, "LlamaModelKvOverride")
        .def(nb::init<>())
        PARAM_VAL(LlamaModelKvOverrideW, int,    tag,      "tag")
        PARAM_VAL(LlamaModelKvOverrideW, int64_t, val_i64, "val_i64")
        PARAM_VAL(LlamaModelKvOverrideW, double, val_f64,  "val_f64")
        PARAM_VAL(LlamaModelKvOverrideW, bool,   val_bool, "val_bool")
        .def_prop_rw("key",
            [](LlamaModelKvOverrideW& s){ return std::string(s.p.key); },
            [](LlamaModelKvOverrideW& s, const std::string& v){
                std::strncpy(s.p.key, v.c_str(), sizeof(s.p.key) - 1);
                s.p.key[sizeof(s.p.key) - 1] = '\0';
            })
        .def_prop_rw("val_str",
            [](LlamaModelKvOverrideW& s){ return std::string(s.p.val_str); },
            [](LlamaModelKvOverrideW& s, const std::string& v){
                std::strncpy(s.p.val_str, v.c_str(), sizeof(s.p.val_str) - 1);
                s.p.val_str[sizeof(s.p.val_str) - 1] = '\0';
            });

    // -------------------------------------------------------------------------
    // LlamaModelTensorBuftOverride
    // -------------------------------------------------------------------------
    nb::class_<LlamaModelTensorBuftOverrideW>(m, "LlamaModelTensorBuftOverride")
        .def(nb::init<>())
        PARAM_PATH(LlamaModelTensorBuftOverrideW, pattern, pattern_s, "pattern");

    // -------------------------------------------------------------------------
    // LlamaVocab — non-owning view; constructor not exposed (use LlamaModel.get_vocab).
    // -------------------------------------------------------------------------
    nb::class_<LlamaVocabW>(m, "LlamaVocab")
        .def_prop_ro("vocab_type", [](LlamaVocabW& s){
            return (int) llama_vocab_type(s.ptr);
        })
        .def_prop_ro("n_vocab", [](LlamaVocabW& s){
            return llama_vocab_n_tokens(s.ptr);
        })
        .def("get_text",  [](LlamaVocabW& s, int t){
            return std::string(llama_vocab_get_text(s.ptr, t));
        })
        .def("get_score", [](LlamaVocabW& s, int t){ return llama_vocab_get_score(s.ptr, t); })
        .def("get_attr",  [](LlamaVocabW& s, int t){ return (int) llama_vocab_get_attr(s.ptr, t); })
        .def("is_eog",     [](LlamaVocabW& s, int t){ return (bool) llama_vocab_is_eog(s.ptr, t); })
        .def("is_control", [](LlamaVocabW& s, int t){ return (bool) llama_vocab_is_control(s.ptr, t); })
        .def("token_bos", [](LlamaVocabW& s){ return llama_vocab_bos(s.ptr); })
        .def("token_eos", [](LlamaVocabW& s){ return llama_vocab_eos(s.ptr); })
        .def("token_eot", [](LlamaVocabW& s){ return llama_vocab_eot(s.ptr); })
        .def("token_sep", [](LlamaVocabW& s){ return llama_vocab_sep(s.ptr); })
        .def("token_nl",  [](LlamaVocabW& s){ return llama_vocab_nl(s.ptr); })
        .def("token_pad", [](LlamaVocabW& s){ return llama_vocab_pad(s.ptr); })
        .def("get_add_bos", [](LlamaVocabW& s){ return (bool) llama_vocab_get_add_bos(s.ptr); })
        .def("get_add_eos", [](LlamaVocabW& s){ return (bool) llama_vocab_get_add_eos(s.ptr); })
        .def("get_add_sep", [](LlamaVocabW& s){ return (bool) llama_vocab_get_add_sep(s.ptr); })
        // Note: the public method names (fim_prefix/middle/suffix) intentionally
        // map onto llama.cpp's fim_pre/suf/mid in a shuffled order. Don't
        // "fix" — this is the contract callers depend on.
        .def("fim_prefix", [](LlamaVocabW& s){ return llama_vocab_fim_pre(s.ptr); })
        .def("fim_middle", [](LlamaVocabW& s){ return llama_vocab_fim_suf(s.ptr); })
        .def("fim_suffix", [](LlamaVocabW& s){ return llama_vocab_fim_mid(s.ptr); })
        .def("fim_pad", [](LlamaVocabW& s){ return llama_vocab_fim_pad(s.ptr); })
        .def("fim_rep", [](LlamaVocabW& s){ return llama_vocab_fim_rep(s.ptr); })
        .def("fim_sep", [](LlamaVocabW& s){ return llama_vocab_fim_sep(s.ptr); })
        .def("tokenize", [](LlamaVocabW& s, const std::string& text,
                              bool add_special, bool parse_special) {
            // llama_tokenize returns the negative required size when the
            // buffer is too small; honor that contract by retrying once
            // with the exact capacity it asked for.
            int cap = (int) text.size() + 16;
            std::vector<llama_token> tokens(cap);
            int n = llama_tokenize(s.ptr, text.c_str(), (int)text.size(),
                                    tokens.data(), cap, add_special, parse_special);
            if (n < 0) {
                cap = -n;
                tokens.resize(cap);
                n = llama_tokenize(s.ptr, text.c_str(), (int)text.size(),
                                    tokens.data(), cap, add_special, parse_special);
                if (n < 0) {
                    throw std::runtime_error(
                        "Failed to tokenize after retry: rc=" + std::to_string(n));
                }
            }
            return std::vector<int>(tokens.begin(), tokens.begin() + n);
        }, "text"_a, "add_special"_a, "parse_special"_a)
        .def("token_to_piece", [](LlamaVocabW& s, int token, int lstrip, bool special) {
            char buf[128];
            int len = llama_token_to_piece(s.ptr, token, buf, sizeof(buf), lstrip, special);
            if (len < 0) throw std::invalid_argument(
                "Failed to convert token " + std::to_string(token) + " to piece");
            // errors='replace' would be ideal — std::string accepts arbitrary bytes,
            // but Python decode follows. We hand back raw bytes here and let
            // callers decode if needed; tests pass plain ASCII or do their own decode.
            return std::string(buf, len);
        }, "token"_a, "lstrip"_a = 0, "special"_a = false)
        .def("detokenize", [](LlamaVocabW& s, const std::vector<int>& tokens,
                                int text_len_max, bool remove_special, bool unparse_special) {
            std::vector<llama_token> vec(tokens.begin(), tokens.end());
            std::vector<char> buf(text_len_max);
            int rc = llama_detokenize(s.ptr, vec.data(), (int) vec.size(),
                                       buf.data(), text_len_max,
                                       remove_special, unparse_special);
            if (rc < 0) throw std::runtime_error(
                "Failed to detokenize: text=\"" + std::to_string(rc) +
                "\" n_tokens=" + std::to_string(vec.size()));
            std::string out(buf.data(), rc);
            // Strip leading whitespace from the detokenized output.
            size_t start = 0;
            while (start < out.size() && (unsigned char)out[start] <= ' ') start++;
            return out.substr(start);
        }, "tokens"_a, "text_len_max"_a = 1024,
           "remove_special"_a = false, "unparse_special"_a = false);

    // -------------------------------------------------------------------------
    // LlamaAdapterLora — meta_* lookups; constructed via LlamaModel.lora_adapter_init
    // -------------------------------------------------------------------------
    nb::class_<LlamaAdapterLoraW>(m, "LlamaAdapterLora")
        .def("meta_val_str", [](LlamaAdapterLoraW& s, const std::string& key){
            if (key.empty()) throw std::invalid_argument("key must not be an empty string");
            std::vector<char> buf(512);
            int rc = llama_adapter_meta_val_str(s.ptr, key.c_str(), buf.data(), (int) buf.size());
            if (rc == -1) throw std::invalid_argument("failed to retrieve metadata value");
            if (rc >= (int) buf.size()) {
                buf.resize(rc + 1);
                rc = llama_adapter_meta_val_str(s.ptr, key.c_str(), buf.data(), (int) buf.size());
                if (rc == -1) throw std::invalid_argument("failed to retrieve metadata value");
            }
            return std::string(buf.data(), rc);
        })
        .def("meta_count", [](LlamaAdapterLoraW& s){
            return llama_adapter_meta_count(s.ptr);
        })
        .def("meta_key_by_index", [](LlamaAdapterLoraW& s, int idx){
            std::vector<char> buf(512);
            int rc = llama_adapter_meta_key_by_index(s.ptr, idx, buf.data(), (int) buf.size());
            if (rc == -1) throw std::invalid_argument("failed to retrieve metadata key");
            if (rc >= (int) buf.size()) {
                buf.resize(rc + 1);
                rc = llama_adapter_meta_key_by_index(s.ptr, idx, buf.data(), (int) buf.size());
                if (rc == -1) throw std::invalid_argument("failed to retrieve metadata key");
            }
            return std::string(buf.data(), rc);
        }, "idx"_a = 0)
        .def("meta_val_str_by_index", [](LlamaAdapterLoraW& s, int idx){
            std::vector<char> buf(512);
            int rc = llama_adapter_meta_val_str_by_index(s.ptr, idx, buf.data(), (int) buf.size());
            if (rc == -1) throw std::invalid_argument("failed to retrieve metadata value");
            if (rc >= (int) buf.size()) {
                buf.resize(rc + 1);
                rc = llama_adapter_meta_val_str_by_index(s.ptr, idx, buf.data(), (int) buf.size());
                if (rc == -1) throw std::invalid_argument("failed to retrieve metadata value");
            }
            return std::string(buf.data(), rc);
        }, "idx"_a = 0);

    // -------------------------------------------------------------------------
    // LlamaModel
    // -------------------------------------------------------------------------
    nb::class_<LlamaModelW>(m, "LlamaModel")
        .def("__init__",
             [](LlamaModelW* self, const std::string& path,
                std::optional<LlamaModelParamsW*> params, bool verbose) {
                 new (self) LlamaModelW(path, params, verbose);
             },
             "path_model"_a, "params"_a = nb::none(), "verbose"_a = true)
        .def_ro("path_model", &LlamaModelW::path_model)
        .def_ro("verbose", &LlamaModelW::verbose)
        .def_prop_ro("rope_type", [](LlamaModelW& s){
            return (int) llama_model_rope_type(s.ptr);
        })
        .def_prop_ro("n_ctx_train", [](LlamaModelW& s){ return s.cached_n_ctx_train; })
        .def_prop_ro("n_embd",      [](LlamaModelW& s){ return s.cached_n_embd; })
        .def_prop_ro("n_embd_inp",  [](LlamaModelW& s){ return s.cached_n_embd_inp; })
        .def_prop_ro("n_layer",     [](LlamaModelW& s){ return s.cached_n_layer; })
        .def_prop_ro("n_head",      [](LlamaModelW& s){ return s.cached_n_head; })
        .def_prop_ro("n_head_kv",   [](LlamaModelW& s){ return s.cached_n_head_kv; })
        .def_prop_ro("n_params",    [](LlamaModelW& s){ return s.cached_n_params; })
        .def_prop_ro("size",        [](LlamaModelW& s){ return s.cached_size; })
        .def_prop_ro("rope_freq_scale_train", [](LlamaModelW& s){
            return llama_model_rope_freq_scale_train(s.ptr);
        })
        .def_prop_ro("desc", [](LlamaModelW& s){
            char buf[1024];
            llama_model_desc(s.ptr, buf, sizeof(buf));
            return std::string(buf);
        })
        // Convenience: tests/code commonly access n_vocab via the model.
        .def_prop_ro("n_vocab", [](LlamaModelW& s){
            return llama_vocab_n_tokens(llama_model_get_vocab(s.ptr));
        })
        .def("get_vocab", [](nb::object self_obj){
            LlamaModelW& s = nb::cast<LlamaModelW&>(self_obj);
            auto* v = new LlamaVocabW{};
            v->ptr = llama_model_get_vocab(s.ptr);
            v->parent = self_obj;  // keep model alive while vocab view exists
            return nb::cast(v, nb::rv_policy::take_ownership);
        })
        .def("lora_adapter_init", [](nb::object self_obj, const std::string& path_lora){
            LlamaModelW& s = nb::cast<LlamaModelW&>(self_obj);
            nb::module_ os = nb::module_::import_("os");
            if (!nb::cast<bool>(os.attr("path").attr("exists")(path_lora))) {
                std::string msg = "LoRA adapter file not found: " + path_lora;
                PyErr_SetString(PyExc_FileNotFoundError, msg.c_str());
                throw nb::python_error();
            }
            llama_adapter_lora* a = llama_adapter_lora_init(s.ptr, path_lora.c_str());
            if (!a) throw std::invalid_argument(
                "Failed to load LoRA adapter from: " + path_lora);
            auto* w = new LlamaAdapterLoraW{};
            w->ptr = a;
            // The model owns the adapter (frees on dtor); don't double-free.
            w->owner = false;
            return nb::cast(w, nb::rv_policy::take_ownership);
        })
        .def("meta_val_str", [](LlamaModelW& s, const std::string& key){
            std::vector<char> buf(512);
            int rc = llama_model_meta_val_str(s.ptr, key.c_str(), buf.data(), (int) buf.size());
            if (rc == -1) throw std::invalid_argument(
                "could not get metadata value from " + key);
            if (rc >= (int) buf.size()) {
                buf.resize(rc + 1);
                rc = llama_model_meta_val_str(s.ptr, key.c_str(), buf.data(), (int) buf.size());
                if (rc == -1) throw std::invalid_argument(
                    "could not get metadata value from " + key);
            }
            return std::string(buf.data(), rc);
        })
        .def("meta_count", [](LlamaModelW& s){ return llama_model_meta_count(s.ptr); })
        .def("meta_key_by_index", [](LlamaModelW& s, int index){
            std::vector<char> buf(512);
            int rc = llama_model_meta_key_by_index(s.ptr, index, buf.data(), (int) buf.size());
            if (rc == -1) throw std::invalid_argument(
                "could not get metadata key at index " + std::to_string(index));
            if (rc >= (int) buf.size()) {
                buf.resize(rc + 1);
                rc = llama_model_meta_key_by_index(s.ptr, index, buf.data(), (int) buf.size());
                if (rc == -1) throw std::invalid_argument(
                    "could not get metadata key at index " + std::to_string(index));
            }
            return std::string(buf.data(), rc);
        })
        .def("meta_val_str_by_index", [](LlamaModelW& s, int index){
            std::vector<char> buf(512);
            int rc = llama_model_meta_val_str_by_index(s.ptr, index, buf.data(), (int) buf.size());
            if (rc == -1) throw std::invalid_argument(
                "could not get metadata value at index " + std::to_string(index));
            if (rc >= (int) buf.size()) {
                buf.resize(rc + 1);
                rc = llama_model_meta_val_str_by_index(s.ptr, index, buf.data(), (int) buf.size());
                if (rc == -1) throw std::invalid_argument(
                    "could not get metadata value at index " + std::to_string(index));
            }
            return std::string(buf.data(), rc);
        })
        .def("has_encoder", [](LlamaModelW& s){ return (bool) llama_model_has_encoder(s.ptr); })
        .def("has_decoder", [](LlamaModelW& s){ return (bool) llama_model_has_decoder(s.ptr); })
        .def("decoder_start_token", [](LlamaModelW& s){
            return llama_model_decoder_start_token(s.ptr);
        })
        .def("is_recurrent", [](LlamaModelW& s){ return (bool) llama_model_is_recurrent(s.ptr); })
        .def("is_hybrid",    [](LlamaModelW& s){ return (bool) llama_model_is_hybrid(s.ptr); })
        .def("get_default_chat_template", [](LlamaModelW& s){
            const char* r = llama_model_chat_template(s.ptr, nullptr);
            return r ? std::string(r) : std::string();
        })
        .def("get_default_chat_template_by_name", [](LlamaModelW& s, const std::string& name){
            const char* r = llama_model_chat_template(s.ptr, name.c_str());
            return r ? std::string(r) : std::string();
        })
        .def("chat_apply_template", [](LlamaModelW& s, std::optional<std::string> tmpl,
                                        nb::list msgs, bool add_assistant_msg) {
            std::vector<llama_chat_message> vec;
            vec.reserve(msgs.size());
            for (auto h : msgs) {
                LlamaChatMessageW& m = nb::cast<LlamaChatMessageW&>(h);
                vec.push_back(m.p);
            }
            const char* tmpl_ptr = tmpl ? tmpl->c_str() : nullptr;
            int required = llama_chat_apply_template(
                tmpl_ptr, vec.data(), vec.size(), add_assistant_msg, nullptr, 0);
            if (required < 0) throw std::runtime_error("Failed to apply chat template");
            std::vector<char> buf(required + 1);
            int actual = llama_chat_apply_template(
                tmpl_ptr, vec.data(), vec.size(), add_assistant_msg,
                buf.data(), required);
            if (actual < 0) throw std::runtime_error("Failed to apply chat template");
            return std::string(buf.data(), actual);
        }, "tmpl"_a.none(), "msgs"_a, "add_assistant_msg"_a)
        .def("metadata", [](LlamaModelW& s){
            nb::dict out;
            int n = llama_model_meta_count(s.ptr);
            std::vector<char> buf(1024);
            for (int i = 0; i < n; ++i) {
                int sz = llama_model_meta_key_by_index(s.ptr, i, buf.data(), (int) buf.size());
                if (sz > (int) buf.size()) {
                    buf.resize(sz + 1);
                    llama_model_meta_key_by_index(s.ptr, i, buf.data(), (int) buf.size());
                }
                std::string key(buf.data());
                sz = llama_model_meta_val_str_by_index(s.ptr, i, buf.data(), (int) buf.size());
                if (sz > (int) buf.size()) {
                    buf.resize(sz + 1);
                    llama_model_meta_val_str_by_index(s.ptr, i, buf.data(), (int) buf.size());
                }
                out[nb::cast(key)] = nb::cast(std::string(buf.data()));
            }
            return out;
        })
        .def_static("default_params", [](){ return new LlamaModelParamsW(); },
                    nb::rv_policy::take_ownership);

    // -------------------------------------------------------------------------
    // LlamaBatch
    // -------------------------------------------------------------------------
    nb::class_<LlamaBatchW>(m, "LlamaBatch")
        .def("__init__",
             [](LlamaBatchW* self, int n_tokens, int embd, int n_seq_max, bool /*verbose*/) {
                 new (self) LlamaBatchW(n_tokens, embd, n_seq_max);
             },
             nb::kw_only(),
             "n_tokens"_a, "embd"_a, "n_seq_max"_a, "verbose"_a = true)
        .def_prop_ro("n_tokens", [](LlamaBatchW& s){ return s.p.n_tokens; })
        // Expose the construction-time capacity under both names. `_n_tokens`
        // is the canonical key the BatchMemoryPool keys on.
        .def_ro("_n_tokens", &LlamaBatchW::n_tokens_capacity)
        .def_ro("n_tokens_capacity", &LlamaBatchW::n_tokens_capacity)
        .def_ro("embd", &LlamaBatchW::embd)
        .def_ro("n_seq_max", &LlamaBatchW::n_seq_max)
        .def("close", [](LlamaBatchW& s){
            if (s.owner && s.p.token) {
                llama_batch_free(s.p);
                s.owner = false;
                s.p = llama_batch{};
            }
        })
        .def("reset", [](LlamaBatchW& s){ s.p.n_tokens = 0; })
        .def("clear", [](LlamaBatchW& s){ s.p.n_tokens = 0; })
        .def("add", [](LlamaBatchW& s, int id, int pos, std::vector<int> seq_ids, bool logits){
            int n = s.p.n_tokens;
            if (n >= s.n_tokens_capacity)
                throw std::out_of_range("Batch is full (capacity=" + std::to_string(s.n_tokens_capacity) + ")");
            if ((int) seq_ids.size() > s.n_seq_max)
                throw std::invalid_argument(
                    "seq_ids length (" + std::to_string(seq_ids.size()) +
                    ") exceeds n_seq_max (" + std::to_string(s.n_seq_max) + ")");
            s.p.token[n] = id;
            s.p.pos[n]   = pos;
            s.p.n_seq_id[n] = (int32_t) seq_ids.size();
            for (size_t i = 0; i < seq_ids.size(); ++i)
                s.p.seq_id[n][i] = (llama_seq_id) seq_ids[i];
            s.p.logits[n] = logits;
            s.p.n_tokens += 1;
        }, "id"_a, "pos"_a, "seq_ids"_a, "logits"_a)
        .def("set_batch", [](LlamaBatchW& s, std::vector<int> batch, int n_past, bool logits_all){
            int n = (int) batch.size();
            if (n > s.n_tokens_capacity)
                throw std::out_of_range(
                    "batch length (" + std::to_string(n) +
                    ") exceeds capacity (" + std::to_string(s.n_tokens_capacity) + ")");
            if (s.n_seq_max < 1)
                throw std::invalid_argument("set_batch requires n_seq_max >= 1");
            s.p.n_tokens = n;
            for (int i = 0; i < n; ++i) {
                s.p.pos[i] = n_past + i;
                s.p.seq_id[i][0] = 0;
                s.p.n_seq_id[i] = 1;
                s.p.logits[i] = logits_all;
                s.p.token[i] = batch[i];
            }
            if (n > 0) s.p.logits[n - 1] = true;
        })
        .def("add_sequence", [](LlamaBatchW& s, std::vector<int> batch, int seq_id, bool logits_all){
            int n = (int) batch.size();
            int n0 = s.p.n_tokens;
            if (n0 + n > s.n_tokens_capacity)
                throw std::out_of_range(
                    "add_sequence would exceed capacity (n_tokens=" + std::to_string(n0) +
                    " + " + std::to_string(n) + " > " + std::to_string(s.n_tokens_capacity) + ")");
            if (s.n_seq_max < 1)
                throw std::invalid_argument("add_sequence requires n_seq_max >= 1");
            s.p.n_tokens += n;
            for (int i = 0; i < n; ++i) {
                int j = n0 + i;
                s.p.pos[j] = i;
                s.p.seq_id[j][0] = seq_id;
                s.p.n_seq_id[j] = 1;
                s.p.logits[j] = logits_all;
                s.p.token[j] = batch[i];
            }
            if (n > 0) s.p.logits[n0 + n - 1] = true;
        })
        .def("set_last_logits_to_true", [](LlamaBatchW& s){
            if (s.p.n_tokens > 0) s.p.logits[s.p.n_tokens - 1] = true;
        });

    // -------------------------------------------------------------------------
    // LlamaContext
    // -------------------------------------------------------------------------
    nb::class_<LlamaContextW>(m, "LlamaContext")
        .def("__init__",
             [](LlamaContextW* self, nb::object model,
                std::optional<LlamaContextParamsW*> params, bool verbose) {
                 new (self) LlamaContextW(model, params, verbose);
             },
             "model"_a, "params"_a = nb::none(), "verbose"_a = true)
        .def_ro("verbose", &LlamaContextW::verbose)
        .def_rw("n_tokens", &LlamaContextW::n_tokens)
        .def_prop_ro("model", [](LlamaContextW& s){ return s.model_obj; })
        .def("close", [](LlamaContextW& s){
            if (s.ptr && s.owner) { llama_free(s.ptr); s.ptr = nullptr; }
        })
        .def_prop_ro("is_valid", [](LlamaContextW& s){ return s.ptr != nullptr; })
        .def_prop_ro("n_ctx",      [](LlamaContextW& s){ s.ensure_valid(); return llama_n_ctx(s.ptr); })
        .def_prop_ro("n_ctx_seq",  [](LlamaContextW& s){ s.ensure_valid(); return llama_n_ctx_seq(s.ptr); })
        .def_prop_ro("n_batch",    [](LlamaContextW& s){ s.ensure_valid(); return llama_n_batch(s.ptr); })
        .def_prop_ro("n_ubatch",   [](LlamaContextW& s){ s.ensure_valid(); return llama_n_ubatch(s.ptr); })
        .def_prop_ro("n_seq_max",  [](LlamaContextW& s){ s.ensure_valid(); return llama_n_seq_max(s.ptr); })
        .def_prop_ro("pooling_type", [](LlamaContextW& s){
            s.ensure_valid();
            return (int) llama_pooling_type(s.ptr);
        })
        .def("encode", [](LlamaContextW& s, LlamaBatchW& batch){
            s.ensure_valid();
            llama_context* ctx = s.ptr;
            llama_batch b = batch.p;
            int rc;
            {
                nb::gil_scoped_release rel;
                rc = llama_encode(ctx, b);
            }
            if (rc < 0) throw std::runtime_error("error encoding batch");
        })
        .def("decode", [](LlamaContextW& s, LlamaBatchW& batch){
            s.ensure_valid();
            llama_context* ctx = s.ptr;
            llama_batch b = batch.p;
            int rc;
            {
                nb::gil_scoped_release rel;
                rc = llama_decode(ctx, b);
            }
            s.n_tokens = batch.p.n_tokens;
            if (rc == 1) throw std::invalid_argument(
                "could not find a KV slot for the batch (try reducing the size "
                "of the batch or increase the context)");
            if (rc < 0) throw std::runtime_error("llama_decode failed");
            return rc;
        })
        .def("set_n_threads", [](LlamaContextW& s, int n, int nb){
            s.ensure_valid();
            llama_set_n_threads(s.ptr, n, nb);
        })
        .def("n_threads", [](LlamaContextW& s){ s.ensure_valid(); return llama_n_threads(s.ptr); })
        .def("n_threads_batch", [](LlamaContextW& s){ s.ensure_valid(); return llama_n_threads_batch(s.ptr); })
        .def("set_embeddings_mode", [](LlamaContextW& s, bool e){
            s.ensure_valid();
            llama_set_embeddings(s.ptr, e);
        })
        .def("set_causal_attn", [](LlamaContextW& s, bool c){
            s.ensure_valid();
            llama_set_causal_attn(s.ptr, c);
        })
        .def("install_cancel_callback", [](LlamaContextW& s){
            s.ensure_valid();
            llama_set_abort_callback(s.ptr, _cancel_flag_callback, &s.cancel_flag);
        })
        .def_prop_rw("cancel",
            [](LlamaContextW& s){ return s.cancel_flag; },
            [](LlamaContextW& s, bool v){ s.cancel_flag = v; })
        .def("synchronize", [](LlamaContextW& s){ s.ensure_valid(); llama_synchronize(s.ptr); })
        .def("get_state_size", [](LlamaContextW& s){ s.ensure_valid(); return llama_state_get_size(s.ptr); })
        .def("kv_cache_clear", [](LlamaContextW& s, bool clear_data){
            s.ensure_valid();
            llama_memory_t mem = llama_get_memory(s.ptr);
            if (mem) llama_memory_clear(mem, clear_data);
        }, "clear_data"_a = true)
        .def("memory_seq_rm", [](LlamaContextW& s, int seq_id, int p0, int p1){
            s.ensure_valid();
            llama_memory_t mem = llama_get_memory(s.ptr);
            return mem ? (bool) llama_memory_seq_rm(mem, seq_id, p0, p1) : false;
        })
        .def("memory_seq_cp", [](LlamaContextW& s, int src, int dst, int p0, int p1){
            s.ensure_valid();
            llama_memory_t mem = llama_get_memory(s.ptr);
            if (mem) llama_memory_seq_cp(mem, src, dst, p0, p1);
        })
        .def("memory_seq_keep", [](LlamaContextW& s, int seq_id){
            s.ensure_valid();
            llama_memory_t mem = llama_get_memory(s.ptr);
            if (mem) llama_memory_seq_keep(mem, seq_id);
        })
        .def("memory_seq_add", [](LlamaContextW& s, int seq_id, int p0, int p1, int delta){
            s.ensure_valid();
            llama_memory_t mem = llama_get_memory(s.ptr);
            if (mem) llama_memory_seq_add(mem, seq_id, p0, p1, delta);
        })
        .def("memory_seq_pos_min", [](LlamaContextW& s, int seq_id){
            s.ensure_valid();
            llama_memory_t mem = llama_get_memory(s.ptr);
            return mem ? llama_memory_seq_pos_min(mem, seq_id) : -1;
        })
        .def("memory_seq_pos_max", [](LlamaContextW& s, int seq_id){
            s.ensure_valid();
            llama_memory_t mem = llama_get_memory(s.ptr);
            return mem ? llama_memory_seq_pos_max(mem, seq_id) : -1;
        })
        .def("get_logits", [](LlamaContextW& s){
            s.ensure_valid();
            LlamaModelW& model = nb::cast<LlamaModelW&>(s.model_obj);
            int n = llama_vocab_n_tokens(llama_model_get_vocab(model.ptr));
            float* logits = llama_get_logits(s.ptr);
            if (!logits) throw std::invalid_argument("no logits available");
            return std::vector<float>(logits, logits + n);
        })
        .def("get_logits_ith", [](LlamaContextW& s, int i){
            s.ensure_valid();
            LlamaModelW& model = nb::cast<LlamaModelW&>(s.model_obj);
            int n = llama_vocab_n_tokens(llama_model_get_vocab(model.ptr));
            float* logits = llama_get_logits_ith(s.ptr, i);
            if (!logits) throw std::invalid_argument(std::to_string(i) + " is an invalid id");
            return std::vector<float>(logits, logits + n);
        })
        .def("get_embeddings", [](LlamaContextW& s){
            s.ensure_valid();
            LlamaModelW& model = nb::cast<LlamaModelW&>(s.model_obj);
            int n_embd = model.cached_n_embd;
            float* e = llama_get_embeddings(s.ptr);
            if (!e) throw std::invalid_argument("no embeddings available");
            return std::vector<float>(e, e + n_embd);
        })
        .def("get_embeddings_ith", [](LlamaContextW& s, int i){
            s.ensure_valid();
            LlamaModelW& model = nb::cast<LlamaModelW&>(s.model_obj);
            int n_embd = model.cached_n_embd;
            float* e = llama_get_embeddings_ith(s.ptr, i);
            if (!e) throw std::invalid_argument(std::to_string(i) + " is an invalid id");
            return std::vector<float>(e, e + n_embd);
        })
        .def("get_perf_data", [](LlamaContextW& s){
            s.ensure_valid();
            llama_perf_context_data d = llama_perf_context(s.ptr);
            nb::dict out;
            out["t_start_ms"]  = d.t_start_ms;
            out["t_load_ms"]   = d.t_load_ms;
            out["t_p_eval_ms"] = d.t_p_eval_ms;
            out["t_eval_ms"]   = d.t_eval_ms;
            out["n_p_eval"]    = d.n_p_eval;
            out["n_eval"]      = d.n_eval;
            out["n_reused"]    = d.n_reused;
            return out;
        })
        .def("print_perf_data", [](LlamaContextW& s){ s.ensure_valid(); llama_perf_context_print(s.ptr); })
        .def("reset_perf_data", [](LlamaContextW& s){ s.ensure_valid(); llama_perf_context_reset(s.ptr); });

    // -------------------------------------------------------------------------
    // LlamaSampler
    // -------------------------------------------------------------------------
    nb::class_<LlamaSamplerW>(m, "LlamaSampler")
        .def("__init__",
             [](LlamaSamplerW* self, std::optional<LlamaSamplerChainParamsW*> params) {
                 new (self) LlamaSamplerW(params);
             },
             "params"_a = nb::none())
        .def("name", [](LlamaSamplerW& s){
            return std::string(llama_sampler_name(s.ptr));
        })
        .def("accept", [](LlamaSamplerW& s, int t){ llama_sampler_accept(s.ptr, t); })
        .def("reset",  [](LlamaSamplerW& s){ llama_sampler_reset(s.ptr); })
        .def("clone",  [](LlamaSamplerW& s){
            auto* w = new LlamaSamplerW{};
            w->ptr = llama_sampler_clone(s.ptr);
            w->owner = true;
            return nb::cast(w, nb::rv_policy::take_ownership);
        })
        .def("get_seed", [](LlamaSamplerW& s){ return llama_sampler_get_seed(s.ptr); })
        .def("add_greedy", [](LlamaSamplerW& s){
            llama_sampler_chain_add(s.ptr, llama_sampler_init_greedy());
        })
        .def("add_dist", [](LlamaSamplerW& s, uint32_t seed){
            llama_sampler_chain_add(s.ptr, llama_sampler_init_dist(seed));
        })
        .def("add_top_k", [](LlamaSamplerW& s, int32_t k){
            llama_sampler_chain_add(s.ptr, llama_sampler_init_top_k(k));
        })
        .def("add_top_p", [](LlamaSamplerW& s, float p, size_t mk){
            llama_sampler_chain_add(s.ptr, llama_sampler_init_top_p(p, mk));
        })
        .def("add_min_p", [](LlamaSamplerW& s, float p, size_t mk){
            llama_sampler_chain_add(s.ptr, llama_sampler_init_min_p(p, mk));
        })
        .def("add_typical", [](LlamaSamplerW& s, float p, size_t mk){
            llama_sampler_chain_add(s.ptr, llama_sampler_init_typical(p, mk));
        })
        .def("add_temp", [](LlamaSamplerW& s, float t){
            llama_sampler_chain_add(s.ptr, llama_sampler_init_temp(t));
        })
        .def("add_temp_ext", [](LlamaSamplerW& s, float t, float d, float e){
            llama_sampler_chain_add(s.ptr, llama_sampler_init_temp_ext(t, d, e));
        })
        .def("add_xtc", [](LlamaSamplerW& s, float p, float t, size_t mk, uint32_t seed){
            llama_sampler_chain_add(s.ptr, llama_sampler_init_xtc(p, t, mk, seed));
        })
        .def("add_mirostat", [](LlamaSamplerW& s, int n_vocab, uint32_t seed,
                                  float tau, float eta, int m){
            llama_sampler_chain_add(s.ptr, llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m));
        })
        .def("add_mirostat_v2", [](LlamaSamplerW& s, uint32_t seed, float tau, float eta){
            llama_sampler_chain_add(s.ptr, llama_sampler_init_mirostat_v2(seed, tau, eta));
        })
        .def("add_grammar", [](LlamaSamplerW& s, LlamaVocabW& vocab,
                                 const std::string& grammar_str, const std::string& grammar_root){
            llama_sampler_chain_add(s.ptr,
                llama_sampler_init_grammar(vocab.ptr, grammar_str.c_str(), grammar_root.c_str()));
        })
        .def("add_penalties", [](LlamaSamplerW& s, int last_n, float repeat,
                                  float freq, float present){
            llama_sampler_chain_add(s.ptr,
                llama_sampler_init_penalties(last_n, repeat, freq, present));
        })
        .def("add_logit_bias", [](LlamaSamplerW& s, int n_vocab, nb::list biases){
            std::vector<llama_logit_bias> arr;
            arr.reserve(biases.size());
            for (auto h : biases) {
                nb::tuple t = nb::cast<nb::tuple>(h);
                llama_logit_bias b{};
                b.token = nb::cast<int>(t[0]);
                b.bias  = nb::cast<float>(t[1]);
                arr.push_back(b);
            }
            llama_sampler_chain_add(s.ptr,
                llama_sampler_init_logit_bias(n_vocab, (int) arr.size(),
                                                arr.empty() ? nullptr : arr.data()));
        })
        .def("add_infill", [](LlamaSamplerW& s, LlamaVocabW& vocab){
            llama_sampler_chain_add(s.ptr, llama_sampler_init_infill(vocab.ptr));
        })
        .def("sample", [](LlamaSamplerW& s, LlamaContextW& ctx, int idx){
            return llama_sampler_sample(s.ptr, ctx.ptr, idx);
        })
        .def("get_perf_data", [](LlamaSamplerW& s){
            llama_perf_sampler_data d = llama_perf_sampler(s.ptr);
            nb::dict out;
            out["t_sample_ms"] = d.t_sample_ms;
            out["n_sample"]    = d.n_sample;
            return out;
        })
        .def("print_perf_data", [](LlamaSamplerW& s){ llama_perf_sampler_print(s.ptr); })
        .def("reset_perf_data", [](LlamaSamplerW& s){ llama_perf_sampler_reset(s.ptr); });

    // -------------------------------------------------------------------------
    // Module-level functions
    // -------------------------------------------------------------------------
    m.def("disable_logging", [](){
        llama_log_set(_llama_no_log_cb, nullptr);
    });
    m.def("set_log_callback", [](nb::object cb){
        g_log_cb = cb;
        if (cb.is_none()) llama_log_set(nullptr, nullptr);
        else              llama_log_set(_llama_log_cb, nullptr);
    }, "cb"_a.none());

    m.def("chat_builtin_templates", [](){
        int32_t n = llama_chat_builtin_templates(nullptr, 0);
        std::vector<const char*> tmpls(n);
        llama_chat_builtin_templates(tmpls.data(), tmpls.size());
        std::vector<std::string> out;
        out.reserve(n);
        for (const char* t : tmpls) out.emplace_back(t);
        return out;
    });

    m.def("ggml_version", [](){ return std::string(ggml_version()); });
    m.def("ggml_commit",  [](){ return std::string(ggml_commit()); });
    m.def("ggml_time_us", [](){ return ggml_time_us(); });

    m.def("ggml_backend_load_all", [](){
        // Discover and load backend plugins relative to this extension's
        // install location, then any extras reported by the Python-side
        // `inferna._internal.backend_dl` resolver.
        nb::module_ os = nb::module_::import_("os");
        nb::module_ backend_dl = nb::module_::import_("inferna._internal.backend_dl");
        nb::object __file__ = nb::module_::import_("inferna.llama._llama_native").attr("__file__");
        std::string this_file = nb::cast<std::string>(__file__);
        std::string this_dir  = nb::cast<std::string>(os.attr("path").attr("dirname")(
            os.attr("path").attr("abspath")(this_file)));
        ggml_backend_load_all_from_path(this_dir.c_str());
        std::string site = nb::cast<std::string>(os.attr("path").attr("dirname")(
            os.attr("path").attr("dirname")(this_dir)));
        nb::object paths = backend_dl.attr("libs_to_load")(site);
        for (nb::handle p : paths) {
            std::string sp = nb::cast<std::string>(p);
            ggml_backend_load(sp.c_str());
        }
    });

    m.def("ggml_backend_unload", [](const std::string& name){
        size_t n = ggml_backend_reg_count();
        for (size_t i = 0; i < n; ++i) {
            ggml_backend_reg_t reg = ggml_backend_reg_get(i);
            if (name == ggml_backend_reg_name(reg)) {
                ggml_backend_unload(reg);
                return;
            }
        }
        throw std::invalid_argument("backend '" + name + "' not found in registry");
    });

    m.def("ggml_backend_reg_count", [](){ return ggml_backend_reg_count(); });
    m.def("ggml_backend_reg_names", [](){
        size_t n = ggml_backend_reg_count();
        std::vector<std::string> out;
        for (size_t i = 0; i < n; ++i)
            out.emplace_back(ggml_backend_reg_name(ggml_backend_reg_get(i)));
        return out;
    });
    m.def("ggml_backend_dev_count", [](){ return ggml_backend_dev_count(); });
    m.def("ggml_backend_dev_info", [](){
        size_t n = ggml_backend_dev_count();
        nb::list out;
        for (size_t i = 0; i < n; ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            nb::dict info;
            info["name"]        = std::string(ggml_backend_dev_name(d));
            info["description"] = std::string(ggml_backend_dev_description(d));
            // Type name `ggml_backend_dev_type` collides with the same-named
            // function on this version of ggml.h — use auto to disambiguate.
            auto t = ggml_backend_dev_type(d);
            const char* tname = "unknown";
            switch (t) {
                case GGML_BACKEND_DEVICE_TYPE_CPU:   tname = "CPU"; break;
                case GGML_BACKEND_DEVICE_TYPE_GPU:   tname = "GPU"; break;
                case GGML_BACKEND_DEVICE_TYPE_IGPU:  tname = "iGPU"; break;
                case GGML_BACKEND_DEVICE_TYPE_ACCEL: tname = "ACCEL"; break;
                default: break;
            }
            info["type"] = std::string(tname);
            out.append(info);
        }
        return out;
    });

    m.def("llama_backend_init", [](){ llama_backend_init(); });
    m.def("llama_backend_free", [](){ llama_backend_free(); });
    m.def("llama_numa_init", [](int strategy){
        llama_numa_init((ggml_numa_strategy) strategy);
    });
    m.def("llama_time_us", [](){ return llama_time_us(); });
    m.def("llama_max_devices", [](){ return llama_max_devices(); });
    m.def("llama_supports_mmap",        [](){ return (bool) llama_supports_mmap(); });
    m.def("llama_supports_mlock",       [](){ return (bool) llama_supports_mlock(); });
    m.def("llama_supports_gpu_offload", [](){ return (bool) llama_supports_gpu_offload(); });
    m.def("llama_supports_rpc",         [](){ return (bool) llama_supports_rpc(); });

    m.def("llama_batch_get_one", [](std::vector<int> tokens, int n_past){
        // Build a batch with n_tokens tokens, fill pos / seq_id / logits,
        // with only the last token's logits enabled.
        int n = (int) tokens.size();
        auto* w = new LlamaBatchW(n, 0, 1);
        w->p.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            w->p.pos[i]      = n_past + i;
            w->p.seq_id[i][0] = 0;
            w->p.n_seq_id[i] = 1;
            w->p.logits[i]   = false;
            w->p.token[i]    = tokens[i];
        }
        if (n > 0) w->p.logits[n - 1] = true;
        return nb::cast(w, nb::rv_policy::take_ownership);
    }, "tokens"_a, "n_past"_a = 0);

    m.def("llama_attach_threadpool", [](LlamaContextW& ctx,
                                          GgmlThreadPoolW& tp,
                                          GgmlThreadPoolW& tp_batch){
        llama_attach_threadpool(ctx.ptr, tp.ptr, tp_batch.ptr);
    });
    m.def("llama_detach_threadpool", [](LlamaContextW& ctx){
        llama_detach_threadpool(ctx.ptr);
    });

    m.def("llama_flash_attn_type_name", [](int t){
        return std::string(llama_flash_attn_type_name((llama_flash_attn_type) t));
    });

    // -------------------------------------------------------------------------
    // GgmlBackendDevice — non-owning view into the global registry.
    // -------------------------------------------------------------------------
    nb::class_<GgmlBackendDeviceW>(m, "GgmlBackendDevice")
        .def_prop_ro("name", [](GgmlBackendDeviceW& s){
            return std::string(ggml_backend_dev_name(s.ptr));
        })
        .def_prop_ro("description", [](GgmlBackendDeviceW& s){
            return std::string(ggml_backend_dev_description(s.ptr));
        })
        .def_prop_ro("type", [](GgmlBackendDeviceW& s){
            return (int) ggml_backend_dev_type(s.ptr);
        });

    // -------------------------------------------------------------------------
    // GgmlBackend — created from a device + params string.
    // -------------------------------------------------------------------------
    nb::class_<GgmlBackendW>(m, "GgmlBackend")
        .def("__init__",
             [](GgmlBackendW* self, GgmlBackendDeviceW& dev, const std::string& params){
                 new (self) GgmlBackendW();
                 self->ptr = ggml_backend_dev_init(dev.ptr, params.c_str());
                 if (!self->ptr) throw std::runtime_error("ggml_backend_dev_init failed");
                 self->owner = true;
             },
             "dev"_a, "params"_a)
        .def_prop_ro("name", [](GgmlBackendW& s){
            return std::string(ggml_backend_name(s.ptr));
        });

    // -------------------------------------------------------------------------
    // GgmlTensor — non-owning view; constructor not exposed.
    // -------------------------------------------------------------------------
    nb::class_<GgmlTensorW>(m, "GgmlTensor");

    // -------------------------------------------------------------------------
    // GgmlThreadPoolParams
    // -------------------------------------------------------------------------
    nb::class_<GgmlThreadPoolParamsW>(m, "GgmlThreadPoolParams")
        .def(nb::init<int>(), "n_threads"_a)
        .def("match", [](GgmlThreadPoolParamsW& s, GgmlThreadPoolParamsW& other){
            return (bool) ggml_threadpool_params_match(&s.p, &other.p);
        })
        .def_prop_rw("n_threads",
            [](GgmlThreadPoolParamsW& s){ return s.p.n_threads; },
            [](GgmlThreadPoolParamsW& s, int v){ s.p.n_threads = v; })
        .def_prop_rw("prio",
            [](GgmlThreadPoolParamsW& s){ return (int) s.p.prio; },
            [](GgmlThreadPoolParamsW& s, int v){ s.p.prio = (ggml_sched_priority) v; })
        .def_prop_rw("poll",
            [](GgmlThreadPoolParamsW& s){ return s.p.poll; },
            [](GgmlThreadPoolParamsW& s, uint32_t v){ s.p.poll = v; })
        .def_prop_rw("strict_cpu",
            [](GgmlThreadPoolParamsW& s){ return (bool) s.p.strict_cpu; },
            [](GgmlThreadPoolParamsW& s, bool v){ s.p.strict_cpu = v; })
        .def_prop_rw("paused",
            [](GgmlThreadPoolParamsW& s){ return (bool) s.p.paused; },
            [](GgmlThreadPoolParamsW& s, bool v){ s.p.paused = v; })
        .def_prop_rw("cpumask",
            [](GgmlThreadPoolParamsW& s){
                std::vector<bool> out;
                out.reserve(GGML_MAX_N_THREADS);
                for (int i = 0; i < GGML_MAX_N_THREADS; ++i)
                    out.push_back((bool) s.p.cpumask[i]);
                return out;
            },
            [](GgmlThreadPoolParamsW& s, std::vector<bool> values){
                if ((int) values.size() != GGML_MAX_N_THREADS)
                    throw std::invalid_argument(
                        "cpumask must have exactly " + std::to_string(GGML_MAX_N_THREADS) +
                        " elements, got " + std::to_string(values.size()));
                for (int i = 0; i < GGML_MAX_N_THREADS; ++i)
                    s.p.cpumask[i] = values[i];
            });

    // -------------------------------------------------------------------------
    // GgmlThreadPool
    // -------------------------------------------------------------------------
    nb::class_<GgmlThreadPoolW>(m, "GgmlThreadPool")
        .def(nb::init<GgmlThreadPoolParamsW&>(), "params"_a)
        .def("pause",  [](GgmlThreadPoolW& s){ ggml_threadpool_pause(s.ptr); })
        .def("resume", [](GgmlThreadPoolW& s){ ggml_threadpool_resume(s.ptr); });

    // -------------------------------------------------------------------------
    // GGUFContext
    // -------------------------------------------------------------------------
    auto gguf_get_array_value = [](gguf_context* ctx, int64_t key_id) -> nb::object {
        int arr_type = (int) gguf_get_arr_type(ctx, key_id);
        size_t n = gguf_get_arr_n(ctx, key_id);
        if (arr_type == GGUF_TYPE_STRING) {
            nb::list out;
            for (size_t i = 0; i < n; ++i) {
                const char* s = gguf_get_arr_str(ctx, key_id, i);
                if (s) out.append(nb::cast(std::string(s)));
                else   out.append(nb::none());
            }
            return out;
        } else {
            nb::dict d;
            d["type"]   = arr_type;
            d["length"] = n;
            return d;
        }
    };

    nb::class_<GGUFContextW> gguf_cls(m, "GGUFContext");
    gguf_cls
        .def_static("empty", [](){
            auto* w = new GGUFContextW();
            w->ptr = gguf_init_empty();
            if (!w->ptr) { delete w; throw std::bad_alloc(); }
            w->owner = true;
            return nb::cast(w, nb::rv_policy::take_ownership);
        })
        .def_static("from_file", [](const std::string& filename, bool no_alloc){
            auto* w = new GGUFContextW();
            gguf_init_params params{};
            params.no_alloc = no_alloc;
            params.ctx = nullptr;
            w->ptr = gguf_init_from_file(filename.c_str(), params);
            if (!w->ptr) {
                delete w;
                throw std::runtime_error("Failed to load GGUF file: " + filename);
            }
            w->owner = true;
            return nb::cast(w, nb::rv_policy::take_ownership);
        }, "filename"_a, "no_alloc"_a = true)
        .def_prop_ro("version",   [](GGUFContextW& s){ return gguf_get_version(s.ptr); })
        .def_prop_ro("alignment", [](GGUFContextW& s){ return gguf_get_alignment(s.ptr); })
        .def_prop_ro("data_offset", [](GGUFContextW& s){ return gguf_get_data_offset(s.ptr); })
        .def_prop_ro("n_kv",      [](GGUFContextW& s){ return gguf_get_n_kv(s.ptr); })
        .def_prop_ro("n_tensors", [](GGUFContextW& s){ return gguf_get_n_tensors(s.ptr); })
        .def("find_key", [](GGUFContextW& s, const std::string& key){
            return gguf_find_key(s.ptr, key.c_str());
        })
        .def("get_key", [](GGUFContextW& s, int key_id){
            const char* k = gguf_get_key(s.ptr, key_id);
            if (!k) throw std::invalid_argument("Invalid key ID: " + std::to_string(key_id));
            return std::string(k);
        })
        .def("get_kv_type", [](GGUFContextW& s, int key_id){
            return (int) gguf_get_kv_type(s.ptr, key_id);
        })
        .def("get_value", [gguf_get_array_value](GGUFContextW& s, const std::string& key) -> nb::object {
            int64_t key_id = gguf_find_key(s.ptr, key.c_str());
            if (key_id < 0) throw nb::key_error(("Key not found: " + key).c_str());
            int vtype = (int) gguf_get_kv_type(s.ptr, key_id);
            switch (vtype) {
                case GGUF_TYPE_UINT8:   return nb::cast(gguf_get_val_u8(s.ptr, key_id));
                case GGUF_TYPE_INT8:    return nb::cast(gguf_get_val_i8(s.ptr, key_id));
                case GGUF_TYPE_UINT16:  return nb::cast(gguf_get_val_u16(s.ptr, key_id));
                case GGUF_TYPE_INT16:   return nb::cast(gguf_get_val_i16(s.ptr, key_id));
                case GGUF_TYPE_UINT32:  return nb::cast(gguf_get_val_u32(s.ptr, key_id));
                case GGUF_TYPE_INT32:   return nb::cast(gguf_get_val_i32(s.ptr, key_id));
                case GGUF_TYPE_FLOAT32: return nb::cast(gguf_get_val_f32(s.ptr, key_id));
                case GGUF_TYPE_UINT64:  return nb::cast(gguf_get_val_u64(s.ptr, key_id));
                case GGUF_TYPE_INT64:   return nb::cast(gguf_get_val_i64(s.ptr, key_id));
                case GGUF_TYPE_FLOAT64: return nb::cast(gguf_get_val_f64(s.ptr, key_id));
                case GGUF_TYPE_BOOL:    return nb::cast((bool) gguf_get_val_bool(s.ptr, key_id));
                case GGUF_TYPE_STRING: {
                    const char* str = gguf_get_val_str(s.ptr, key_id);
                    return str ? nb::cast(std::string(str)) : nb::none();
                }
                case GGUF_TYPE_ARRAY:   return gguf_get_array_value(s.ptr, key_id);
                default:
                    throw std::runtime_error("Unknown GGUF type: " + std::to_string(vtype));
            }
        })
        .def("get_all_metadata", [gguf_get_array_value](GGUFContextW& s){
            nb::dict out;
            int64_t n = gguf_get_n_kv(s.ptr);
            for (int64_t i = 0; i < n; ++i) {
                const char* k = gguf_get_key(s.ptr, i);
                if (!k) continue;
                std::string key(k);
                try {
                    int vtype = (int) gguf_get_kv_type(s.ptr, i);
                    nb::object val;
                    switch (vtype) {
                        case GGUF_TYPE_UINT8:   val = nb::cast(gguf_get_val_u8(s.ptr, i)); break;
                        case GGUF_TYPE_INT8:    val = nb::cast(gguf_get_val_i8(s.ptr, i)); break;
                        case GGUF_TYPE_UINT16:  val = nb::cast(gguf_get_val_u16(s.ptr, i)); break;
                        case GGUF_TYPE_INT16:   val = nb::cast(gguf_get_val_i16(s.ptr, i)); break;
                        case GGUF_TYPE_UINT32:  val = nb::cast(gguf_get_val_u32(s.ptr, i)); break;
                        case GGUF_TYPE_INT32:   val = nb::cast(gguf_get_val_i32(s.ptr, i)); break;
                        case GGUF_TYPE_FLOAT32: val = nb::cast(gguf_get_val_f32(s.ptr, i)); break;
                        case GGUF_TYPE_UINT64:  val = nb::cast(gguf_get_val_u64(s.ptr, i)); break;
                        case GGUF_TYPE_INT64:   val = nb::cast(gguf_get_val_i64(s.ptr, i)); break;
                        case GGUF_TYPE_FLOAT64: val = nb::cast(gguf_get_val_f64(s.ptr, i)); break;
                        case GGUF_TYPE_BOOL:    val = nb::cast((bool) gguf_get_val_bool(s.ptr, i)); break;
                        case GGUF_TYPE_STRING: {
                            const char* str = gguf_get_val_str(s.ptr, i);
                            val = str ? nb::cast(std::string(str)) : nb::none();
                            break;
                        }
                        case GGUF_TYPE_ARRAY:   val = gguf_get_array_value(s.ptr, i); break;
                        default:
                            val = nb::cast(std::string("<error: unknown type>"));
                    }
                    out[nb::cast(key)] = val;
                } catch (const std::exception& e) {
                    out[nb::cast(key)] = nb::cast(std::string("<error: ") + e.what() + ">");
                }
            }
            return out;
        })
        .def("set_val_str",  [](GGUFContextW& s, const std::string& k, const std::string& v){
            gguf_set_val_str(s.ptr, k.c_str(), v.c_str());
        })
        .def("set_val_bool", [](GGUFContextW& s, const std::string& k, bool v){
            gguf_set_val_bool(s.ptr, k.c_str(), v);
        })
        .def("set_val_u8",   [](GGUFContextW& s, const std::string& k, int v){ gguf_set_val_u8 (s.ptr, k.c_str(), (uint8_t) v); })
        .def("set_val_i8",   [](GGUFContextW& s, const std::string& k, int v){ gguf_set_val_i8 (s.ptr, k.c_str(), (int8_t)  v); })
        .def("set_val_u16",  [](GGUFContextW& s, const std::string& k, int v){ gguf_set_val_u16(s.ptr, k.c_str(), (uint16_t) v); })
        .def("set_val_i16",  [](GGUFContextW& s, const std::string& k, int v){ gguf_set_val_i16(s.ptr, k.c_str(), (int16_t)  v); })
        .def("set_val_u32",  [](GGUFContextW& s, const std::string& k, uint32_t v){ gguf_set_val_u32(s.ptr, k.c_str(), v); })
        .def("set_val_i32",  [](GGUFContextW& s, const std::string& k, int32_t v){  gguf_set_val_i32(s.ptr, k.c_str(), v); })
        .def("set_val_f32",  [](GGUFContextW& s, const std::string& k, float v){    gguf_set_val_f32(s.ptr, k.c_str(), v); })
        .def("set_val_u64",  [](GGUFContextW& s, const std::string& k, uint64_t v){ gguf_set_val_u64(s.ptr, k.c_str(), v); })
        .def("set_val_i64",  [](GGUFContextW& s, const std::string& k, int64_t v){  gguf_set_val_i64(s.ptr, k.c_str(), v); })
        .def("set_val_f64",  [](GGUFContextW& s, const std::string& k, double v){   gguf_set_val_f64(s.ptr, k.c_str(), v); })
        .def("remove_key", [](GGUFContextW& s, const std::string& key){
            return gguf_remove_key(s.ptr, key.c_str());
        })
        .def("find_tensor", [](GGUFContextW& s, const std::string& name){
            return gguf_find_tensor(s.ptr, name.c_str());
        })
        .def("get_tensor_name", [](GGUFContextW& s, int tensor_id){
            const char* n = gguf_get_tensor_name(s.ptr, tensor_id);
            if (!n) throw std::invalid_argument("Invalid tensor ID: " + std::to_string(tensor_id));
            return std::string(n);
        })
        .def("get_tensor_type",   [](GGUFContextW& s, int tensor_id){
            return (int) gguf_get_tensor_type(s.ptr, tensor_id);
        })
        .def("get_tensor_offset", [](GGUFContextW& s, int tensor_id){
            return gguf_get_tensor_offset(s.ptr, tensor_id);
        })
        .def("get_tensor_size",   [](GGUFContextW& s, int tensor_id){
            return gguf_get_tensor_size(s.ptr, tensor_id);
        })
        .def("get_all_tensor_info", [](GGUFContextW& s){
            nb::list out;
            int64_t n = gguf_get_n_tensors(s.ptr);
            for (int64_t i = 0; i < n; ++i) {
                nb::dict d;
                d["id"]     = i;
                const char* nm = gguf_get_tensor_name(s.ptr, i);
                d["name"]   = nm ? std::string(nm) : std::string();
                d["type"]   = (int) gguf_get_tensor_type(s.ptr, i);
                d["offset"] = gguf_get_tensor_offset(s.ptr, i);
                d["size"]   = gguf_get_tensor_size(s.ptr, i);
                out.append(d);
            }
            return out;
        })
        .def("write_to_file", [](GGUFContextW& s, const std::string& filename, bool only_meta){
            return (bool) gguf_write_to_file(s.ptr, filename.c_str(), only_meta);
        }, "filename"_a, "only_meta"_a = false)
        .def("get_meta_size", [](GGUFContextW& s){ return gguf_get_meta_size(s.ptr); })
        .def("__repr__", [](GGUFContextW& s){
            return std::string("<GGUFContext: version=") + std::to_string(gguf_get_version(s.ptr)) +
                   ", tensors=" + std::to_string(gguf_get_n_tensors(s.ptr)) +
                   ", kv_pairs=" + std::to_string(gguf_get_n_kv(s.ptr)) + ">";
        });

    // -------------------------------------------------------------------------
    // Sub-modules (companion TUs)
    // -------------------------------------------------------------------------
    register_tts(m);
    register_mtmd(m);
    register_enums(m);
}
