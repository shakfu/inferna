// nanobind bindings for whisper.cpp. Produces the `_whisper_native`
// extension; the public Python surface lives in `inferna.whisper.whisper_cpp`,
// which re-exports from this module. `inferna.whisper.cli` and the
// `tests/test_whisper*.py` suite import names directly by their bound
// identifiers — keep them stable.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>

#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "whisper.h"

// Forward-declare ggml-backend symbols rather than including ggml-backend.h:
// whisper.cpp and llama.cpp ship their own copies of that header (with subtle
// signature drift), and including either alongside whisper.h's transitive
// ggml.h triggers redefinition errors at compile time. We only need three
// symbols, all stable across both vendor trees.
extern "C" {
    struct ggml_backend_reg;
    typedef ggml_backend_reg* ggml_backend_reg_t;
    ggml_backend_reg_t ggml_backend_load(const char* path);
    void ggml_backend_load_all(void);
    void ggml_backend_load_all_from_path(const char* dir_path);
}

namespace nb = nanobind;
using namespace nb::literals;

// -----------------------------------------------------------------------------
// Wrappers
// -----------------------------------------------------------------------------

struct WhisperContextParamsW {
    whisper_context_params c;
    WhisperContextParamsW() : c(whisper_context_default_params()) {}
};

struct WhisperVadParamsW {
    whisper_vad_params c;
    WhisperVadParamsW() : c(whisper_vad_default_params()) {}
};

struct WhisperFullParamsW {
    whisper_full_params c;
    // Owning storage for char* fields — whisper.cpp keeps the pointers raw,
    // so the wrapper must outlive any whisper_full() call that uses them.
    std::optional<std::string> language_s;
    std::optional<std::string> initial_prompt_s;
    std::optional<std::string> suppress_regex_s;
    std::optional<std::string> vad_model_path_s;

    explicit WhisperFullParamsW(int strategy = (int)WHISPER_SAMPLING_GREEDY)
        : c(whisper_full_default_params((whisper_sampling_strategy)strategy)) {}
};

struct WhisperTokenDataW {
    whisper_token_data c{};
};

// Forward decl — WhisperState references it.
struct WhisperContextW;

// Holds whisper_context* with a Python-level non-blocking thread-safety guard
// (see docs/dev/runtime-guard.md for the rationale).
struct WhisperContextW {
    whisper_context* ctx = nullptr;
    nb::object busy_lock;  // threading.Lock instance, exposed to Python as `_busy_lock`

    WhisperContextW(const std::string& model_path, std::optional<WhisperContextParamsW*> params_opt) {
        // Reuse Python-side validation for typed/clear error messages.
        nb::module_ validation = nb::module_::import_("inferna.utils.validation");
        validation.attr("validate_whisper_file")(model_path, "kind"_a = "whisper model");

        whisper_context_params cp = params_opt && *params_opt
            ? (*params_opt)->c
            : whisper_context_default_params();
        ctx = whisper_init_from_file_with_params(model_path.c_str(), cp);
        if (!ctx) {
            throw std::runtime_error(
                "Failed to load whisper model from " + model_path +
                ". The file passed basic checks but whisper.cpp could not load it. "
                "Possible causes: unsupported model format/version, corrupt file, "
                "or insufficient memory.");
        }
        nb::module_ threading = nb::module_::import_("threading");
        busy_lock = threading.attr("Lock")();
    }

    ~WhisperContextW() {
        if (ctx) {
            whisper_free(ctx);
            ctx = nullptr;
        }
    }

    WhisperContextW(const WhisperContextW&) = delete;
    WhisperContextW& operator=(const WhisperContextW&) = delete;

    void try_acquire_busy() {
        nb::object acquired = busy_lock.attr("acquire")("blocking"_a = false);
        if (!nb::cast<bool>(acquired)) {
            throw std::runtime_error(
                "WhisperContext is currently being used by another thread. "
                "whisper.cpp contexts are not thread-safe -- create one "
                "WhisperContext per thread instead of sharing a single "
                "instance across threads.");
        }
    }
    void release_busy() { busy_lock.attr("release")(); }
};

struct WhisperStateW {
    whisper_state* state = nullptr;
    // Keep the parent context alive so its ctx pointer stays valid.
    nb::object parent;

    WhisperStateW(WhisperContextW* ctx, nb::object parent_)
        : parent(std::move(parent_))
    {
        state = whisper_init_state(ctx->ctx);
        if (!state) {
            throw std::runtime_error("Failed to initialize whisper state");
        }
    }
    ~WhisperStateW() {
        if (state) {
            whisper_free_state(state);
            state = nullptr;
        }
    }
    WhisperStateW(const WhisperStateW&) = delete;
    WhisperStateW& operator=(const WhisperStateW&) = delete;
};

// -----------------------------------------------------------------------------
// Module-level helpers
// -----------------------------------------------------------------------------

// no-op log callback used by disable_logging() below
static void _whisper_no_log_cb(ggml_log_level, const char*, void*) {}

// =============================================================================
// Module
// =============================================================================

NB_MODULE(_whisper_native, m) {
    // -------------------------------------------------------------------------
    // Constant containers exposed as type-namespaced attribute bags
    // (e.g. WHISPER.SAMPLE_RATE, WhisperSamplingStrategy.GREEDY).
    // -------------------------------------------------------------------------
    {
        nb::object pytype = nb::module_::import_("builtins").attr("type");
        nb::dict ns;
        ns["SAMPLE_RATE"] = WHISPER_SAMPLE_RATE;
        ns["N_FFT"]       = WHISPER_N_FFT;
        ns["HOP_LENGTH"]  = WHISPER_HOP_LENGTH;
        ns["CHUNK_SIZE"]  = WHISPER_CHUNK_SIZE;
        m.attr("WHISPER") = pytype("WHISPER", nb::make_tuple(), ns);

        nb::dict ss;
        ss["GREEDY"]      = (int) WHISPER_SAMPLING_GREEDY;
        ss["BEAM_SEARCH"] = (int) WHISPER_SAMPLING_BEAM_SEARCH;
        m.attr("WhisperSamplingStrategy") = pytype("WhisperSamplingStrategy", nb::make_tuple(), ss);

        nb::dict ah;
        ah["NONE"]            = (int) WHISPER_AHEADS_NONE;
        ah["N_TOP_MOST"]      = (int) WHISPER_AHEADS_N_TOP_MOST;
        ah["CUSTOM"]          = (int) WHISPER_AHEADS_CUSTOM;
        ah["TINY_EN"]         = (int) WHISPER_AHEADS_TINY_EN;
        ah["TINY"]            = (int) WHISPER_AHEADS_TINY;
        ah["BASE_EN"]         = (int) WHISPER_AHEADS_BASE_EN;
        ah["BASE"]            = (int) WHISPER_AHEADS_BASE;
        ah["SMALL_EN"]        = (int) WHISPER_AHEADS_SMALL_EN;
        ah["SMALL"]           = (int) WHISPER_AHEADS_SMALL;
        ah["MEDIUM_EN"]       = (int) WHISPER_AHEADS_MEDIUM_EN;
        ah["MEDIUM"]          = (int) WHISPER_AHEADS_MEDIUM;
        ah["LARGE_V1"]        = (int) WHISPER_AHEADS_LARGE_V1;
        ah["LARGE_V2"]        = (int) WHISPER_AHEADS_LARGE_V2;
        ah["LARGE_V3"]        = (int) WHISPER_AHEADS_LARGE_V3;
        ah["LARGE_V3_TURBO"]  = (int) WHISPER_AHEADS_LARGE_V3_TURBO;
        m.attr("WhisperAheadsPreset") = pytype("WhisperAheadsPreset", nb::make_tuple(), ah);

        nb::dict gr;
        gr["END"]            = (int) WHISPER_GRETYPE_END;
        gr["ALT"]            = (int) WHISPER_GRETYPE_ALT;
        gr["RULE_REF"]       = (int) WHISPER_GRETYPE_RULE_REF;
        gr["CHAR"]           = (int) WHISPER_GRETYPE_CHAR;
        gr["CHAR_NOT"]       = (int) WHISPER_GRETYPE_CHAR_NOT;
        gr["CHAR_RNG_UPPER"] = (int) WHISPER_GRETYPE_CHAR_RNG_UPPER;
        gr["CHAR_ALT"]       = (int) WHISPER_GRETYPE_CHAR_ALT;
        m.attr("WhisperGretype") = pytype("WhisperGretype", nb::make_tuple(), gr);
    }

    // -------------------------------------------------------------------------
    // WhisperContextParams
    // -------------------------------------------------------------------------
    nb::class_<WhisperContextParamsW>(m, "WhisperContextParams")
        .def(nb::init<>())
        .def_prop_rw("use_gpu",
            [](WhisperContextParamsW& s){ return (bool)s.c.use_gpu; },
            [](WhisperContextParamsW& s, bool v){ s.c.use_gpu = v; })
        .def_prop_rw("flash_attn",
            [](WhisperContextParamsW& s){ return (bool)s.c.flash_attn; },
            [](WhisperContextParamsW& s, bool v){ s.c.flash_attn = v; })
        .def_prop_rw("gpu_device",
            [](WhisperContextParamsW& s){ return s.c.gpu_device; },
            [](WhisperContextParamsW& s, int v){ s.c.gpu_device = v; })
        .def_prop_rw("dtw_token_timestamps",
            [](WhisperContextParamsW& s){ return (bool)s.c.dtw_token_timestamps; },
            [](WhisperContextParamsW& s, bool v){ s.c.dtw_token_timestamps = v; });

    // -------------------------------------------------------------------------
    // WhisperVadParams
    // -------------------------------------------------------------------------
    nb::class_<WhisperVadParamsW>(m, "WhisperVadParams")
        .def(nb::init<>())
        .def_prop_rw("threshold",
            [](WhisperVadParamsW& s){ return s.c.threshold; },
            [](WhisperVadParamsW& s, float v){ s.c.threshold = v; })
        .def_prop_rw("min_speech_duration_ms",
            [](WhisperVadParamsW& s){ return s.c.min_speech_duration_ms; },
            [](WhisperVadParamsW& s, int v){ s.c.min_speech_duration_ms = v; })
        .def_prop_rw("min_silence_duration_ms",
            [](WhisperVadParamsW& s){ return s.c.min_silence_duration_ms; },
            [](WhisperVadParamsW& s, int v){ s.c.min_silence_duration_ms = v; })
        .def_prop_rw("max_speech_duration_s",
            [](WhisperVadParamsW& s){ return s.c.max_speech_duration_s; },
            [](WhisperVadParamsW& s, float v){ s.c.max_speech_duration_s = v; })
        .def_prop_rw("speech_pad_ms",
            [](WhisperVadParamsW& s){ return s.c.speech_pad_ms; },
            [](WhisperVadParamsW& s, int v){ s.c.speech_pad_ms = v; })
        .def_prop_rw("samples_overlap",
            [](WhisperVadParamsW& s){ return s.c.samples_overlap; },
            [](WhisperVadParamsW& s, float v){ s.c.samples_overlap = v; });

    // -------------------------------------------------------------------------
    // WhisperFullParams
    // -------------------------------------------------------------------------
    auto opt_str_get = [](const char* p) -> nb::object {
        if (!p) return nb::none();
        return nb::cast(std::string(p));
    };

    nb::class_<WhisperFullParamsW>(m, "WhisperFullParams")
        .def(nb::init<int>(), "strategy"_a = (int)WHISPER_SAMPLING_GREEDY)
        .def_prop_rw("strategy",
            [](WhisperFullParamsW& s){ return (int)s.c.strategy; },
            [](WhisperFullParamsW& s, int v){ s.c.strategy = (whisper_sampling_strategy)v; })
        .def_prop_rw("n_threads",
            [](WhisperFullParamsW& s){ return s.c.n_threads; },
            [](WhisperFullParamsW& s, int v){ s.c.n_threads = v; })
        .def_prop_rw("n_max_text_ctx",
            [](WhisperFullParamsW& s){ return s.c.n_max_text_ctx; },
            [](WhisperFullParamsW& s, int v){ s.c.n_max_text_ctx = v; })
        .def_prop_rw("offset_ms",
            [](WhisperFullParamsW& s){ return s.c.offset_ms; },
            [](WhisperFullParamsW& s, int v){ s.c.offset_ms = v; })
        .def_prop_rw("duration_ms",
            [](WhisperFullParamsW& s){ return s.c.duration_ms; },
            [](WhisperFullParamsW& s, int v){ s.c.duration_ms = v; })
        .def_prop_rw("translate",
            [](WhisperFullParamsW& s){ return (bool)s.c.translate; },
            [](WhisperFullParamsW& s, bool v){ s.c.translate = v; })
        .def_prop_rw("no_context",
            [](WhisperFullParamsW& s){ return (bool)s.c.no_context; },
            [](WhisperFullParamsW& s, bool v){ s.c.no_context = v; })
        .def_prop_rw("no_timestamps",
            [](WhisperFullParamsW& s){ return (bool)s.c.no_timestamps; },
            [](WhisperFullParamsW& s, bool v){ s.c.no_timestamps = v; })
        .def_prop_rw("single_segment",
            [](WhisperFullParamsW& s){ return (bool)s.c.single_segment; },
            [](WhisperFullParamsW& s, bool v){ s.c.single_segment = v; })
        .def_prop_rw("print_special",
            [](WhisperFullParamsW& s){ return (bool)s.c.print_special; },
            [](WhisperFullParamsW& s, bool v){ s.c.print_special = v; })
        .def_prop_rw("print_progress",
            [](WhisperFullParamsW& s){ return (bool)s.c.print_progress; },
            [](WhisperFullParamsW& s, bool v){ s.c.print_progress = v; })
        .def_prop_rw("print_realtime",
            [](WhisperFullParamsW& s){ return (bool)s.c.print_realtime; },
            [](WhisperFullParamsW& s, bool v){ s.c.print_realtime = v; })
        .def_prop_rw("print_timestamps",
            [](WhisperFullParamsW& s){ return (bool)s.c.print_timestamps; },
            [](WhisperFullParamsW& s, bool v){ s.c.print_timestamps = v; })
        .def_prop_rw("token_timestamps",
            [](WhisperFullParamsW& s){ return (bool)s.c.token_timestamps; },
            [](WhisperFullParamsW& s, bool v){ s.c.token_timestamps = v; })
        .def_prop_rw("temperature",
            [](WhisperFullParamsW& s){ return s.c.temperature; },
            [](WhisperFullParamsW& s, float v){ s.c.temperature = v; })
        .def_prop_rw("language",
            [opt_str_get](WhisperFullParamsW& s){ return opt_str_get(s.c.language); },
            [](WhisperFullParamsW& s, std::optional<std::string> v) {
                if (!v) { s.c.language = nullptr; s.language_s.reset(); }
                else    { s.language_s = std::move(*v); s.c.language = s.language_s->c_str(); }
            })
        .def_prop_rw("thold_pt",
            [](WhisperFullParamsW& s){ return s.c.thold_pt; },
            [](WhisperFullParamsW& s, float v){ s.c.thold_pt = v; })
        .def_prop_rw("thold_ptsum",
            [](WhisperFullParamsW& s){ return s.c.thold_ptsum; },
            [](WhisperFullParamsW& s, float v){ s.c.thold_ptsum = v; })
        .def_prop_rw("max_len",
            [](WhisperFullParamsW& s){ return s.c.max_len; },
            [](WhisperFullParamsW& s, int v){ s.c.max_len = v; })
        .def_prop_rw("split_on_word",
            [](WhisperFullParamsW& s){ return (bool)s.c.split_on_word; },
            [](WhisperFullParamsW& s, bool v){ s.c.split_on_word = v; })
        .def_prop_rw("max_tokens",
            [](WhisperFullParamsW& s){ return s.c.max_tokens; },
            [](WhisperFullParamsW& s, int v){ s.c.max_tokens = v; })
        .def_prop_rw("debug_mode",
            [](WhisperFullParamsW& s){ return (bool)s.c.debug_mode; },
            [](WhisperFullParamsW& s, bool v){ s.c.debug_mode = v; })
        .def_prop_rw("audio_ctx",
            [](WhisperFullParamsW& s){ return s.c.audio_ctx; },
            [](WhisperFullParamsW& s, int v){ s.c.audio_ctx = v; })
        .def_prop_rw("tdrz_enable",
            [](WhisperFullParamsW& s){ return (bool)s.c.tdrz_enable; },
            [](WhisperFullParamsW& s, bool v){ s.c.tdrz_enable = v; })
        .def_prop_rw("suppress_regex",
            [opt_str_get](WhisperFullParamsW& s){ return opt_str_get(s.c.suppress_regex); },
            [](WhisperFullParamsW& s, std::optional<std::string> v) {
                if (!v) { s.c.suppress_regex = nullptr; s.suppress_regex_s.reset(); }
                else    { s.suppress_regex_s = std::move(*v); s.c.suppress_regex = s.suppress_regex_s->c_str(); }
            })
        .def_prop_rw("initial_prompt",
            [opt_str_get](WhisperFullParamsW& s){ return opt_str_get(s.c.initial_prompt); },
            [](WhisperFullParamsW& s, std::optional<std::string> v) {
                if (!v) { s.c.initial_prompt = nullptr; s.initial_prompt_s.reset(); }
                else    { s.initial_prompt_s = std::move(*v); s.c.initial_prompt = s.initial_prompt_s->c_str(); }
            })
        .def_prop_rw("carry_initial_prompt",
            [](WhisperFullParamsW& s){ return (bool)s.c.carry_initial_prompt; },
            [](WhisperFullParamsW& s, bool v){ s.c.carry_initial_prompt = v; })
        .def_prop_rw("detect_language",
            [](WhisperFullParamsW& s){ return (bool)s.c.detect_language; },
            [](WhisperFullParamsW& s, bool v){ s.c.detect_language = v; })
        .def_prop_rw("suppress_blank",
            [](WhisperFullParamsW& s){ return (bool)s.c.suppress_blank; },
            [](WhisperFullParamsW& s, bool v){ s.c.suppress_blank = v; })
        .def_prop_rw("suppress_nst",
            [](WhisperFullParamsW& s){ return (bool)s.c.suppress_nst; },
            [](WhisperFullParamsW& s, bool v){ s.c.suppress_nst = v; })
        .def_prop_rw("max_initial_ts",
            [](WhisperFullParamsW& s){ return s.c.max_initial_ts; },
            [](WhisperFullParamsW& s, float v){ s.c.max_initial_ts = v; })
        .def_prop_rw("length_penalty",
            [](WhisperFullParamsW& s){ return s.c.length_penalty; },
            [](WhisperFullParamsW& s, float v){ s.c.length_penalty = v; })
        .def_prop_rw("temperature_inc",
            [](WhisperFullParamsW& s){ return s.c.temperature_inc; },
            [](WhisperFullParamsW& s, float v){ s.c.temperature_inc = v; })
        .def_prop_rw("entropy_thold",
            [](WhisperFullParamsW& s){ return s.c.entropy_thold; },
            [](WhisperFullParamsW& s, float v){ s.c.entropy_thold = v; })
        .def_prop_rw("logprob_thold",
            [](WhisperFullParamsW& s){ return s.c.logprob_thold; },
            [](WhisperFullParamsW& s, float v){ s.c.logprob_thold = v; })
        .def_prop_rw("no_speech_thold",
            [](WhisperFullParamsW& s){ return s.c.no_speech_thold; },
            [](WhisperFullParamsW& s, float v){ s.c.no_speech_thold = v; })
        .def_prop_rw("greedy_best_of",
            [](WhisperFullParamsW& s){ return s.c.greedy.best_of; },
            [](WhisperFullParamsW& s, int v){ s.c.greedy.best_of = v; })
        .def_prop_rw("beam_size",
            [](WhisperFullParamsW& s){ return s.c.beam_search.beam_size; },
            [](WhisperFullParamsW& s, int v){ s.c.beam_search.beam_size = v; })
        .def_prop_rw("beam_patience",
            [](WhisperFullParamsW& s){ return s.c.beam_search.patience; },
            [](WhisperFullParamsW& s, float v){ s.c.beam_search.patience = v; })
        .def_prop_rw("grammar_penalty",
            [](WhisperFullParamsW& s){ return s.c.grammar_penalty; },
            [](WhisperFullParamsW& s, float v){ s.c.grammar_penalty = v; })
        .def_prop_rw("vad",
            [](WhisperFullParamsW& s){ return (bool)s.c.vad; },
            [](WhisperFullParamsW& s, bool v){ s.c.vad = v; })
        .def_prop_rw("vad_model_path",
            [opt_str_get](WhisperFullParamsW& s){ return opt_str_get(s.c.vad_model_path); },
            [](WhisperFullParamsW& s, std::optional<std::string> v) {
                if (!v) { s.c.vad_model_path = nullptr; s.vad_model_path_s.reset(); }
                else    { s.vad_model_path_s = std::move(*v); s.c.vad_model_path = s.vad_model_path_s->c_str(); }
            });

    // -------------------------------------------------------------------------
    // WhisperTokenData
    // -------------------------------------------------------------------------
    nb::class_<WhisperTokenDataW>(m, "WhisperTokenData")
        .def(nb::init<>())
        .def_prop_ro("id",    [](WhisperTokenDataW& s){ return s.c.id; })
        .def_prop_ro("tid",   [](WhisperTokenDataW& s){ return s.c.tid; })
        .def_prop_ro("p",     [](WhisperTokenDataW& s){ return s.c.p; })
        .def_prop_ro("plog",  [](WhisperTokenDataW& s){ return s.c.plog; })
        .def_prop_ro("pt",    [](WhisperTokenDataW& s){ return s.c.pt; })
        .def_prop_ro("ptsum", [](WhisperTokenDataW& s){ return s.c.ptsum; })
        .def_prop_ro("t0",    [](WhisperTokenDataW& s){ return s.c.t0; })
        .def_prop_ro("t1",    [](WhisperTokenDataW& s){ return s.c.t1; })
        .def_prop_ro("t_dtw", [](WhisperTokenDataW& s){ return s.c.t_dtw; })
        .def_prop_ro("vlen",  [](WhisperTokenDataW& s){ return s.c.vlen; });

    // -------------------------------------------------------------------------
    // WhisperContext
    // -------------------------------------------------------------------------
    nb::class_<WhisperContextW>(m, "WhisperContext")
        .def(nb::init<const std::string&, std::optional<WhisperContextParamsW*>>(),
             "model_path"_a, "params"_a = nb::none())
        .def_prop_ro("_busy_lock", [](WhisperContextW& s){ return s.busy_lock; })
        .def("_try_acquire_busy", &WhisperContextW::try_acquire_busy)
        .def("version",       [](WhisperContextW&) { return std::string(whisper_version()); })
        .def("system_info",   [](WhisperContextW&) { return std::string(whisper_print_system_info()); })
        .def("n_vocab",            [](WhisperContextW& s){ return whisper_n_vocab(s.ctx); })
        .def("n_text_ctx",         [](WhisperContextW& s){ return whisper_n_text_ctx(s.ctx); })
        .def("n_audio_ctx",        [](WhisperContextW& s){ return whisper_n_audio_ctx(s.ctx); })
        .def("is_multilingual",    [](WhisperContextW& s){ return (bool)whisper_is_multilingual(s.ctx); })
        .def("model_n_vocab",        [](WhisperContextW& s){ return whisper_model_n_vocab(s.ctx); })
        .def("model_n_audio_ctx",    [](WhisperContextW& s){ return whisper_model_n_audio_ctx(s.ctx); })
        .def("model_n_audio_state",  [](WhisperContextW& s){ return whisper_model_n_audio_state(s.ctx); })
        .def("model_n_audio_head",   [](WhisperContextW& s){ return whisper_model_n_audio_head(s.ctx); })
        .def("model_n_audio_layer",  [](WhisperContextW& s){ return whisper_model_n_audio_layer(s.ctx); })
        .def("model_n_text_ctx",     [](WhisperContextW& s){ return whisper_model_n_text_ctx(s.ctx); })
        .def("model_n_text_state",   [](WhisperContextW& s){ return whisper_model_n_text_state(s.ctx); })
        .def("model_n_text_head",    [](WhisperContextW& s){ return whisper_model_n_text_head(s.ctx); })
        .def("model_n_text_layer",   [](WhisperContextW& s){ return whisper_model_n_text_layer(s.ctx); })
        .def("model_n_mels",         [](WhisperContextW& s){ return whisper_model_n_mels(s.ctx); })
        .def("model_ftype",          [](WhisperContextW& s){ return whisper_model_ftype(s.ctx); })
        .def("model_type",           [](WhisperContextW& s){ return whisper_model_type(s.ctx); })
        .def("model_type_readable",  [](WhisperContextW& s){
            return std::string(whisper_model_type_readable(s.ctx)); })
        .def("token_to_str", [](WhisperContextW& s, int token) {
            const char* r = whisper_token_to_str(s.ctx, token);
            return r ? std::string(r) : std::string();
        })
        .def("token_eot",        [](WhisperContextW& s){ return whisper_token_eot(s.ctx); })
        .def("token_sot",        [](WhisperContextW& s){ return whisper_token_sot(s.ctx); })
        .def("token_solm",       [](WhisperContextW& s){ return whisper_token_solm(s.ctx); })
        .def("token_prev",       [](WhisperContextW& s){ return whisper_token_prev(s.ctx); })
        .def("token_nosp",       [](WhisperContextW& s){ return whisper_token_nosp(s.ctx); })
        .def("token_not",        [](WhisperContextW& s){ return whisper_token_not(s.ctx); })
        .def("token_beg",        [](WhisperContextW& s){ return whisper_token_beg(s.ctx); })
        .def("token_lang",       [](WhisperContextW& s, int lang_id){ return whisper_token_lang(s.ctx, lang_id); })
        .def("token_translate",  [](WhisperContextW& s){ return whisper_token_translate(s.ctx); })
        .def("token_transcribe", [](WhisperContextW& s){ return whisper_token_transcribe(s.ctx); })
        .def("tokenize",
             [](WhisperContextW& s, const std::string& text, int max_tokens) {
                 std::vector<whisper_token> tokens(max_tokens);
                 int n = whisper_tokenize(s.ctx, text.c_str(), tokens.data(), max_tokens);
                 if (n < 0) {
                     throw std::runtime_error(
                         "Tokenization failed, need " + std::to_string(-n) +
                         " tokens but only " + std::to_string(max_tokens) + " provided");
                 }
                 std::vector<int> out(tokens.begin(), tokens.begin() + n);
                 return out;
             },
             "text"_a, "max_tokens"_a = 512)
        .def("token_count", [](WhisperContextW& s, const std::string& t){
            return whisper_token_count(s.ctx, t.c_str()); })
        .def("lang_max_id", [](WhisperContextW&) { return whisper_lang_max_id(); })
        .def("lang_id",     [](WhisperContextW&, const std::string& lang) {
            return whisper_lang_id(lang.c_str()); })
        .def("lang_str", [](WhisperContextW&, int id) -> nb::object {
            const char* r = whisper_lang_str(id);
            return r ? nb::cast(std::string(r)) : nb::none();
        })
        .def("lang_str_full", [](WhisperContextW&, int id) -> nb::object {
            const char* r = whisper_lang_str_full(id);
            return r ? nb::cast(std::string(r)) : nb::none();
        })
        .def("encode",
             [](WhisperContextW& s, int offset, int n_threads) {
                 s.try_acquire_busy();
                 int rc = 0;
                 try {
                     rc = whisper_encode(s.ctx, offset, n_threads);
                 } catch (...) { s.release_busy(); throw; }
                 s.release_busy();
                 if (rc != 0) {
                     throw std::runtime_error(
                         "Encoding failed with error " + std::to_string(rc));
                 }
             },
             "offset"_a = 0, "n_threads"_a = 1)
        .def("full",
             [](WhisperContextW& s,
                nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu> samples,
                std::optional<WhisperFullParamsW*> params_opt) {
                 // Hold a default-constructed instance if no params given,
                 // so the c_str-backing storage stays alive for the call.
                 std::unique_ptr<WhisperFullParamsW> default_owner;
                 WhisperFullParamsW* params;
                 if (params_opt && *params_opt) {
                     params = *params_opt;
                 } else {
                     default_owner = std::make_unique<WhisperFullParamsW>();
                     params = default_owner.get();
                 }
                 const float* data = samples.data();
                 int n_samples = (int) samples.shape(0);
                 whisper_context* ctx = s.ctx;
                 whisper_full_params c_params = params->c;

                 s.try_acquire_busy();
                 int rc = 0;
                 try {
                     nb::gil_scoped_release rel;
                     rc = whisper_full(ctx, c_params, data, n_samples);
                 } catch (...) { s.release_busy(); throw; }
                 s.release_busy();
                 if (rc != 0) {
                     throw std::runtime_error(
                         "Whisper full processing failed with error " + std::to_string(rc));
                 }
                 return rc;
             },
             "samples"_a, "params"_a = nb::none())
        .def("full_n_segments",  [](WhisperContextW& s){ return whisper_full_n_segments(s.ctx); })
        .def("full_lang_id",     [](WhisperContextW& s){ return whisper_full_lang_id(s.ctx); })
        .def("full_get_segment_t0", [](WhisperContextW& s, int i){
            return whisper_full_get_segment_t0(s.ctx, i); })
        .def("full_get_segment_t1", [](WhisperContextW& s, int i){
            return whisper_full_get_segment_t1(s.ctx, i); })
        .def("full_get_segment_text", [](WhisperContextW& s, int i){
            const char* r = whisper_full_get_segment_text(s.ctx, i);
            return r ? std::string(r) : std::string();
        })
        .def("full_n_tokens", [](WhisperContextW& s, int i){
            return whisper_full_n_tokens(s.ctx, i); })
        .def("full_get_token_text", [](WhisperContextW& s, int i, int j){
            const char* r = whisper_full_get_token_text(s.ctx, i, j);
            return r ? std::string(r) : std::string();
        })
        .def("full_get_token_id", [](WhisperContextW& s, int i, int j){
            return whisper_full_get_token_id(s.ctx, i, j); })
        .def("full_get_token_data", [](WhisperContextW& s, int i, int j){
            WhisperTokenDataW out;
            out.c = whisper_full_get_token_data(s.ctx, i, j);
            return out;
        })
        .def("full_get_token_p", [](WhisperContextW& s, int i, int j){
            return whisper_full_get_token_p(s.ctx, i, j); })
        .def("full_get_segment_no_speech_prob", [](WhisperContextW& s, int i){
            return whisper_full_get_segment_no_speech_prob(s.ctx, i); })
        .def("print_timings", [](WhisperContextW& s){ whisper_print_timings(s.ctx); })
        .def("reset_timings", [](WhisperContextW& s){ whisper_reset_timings(s.ctx); });

    // -------------------------------------------------------------------------
    // WhisperState — keeps a reference to its parent context.
    // -------------------------------------------------------------------------
    nb::class_<WhisperStateW>(m, "WhisperState")
        .def("__init__",
             [](WhisperStateW* self, nb::object ctx_obj) {
                 WhisperContextW* ctx = nb::cast<WhisperContextW*>(ctx_obj);
                 new (self) WhisperStateW(ctx, ctx_obj);
             },
             "ctx"_a);

    // -------------------------------------------------------------------------
    // Module-level functions
    // -------------------------------------------------------------------------
    m.def("ggml_backend_load_all", [](){
        // Mirror the Python-side backend discovery used by the llama module.
        nb::module_ os = nb::module_::import_("os");
        nb::module_ backend_dl = nb::module_::import_("inferna._internal.backend_dl");
        nb::object __file__ = nb::module_::import_("inferna.whisper.whisper_cpp").attr("__file__");
        std::string this_file = nb::cast<std::string>(__file__);
        std::string this_dir  = nb::cast<std::string>(os.attr("path").attr("dirname")(
            os.attr("path").attr("abspath")(this_file)));
        std::string llama_dir = nb::cast<std::string>(os.attr("path").attr("join")(
            os.attr("path").attr("dirname")(this_dir), "llama"));
        if (nb::cast<bool>(os.attr("path").attr("isdir")(llama_dir))) {
            ggml_backend_load_all_from_path(llama_dir.c_str());
        } else {
            ggml_backend_load_all_from_path(this_dir.c_str());
        }
        std::string site = nb::cast<std::string>(
            os.attr("path").attr("dirname")(os.attr("path").attr("dirname")(this_dir)));
        nb::object paths = backend_dl.attr("libs_to_load")(site);
        for (nb::handle p : paths) {
            std::string sp = nb::cast<std::string>(p);
            ggml_backend_load(sp.c_str());
        }
    }, "Load all available ggml backends (CUDA, Metal, Vulkan, etc.).");

    m.def("disable_logging", [](){
        whisper_log_set(_whisper_no_log_cb, nullptr);
    }, "Suppress all C-level log output from whisper.cpp and ggml.");

    m.def("version",           [](){ return std::string(whisper_version()); });
    m.def("print_system_info", [](){ return std::string(whisper_print_system_info()); });
    m.def("lang_max_id",       [](){ return whisper_lang_max_id(); });
    m.def("lang_id",  [](const std::string& s){ return whisper_lang_id(s.c_str()); }, "lang"_a);
    m.def("lang_str", [](int id) -> std::optional<std::string> {
        const char* r = whisper_lang_str(id);
        if (!r) return std::nullopt;
        return std::string(r);
    }, "id"_a);
    m.def("lang_str_full", [](int id) -> std::optional<std::string> {
        const char* r = whisper_lang_str_full(id);
        if (!r) return std::nullopt;
        return std::string(r);
    }, "id"_a);
}
