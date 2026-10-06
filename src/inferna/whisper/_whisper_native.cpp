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

#include "common/busy_lock.hpp"

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

// Pulled in after the extern "C" forward decls so the inline helper sees
// `ggml_backend_load*` declarations.
#include "common/backend_loader.hpp"

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
// Where full() results live: a context's default state (st == nullptr) or a
// WhisperState. Each getter picks the matching whisper.h function.
struct WhisperResult {
    whisper_context* ctx;
    whisper_state* st;

    int n_segments() const {
        return st ? whisper_full_n_segments_from_state(st) : whisper_full_n_segments(ctx);
    }
    int lang_id() const {
        return st ? whisper_full_lang_id_from_state(st) : whisper_full_lang_id(ctx);
    }
    int64_t segment_t0(int i) const {
        return st ? whisper_full_get_segment_t0_from_state(st, i) : whisper_full_get_segment_t0(ctx, i);
    }
    int64_t segment_t1(int i) const {
        return st ? whisper_full_get_segment_t1_from_state(st, i) : whisper_full_get_segment_t1(ctx, i);
    }
    bool speaker_turn_next(int i) const {
        return st ? whisper_full_get_segment_speaker_turn_next_from_state(st, i)
                  : whisper_full_get_segment_speaker_turn_next(ctx, i);
    }
    const char* segment_text(int i) const {
        return st ? whisper_full_get_segment_text_from_state(st, i) : whisper_full_get_segment_text(ctx, i);
    }
    float no_speech_prob(int i) const {
        return st ? whisper_full_get_segment_no_speech_prob_from_state(st, i)
                  : whisper_full_get_segment_no_speech_prob(ctx, i);
    }
    int n_tokens(int i) const {
        return st ? whisper_full_n_tokens_from_state(st, i) : whisper_full_n_tokens(ctx, i);
    }
    const char* token_text(int i, int j) const {
        return st ? whisper_full_get_token_text_from_state(ctx, st, i, j) : whisper_full_get_token_text(ctx, i, j);
    }
    whisper_token token_id(int i, int j) const {
        return st ? whisper_full_get_token_id_from_state(st, i, j) : whisper_full_get_token_id(ctx, i, j);
    }
    whisper_token_data token_data(int i, int j) const {
        return st ? whisper_full_get_token_data_from_state(st, i, j) : whisper_full_get_token_data(ctx, i, j);
    }
    float token_p(int i, int j) const {
        return st ? whisper_full_get_token_p_from_state(st, i, j) : whisper_full_get_token_p(ctx, i, j);
    }
    int64_t token_t0(int i, int j) const {
        return st ? whisper_full_get_token_t0_from_state(st, i, j) : whisper_full_get_token_t0(ctx, i, j);
    }
    int64_t token_t1(int i, int j) const {
        return st ? whisper_full_get_token_t1_from_state(st, i, j) : whisper_full_get_token_t1(ctx, i, j);
    }
    int n_vad_segments() const {
        return st ? whisper_full_n_vad_segments_from_state(st) : whisper_full_n_vad_segments(ctx);
    }
    int64_t vad_segment_t0(int i) const {
        return st ? whisper_full_get_vad_segment_t0_from_state(st, i) : whisper_full_get_vad_segment_t0(ctx, i);
    }
    int64_t vad_segment_t1(int i) const {
        return st ? whisper_full_get_vad_segment_t1_from_state(st, i) : whisper_full_get_vad_segment_t1(ctx, i);
    }

    int pcm_to_mel(const float* x, int n, int n_threads) const {
        return st ? whisper_pcm_to_mel_with_state(ctx, st, x, n, n_threads) : whisper_pcm_to_mel(ctx, x, n, n_threads);
    }
    int set_mel(const float* x, int n_len, int n_mel) const {
        return st ? whisper_set_mel_with_state(ctx, st, x, n_len, n_mel) : whisper_set_mel(ctx, x, n_len, n_mel);
    }
    int n_len() const { return st ? whisper_n_len_from_state(st) : whisper_n_len(ctx); }
    int encode(int offset, int n_threads) const {
        return st ? whisper_encode_with_state(ctx, st, offset, n_threads) : whisper_encode(ctx, offset, n_threads);
    }
    int decode(const whisper_token* t, int n, int n_past, int n_threads) const {
        return st ? whisper_decode_with_state(ctx, st, t, n, n_past, n_threads)
                  : whisper_decode(ctx, t, n, n_past, n_threads);
    }
    const float* logits() const { return st ? whisper_get_logits_from_state(st) : whisper_get_logits(ctx); }
    int lang_auto_detect(int offset_ms, int n_threads, float* probs) const {
        return st ? whisper_lang_auto_detect_with_state(ctx, st, offset_ms, n_threads, probs)
                  : whisper_lang_auto_detect(ctx, offset_ms, n_threads, probs);
    }

    // whisper.cpp indexes its result vectors without bounds checks
    static void check(const char* what, int i, int n) {
        if (i < 0 || i >= n) throw std::out_of_range(
            std::string(what) + " " + std::to_string(i) + " is outside [0, " + std::to_string(n) + ")");
    }
    const WhisperResult& segment(int i) const { check("segment", i, n_segments()); return *this; }
    const WhisperResult& token(int i, int j) const { segment(i); check("token", j, n_tokens(i)); return *this; }
    const WhisperResult& vad_segment(int i) const { check("VAD segment", i, n_vad_segments()); return *this; }
};

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

    static constexpr const char* kBusyMsg =
        "WhisperContext is currently being used by another thread. "
        "whisper.cpp contexts are not thread-safe -- create one "
        "WhisperContext per thread instead of sharing a single "
        "instance across threads.";

    // Throw a normal Python exception if the context has been closed,
    // instead of letting whisper.cpp dereference a null pointer.
    void ensure_valid() const {
        if (!ctx) throw std::runtime_error(
            "WhisperContext has been closed and is no longer usable");
    }

    // Live WhisperStates; each uses ctx, so close() refuses while any exist.
    int n_states = 0;
    // Tokens in the last decode(); 0 once anything else rewrites the logits.
    int n_decoded = 0;

    WhisperResult res() const { ensure_valid(); return {ctx, nullptr}; }

    void try_acquire_busy() {
        nb::object acquired = busy_lock.attr("acquire")("blocking"_a = false);
        if (!nb::cast<bool>(acquired)) {
            throw std::runtime_error(kBusyMsg);
        }
    }
    void release_busy() { busy_lock.attr("release")(); }
};

struct WhisperStateW {
    whisper_state* state = nullptr;
    WhisperContextW* ctx = nullptr;
    nb::object parent;     // keeps ctx alive
    nb::object busy_lock;  // one native call at a time per state
    int n_decoded = 0;     // see WhisperContextW::n_decoded

    static constexpr const char* kBusyMsg =
        "WhisperState is currently being used by another thread. "
        "Create one WhisperState per thread.";

    explicit WhisperStateW(nb::object parent_) : parent(std::move(parent_)) {
        ctx = nb::cast<WhisperContextW*>(parent);
        ctx->ensure_valid();
        state = whisper_init_state(ctx->ctx);
        if (!state) {
            throw std::runtime_error("Failed to initialize whisper state");
        }
        ++ctx->n_states;
        busy_lock = nb::module_::import_("threading").attr("Lock")();
    }
    void free_state() {
        if (!state) return;
        whisper_free_state(state);
        state = nullptr;
        --ctx->n_states;
    }
    ~WhisperStateW() { free_state(); }

    WhisperResult res() const {
        if (!state) throw std::runtime_error("WhisperState has been closed and is no longer usable");
        return {ctx->ctx, state};
    }
    WhisperStateW(const WhisperStateW&) = delete;
    WhisperStateW& operator=(const WhisperStateW&) = delete;
};

struct WhisperVadContextParamsW {
    whisper_vad_context_params c = whisper_vad_default_context_params();
};

// A Silero VAD model, separate from any WhisperContext.
struct WhisperVadContextW {
    whisper_vad_context* ctx = nullptr;
    nb::object busy_lock;

    static constexpr const char* kBusyMsg =
        "WhisperVadContext is currently being used by another thread. "
        "Create one WhisperVadContext per thread.";

    WhisperVadContextW(const std::string& path, std::optional<WhisperVadContextParamsW*> params) {
        nb::module_::import_("inferna.utils.validation").attr("validate_whisper_vad_file")(path);
        whisper_vad_context_params p = params && *params ? (*params)->c : whisper_vad_default_context_params();
        ctx = whisper_vad_init_from_file_with_params(path.c_str(), p);
        if (!ctx) throw std::runtime_error("Failed to load VAD model from " + path);
        busy_lock = nb::module_::import_("threading").attr("Lock")();
    }
    ~WhisperVadContextW() { whisper_vad_free(ctx); }
    WhisperVadContextW(const WhisperVadContextW&) = delete;
    WhisperVadContextW& operator=(const WhisperVadContextW&) = delete;

    whisper_vad_context* get() const {
        if (!ctx) throw std::runtime_error("WhisperVadContext has been closed and is no longer usable");
        return ctx;
    }
};

// -----------------------------------------------------------------------------
// Module-level helpers
// -----------------------------------------------------------------------------

using Samples = nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// Run one whisper_full* call under `lock` with the GIL released; no params
// means defaults. `call` takes the C params and returns whisper's code.
template <class F>
static void run_full(nb::object& lock, const char* busy_msg,
                     std::optional<WhisperFullParamsW*> params, F&& call) {
    // a default instance must outlive the call: it backs the C strings
    std::unique_ptr<WhisperFullParamsW> default_owner;
    WhisperFullParamsW* p = params && *params ? *params
                          : (default_owner = std::make_unique<WhisperFullParamsW>()).get();
    whisper_full_params c = p->c;
    inferna::BusyGuard guard(lock, busy_msg);
    int rc;
    {
        nb::gil_scoped_release rel;
        rc = call(c);
    }
    if (rc != 0) throw std::runtime_error(
        "Whisper full processing failed with error " + std::to_string(rc));
}

// Holds W's busy lock while reading results, for one full expression:
// full() clears and rebuilds them on another thread with the GIL released.
template <class W>
struct Reader {
    inferna::BusyGuard guard;
    WhisperResult r;
    explicit Reader(W& s) : guard(s.busy_lock, W::kBusyMsg), r(s.res()) {}
    const WhisperResult* operator->() const { return &r; }
};

static std::string str_or_empty(const char* p) { return p ? std::string(p) : std::string(); }

// The full_* result getters, shared by WhisperContext and WhisperState.
template <class W>
static void bind_results(nb::class_<W>& cls) {
    cls
        .def("full_n_segments", [](W& s){ return Reader<W>(s)->n_segments(); })
        .def("full_lang_id",    [](W& s){ return Reader<W>(s)->lang_id(); })
        .def("full_get_segment_t0", [](W& s, int i){ return Reader<W>(s)->segment(i).segment_t0(i); }, "i"_a)
        .def("full_get_segment_t1", [](W& s, int i){ return Reader<W>(s)->segment(i).segment_t1(i); }, "i"_a)
        .def("full_get_segment_speaker_turn_next",
             [](W& s, int i){ return Reader<W>(s)->segment(i).speaker_turn_next(i); },
             "i"_a, "True if the next segment is predicted as a speaker turn (needs tdrz_enable).")
        .def("full_get_segment_text", [](W& s, int i){
            return str_or_empty(Reader<W>(s)->segment(i).segment_text(i));
        }, "i"_a)
        .def("full_get_segment_no_speech_prob",
             [](W& s, int i){ return Reader<W>(s)->segment(i).no_speech_prob(i); }, "i"_a)
        .def("full_n_tokens", [](W& s, int i){ return Reader<W>(s)->segment(i).n_tokens(i); }, "i"_a)
        .def("full_get_token_text", [](W& s, int i, int j){
            return str_or_empty(Reader<W>(s)->token(i, j).token_text(i, j));
        }, "i"_a, "j"_a)
        .def("full_get_token_id", [](W& s, int i, int j){ return Reader<W>(s)->token(i, j).token_id(i, j); },
             "i"_a, "j"_a)
        .def("full_get_token_data", [](W& s, int i, int j){
            WhisperTokenDataW out;
            out.c = Reader<W>(s)->token(i, j).token_data(i, j);
            return out;
        }, "i"_a, "j"_a)
        .def("full_get_token_p", [](W& s, int i, int j){ return Reader<W>(s)->token(i, j).token_p(i, j); },
             "i"_a, "j"_a)
        .def("full_get_token_t0", [](W& s, int i, int j){ return Reader<W>(s)->token(i, j).token_t0(i, j); },
             "i"_a, "j"_a, "Token start in centiseconds on the original audio timeline (VAD-mapped).")
        .def("full_get_token_t1", [](W& s, int i, int j){ return Reader<W>(s)->token(i, j).token_t1(i, j); },
             "i"_a, "j"_a, "Token end in centiseconds on the original audio timeline (VAD-mapped).")
        .def("full_n_vad_segments", [](W& s){ return Reader<W>(s)->n_vad_segments(); },
             "Speech segments found by VAD in the last full() call; 0 without VAD.")
        .def("full_get_vad_segment_t0", [](W& s, int i){ return Reader<W>(s)->vad_segment(i).vad_segment_t0(i); },
             "i"_a, "VAD segment start in centiseconds on the original audio timeline.")
        .def("full_get_vad_segment_t1", [](W& s, int i){ return Reader<W>(s)->vad_segment(i).vad_segment_t1(i); },
             "i"_a, "VAD segment end in centiseconds on the original audio timeline.");
}

// Hold W's busy lock and release the GIL around a native call.
template <class W, class F>
static int run_locked(W& s, F&& call) {
    inferna::BusyGuard guard(s.busy_lock, W::kBusyMsg);
    nb::gil_scoped_release rel;
    return call();
}

static void check_rc(int rc, const char* what) {
    if (rc != 0) throw std::runtime_error(std::string(what) + " failed with error " + std::to_string(rc));
}

// The step-by-step pipeline (mel -> encode -> decode), shared by
// WhisperContext and WhisperState. full() runs all of it.
template <class W>
static void bind_pipeline(nb::class_<W>& cls) {
    cls
        .def("pcm_to_mel", [](W& s, Samples samples, int n_threads) {
            WhisperResult r = s.res();
            check_rc(run_locked(s, [&]{ return r.pcm_to_mel(samples.data(), (int) samples.shape(0), n_threads); }),
                     "pcm_to_mel");
        }, "samples"_a, "n_threads"_a = 1, "Compute the log mel spectrogram of 16 kHz mono samples.")
        .def("set_mel", [](W& s, nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> mel) {
            WhisperResult r = s.res();
            // whisper.cpp copies n_mel * n_len floats; take both from the shape
            int n_mel = (int) mel.shape(0), n_len = (int) mel.shape(1);
            check_rc(run_locked(s, [&]{ return r.set_mel(mel.data(), n_len, n_mel); }), "set_mel");
        }, "mel"_a, "Set a log mel spectrogram of shape (n_mels, n_len) instead of pcm_to_mel().")
        .def("n_len", [](W& s){ return Reader<W>(s)->n_len(); }, "Mel length in frames (10 ms each).")
        .def("encode", [](W& s, int offset, int n_threads) {
            WhisperResult r = s.res();
            // whisper.cpp clamps the offset only from above
            if (offset < 0) throw std::invalid_argument("offset must be >= 0");
            check_rc(run_locked(s, [&]{ return r.encode(offset, n_threads); }), "encode");
        }, "offset"_a = 0, "n_threads"_a = 1, "Run the encoder on the mel from frame `offset`.")
        .def("decode", [](W& s, const std::vector<whisper_token>& tokens, int n_past, int n_threads) {
            WhisperResult r = s.res();
            // the batch and positional embedding hold n_text_ctx entries
            int n = (int) tokens.size(), n_ctx = whisper_n_text_ctx(r.ctx), n_vocab = whisper_model_n_vocab(r.ctx);
            if (n < 1) throw std::invalid_argument("tokens must not be empty");
            if (n_past < 0 || n_past + n > n_ctx) throw std::invalid_argument(
                "n_past + len(tokens) = " + std::to_string(n_past + n) + " is outside [1, n_text_ctx=" +
                std::to_string(n_ctx) + "]");
            for (whisper_token t : tokens)
                if (t < 0 || t >= n_vocab) throw std::invalid_argument("invalid token id " + std::to_string(t));
            s.n_decoded = 0;
            check_rc(run_locked(s, [&]{ return r.decode(tokens.data(), n, n_past, n_threads); }), "decode");
            s.n_decoded = n;
        }, "tokens"_a, "n_past"_a = 0, "n_threads"_a = 1,
           "Decode tokens after the first n_past cached ones; call encode() first.")
        .def("get_logits", [](W& s) {
            Reader<W> rd(s);
            const WhisperResult& r = rd.r;
            // only the last token's row is computed; the others are stale
            if (!s.n_decoded) throw std::runtime_error("no logits; call decode() first");
            size_t n_vocab = (size_t) whisper_model_n_vocab(r.ctx);
            float* out = new float[n_vocab];
            std::memcpy(out, r.logits() + (size_t)(s.n_decoded - 1) * n_vocab, n_vocab * sizeof(float));
            nb::capsule owner(out, [](void* p) noexcept { delete[] (float*) p; });
            return nb::ndarray<nb::numpy, float, nb::ndim<1>>(out, {n_vocab}, owner);
        }, "Logits of the last token of the last decode(), shape (n_vocab,).")
        .def("lang_auto_detect", [](W& s, int offset_ms, int n_threads) {
            WhisperResult r = s.res();
            // an English-only vocab has no language tokens
            if (!whisper_is_multilingual(r.ctx))
                throw std::invalid_argument("lang_auto_detect needs a multilingual model");
            size_t n = (size_t) whisper_lang_max_id() + 1;
            float* probs = new float[n]();
            nb::capsule owner(probs, [](void* p) noexcept { delete[] (float*) p; });
            s.n_decoded = 0;
            int id = run_locked(s, [&]{ return r.lang_auto_detect(offset_ms, n_threads, probs); });
            if (id < 0) throw std::runtime_error("lang_auto_detect failed with error " + std::to_string(id));
            return nb::make_tuple(id, nb::ndarray<nb::numpy, float, nb::ndim<1>>(probs, {n}, owner));
        }, "offset_ms"_a = 0, "n_threads"_a = 1,
           "Detect the language from the mel at offset_ms: (lang_id, probs indexed by lang id).");
}

// [(t0, t1)] in centiseconds; takes ownership of `seg`.
static nb::list vad_segments_to_list(whisper_vad_segments* seg) {
    if (!seg) throw std::runtime_error("VAD segmentation failed");
    std::unique_ptr<whisper_vad_segments, void (*)(whisper_vad_segments*)> own(seg, whisper_vad_free_segments);
    nb::list out;
    for (int i = 0, n = whisper_vad_segments_n_segments(seg); i < n; ++i)
        out.append(nb::make_tuple(whisper_vad_segments_get_segment_t0(seg, i),
                                  whisper_vad_segments_get_segment_t1(seg, i)));
    return out;
}

static whisper_vad_params vad_params_or_default(std::optional<WhisperVadParamsW*> p) {
    return p && *p ? (*p)->c : whisper_vad_default_params();
}

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
    nb::class_<WhisperContextParamsW>(m, "WhisperContextParams",
        "Parameters for loading a WhisperContext: GPU use, flash attention, "
        "GPU device index, optional DTW token-timestamp alignment.")
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
            [](WhisperContextParamsW& s, bool v){ s.c.dtw_token_timestamps = v; })
        // DTW alignment-head selection. dtw_token_timestamps has no effect
        // without a preset matching the model (see WhisperAheadsPreset).
        .def_prop_rw("dtw_aheads_preset",
            [](WhisperContextParamsW& s){ return (int)s.c.dtw_aheads_preset; },
            [](WhisperContextParamsW& s, int v){
                s.c.dtw_aheads_preset = (enum whisper_alignment_heads_preset) v; })
        .def_prop_rw("dtw_n_top",
            [](WhisperContextParamsW& s){ return s.c.dtw_n_top; },
            [](WhisperContextParamsW& s, int v){ s.c.dtw_n_top = v; })
        .def_prop_rw("dtw_mem_size",
            [](WhisperContextParamsW& s){ return (int64_t)s.c.dtw_mem_size; },
            [](WhisperContextParamsW& s, int64_t v){ s.c.dtw_mem_size = (size_t)v; });

    // -------------------------------------------------------------------------
    // WhisperVadParams
    // -------------------------------------------------------------------------
    nb::class_<WhisperVadParamsW>(m, "WhisperVadParams",
        "Voice-activity detection parameters: speech/silence thresholds and "
        "min/max durations applied to chunk audio before transcription.")
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

    nb::class_<WhisperFullParamsW>(m, "WhisperFullParams",
        "Parameters for whisper_full: sampling strategy, language, thread/segment "
        "controls, prompt tokens, timestamp options, VAD config.")
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
            })
        // vad_params: the embedded whisper_vad_params (tuning for VAD chunking).
        // Returns/accepts a WhisperVadParams; the struct is held by value.
        .def_prop_rw("vad_params",
            [](WhisperFullParamsW& s){
                auto* w = new WhisperVadParamsW();
                w->c = s.c.vad_params;
                return nb::cast(w, nb::rv_policy::take_ownership);
            },
            [](WhisperFullParamsW& s, WhisperVadParamsW& v){ s.c.vad_params = v.c; });

    // -------------------------------------------------------------------------
    // WhisperTokenData
    // -------------------------------------------------------------------------
    nb::class_<WhisperTokenDataW>(m, "WhisperTokenData",
        "Per-token result from whisper inference: id, text-token id, log-probabilities, "
        "voice-print, and start/end timestamps.")
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
    nb::class_<WhisperContextW> ctx_cls(m, "WhisperContext",
        "A loaded whisper.cpp model + inference state. Run transcription via "
        "full(samples, params); read results via the n_segments / segment_text APIs.");
    ctx_cls
        .def(nb::init<const std::string&, std::optional<WhisperContextParamsW*>>(),
             "model_path"_a, "params"_a = nb::none())
        .def_prop_ro("_busy_lock", [](WhisperContextW& s){ return s.busy_lock; })
        .def("_try_acquire_busy", &WhisperContextW::try_acquire_busy)
        .def("close", [](WhisperContextW& s){
            s.ensure_valid();
            // a running full() or encode() holds the lock
            inferna::BusyGuard guard(s.busy_lock, WhisperContextW::kBusyMsg);
            if (s.n_states) throw std::runtime_error(
                "WhisperContext still has " + std::to_string(s.n_states) +
                " open WhisperState(s); close or delete them first");
            whisper_free(s.ctx);
            s.ctx = nullptr;
        })
        .def_prop_ro("is_valid", [](WhisperContextW& s){ return s.ctx != nullptr; })
        .def("version",       [](WhisperContextW&) { return std::string(whisper_version()); })
        .def("system_info",   [](WhisperContextW&) { return std::string(whisper_print_system_info()); })
        .def("n_vocab",            [](WhisperContextW& s){ s.ensure_valid(); return whisper_n_vocab(s.ctx); })
        .def("n_text_ctx",         [](WhisperContextW& s){ s.ensure_valid(); return whisper_n_text_ctx(s.ctx); })
        .def("n_audio_ctx",        [](WhisperContextW& s){ s.ensure_valid(); return whisper_n_audio_ctx(s.ctx); })
        .def("is_multilingual",    [](WhisperContextW& s){ s.ensure_valid(); return (bool)whisper_is_multilingual(s.ctx); })
        .def("model_n_vocab",        [](WhisperContextW& s){ s.ensure_valid(); return whisper_model_n_vocab(s.ctx); })
        .def("model_n_audio_ctx",    [](WhisperContextW& s){ s.ensure_valid(); return whisper_model_n_audio_ctx(s.ctx); })
        .def("model_n_audio_state",  [](WhisperContextW& s){ s.ensure_valid(); return whisper_model_n_audio_state(s.ctx); })
        .def("model_n_audio_head",   [](WhisperContextW& s){ s.ensure_valid(); return whisper_model_n_audio_head(s.ctx); })
        .def("model_n_audio_layer",  [](WhisperContextW& s){ s.ensure_valid(); return whisper_model_n_audio_layer(s.ctx); })
        .def("model_n_text_ctx",     [](WhisperContextW& s){ s.ensure_valid(); return whisper_model_n_text_ctx(s.ctx); })
        .def("model_n_text_state",   [](WhisperContextW& s){ s.ensure_valid(); return whisper_model_n_text_state(s.ctx); })
        .def("model_n_text_head",    [](WhisperContextW& s){ s.ensure_valid(); return whisper_model_n_text_head(s.ctx); })
        .def("model_n_text_layer",   [](WhisperContextW& s){ s.ensure_valid(); return whisper_model_n_text_layer(s.ctx); })
        .def("model_n_mels",         [](WhisperContextW& s){ s.ensure_valid(); return whisper_model_n_mels(s.ctx); })
        .def("model_ftype",          [](WhisperContextW& s){ s.ensure_valid(); return whisper_model_ftype(s.ctx); })
        .def("model_type",           [](WhisperContextW& s){ s.ensure_valid(); return whisper_model_type(s.ctx); })
        .def("model_type_readable",  [](WhisperContextW& s){
            s.ensure_valid();
            return std::string(whisper_model_type_readable(s.ctx)); })
        .def("token_to_str", [](WhisperContextW& s, int token) {
            s.ensure_valid();
            const char* r = whisper_token_to_str(s.ctx, token);
            return r ? std::string(r) : std::string();
        })
        .def("token_eot",        [](WhisperContextW& s){ s.ensure_valid(); return whisper_token_eot(s.ctx); })
        .def("token_sot",        [](WhisperContextW& s){ s.ensure_valid(); return whisper_token_sot(s.ctx); })
        .def("token_solm",       [](WhisperContextW& s){ s.ensure_valid(); return whisper_token_solm(s.ctx); })
        .def("token_prev",       [](WhisperContextW& s){ s.ensure_valid(); return whisper_token_prev(s.ctx); })
        .def("token_nosp",       [](WhisperContextW& s){ s.ensure_valid(); return whisper_token_nosp(s.ctx); })
        .def("token_not",        [](WhisperContextW& s){ s.ensure_valid(); return whisper_token_not(s.ctx); })
        .def("token_beg",        [](WhisperContextW& s){ s.ensure_valid(); return whisper_token_beg(s.ctx); })
        .def("token_lang",       [](WhisperContextW& s, int lang_id){ s.ensure_valid(); return whisper_token_lang(s.ctx, lang_id); })
        .def("token_translate",  [](WhisperContextW& s){ s.ensure_valid(); return whisper_token_translate(s.ctx); })
        .def("token_transcribe", [](WhisperContextW& s){ s.ensure_valid(); return whisper_token_transcribe(s.ctx); })
        .def("tokenize",
             [](WhisperContextW& s, const std::string& text, int max_tokens) {
                 s.ensure_valid();
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
            s.ensure_valid();
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
        .def("full",
             [](WhisperContextW& s, Samples samples, std::optional<WhisperFullParamsW*> params) {
                 whisper_context* ctx = s.res().ctx;
                 s.n_decoded = 0;
                 run_full(s.busy_lock, WhisperContextW::kBusyMsg, params, [&](whisper_full_params p) {
                     return whisper_full(ctx, p, samples.data(), (int) samples.shape(0));
                 });
                 return 0;
             },
             "samples"_a, "params"_a = nb::none())
        .def("full_parallel",
             [](WhisperContextW& s, Samples samples, std::optional<WhisperFullParamsW*> params, int n_processors) {
                 whisper_context* ctx = s.res().ctx;
                 // whisper.cpp sizes a vector with n_processors - 1
                 if (n_processors < 1) throw std::invalid_argument("n_processors must be >= 1");
                 s.n_decoded = 0;
                 run_full(s.busy_lock, WhisperContextW::kBusyMsg, params, [&](whisper_full_params p) {
                     return whisper_full_parallel(ctx, p, samples.data(), (int) samples.shape(0), n_processors);
                 });
             },
             "samples"_a, "params"_a = nb::none(), "n_processors"_a = 2,
             "Split the audio into n_processors chunks run on parallel states; results "
             "are merged into this context. Words at chunk boundaries can be lost.")
        .def("get_timings", [](WhisperContextW& s) -> nb::object {
            s.ensure_valid();
            // allocated with new; the caller owns it
            std::unique_ptr<whisper_timings> t(whisper_get_timings(s.ctx));
            if (!t) return nb::none();
            nb::dict d;
            d["sample_ms"] = t->sample_ms;
            d["encode_ms"] = t->encode_ms;
            d["decode_ms"] = t->decode_ms;
            d["batchd_ms"] = t->batchd_ms;
            d["prompt_ms"] = t->prompt_ms;
            return d;
        }, "Mean ms per sample/encode/decode/batchd/prompt call on the default state.")
        .def("print_timings", [](WhisperContextW& s){ s.ensure_valid(); whisper_print_timings(s.ctx); })
        .def("reset_timings", [](WhisperContextW& s){ s.ensure_valid(); whisper_reset_timings(s.ctx); });
    bind_results(ctx_cls);
    bind_pipeline(ctx_cls);

    // -------------------------------------------------------------------------
    // WhisperState — keeps a reference to its parent context.
    // -------------------------------------------------------------------------
    nb::class_<WhisperStateW> state_cls(m, "WhisperState",
        "An inference state on a shared WhisperContext. Each state holds its own "
        "buffers and results, so states can run full() concurrently on separate "
        "threads; read results with the same full_* getters as WhisperContext.");
    state_cls
        .def("__init__",
             [](WhisperStateW* self, nb::object ctx) { new (self) WhisperStateW(std::move(ctx)); },
             "ctx"_a)
        .def("close", [](WhisperStateW& s){
            // a running full() holds the lock
            inferna::BusyGuard guard(s.busy_lock, WhisperStateW::kBusyMsg);
            s.free_state();
        }, "Free the state's buffers; it is unusable afterwards.")
        .def_prop_ro("is_valid", [](WhisperStateW& s){ return s.state != nullptr; })
        .def_prop_ro("ctx", [](WhisperStateW& s){ return s.parent; })
        .def_prop_ro("_busy_lock", [](WhisperStateW& s){ return s.busy_lock; })
        .def("full",
             [](WhisperStateW& s, Samples samples, std::optional<WhisperFullParamsW*> params) {
                 WhisperResult r = s.res();
                 s.n_decoded = 0;
                 run_full(s.busy_lock, WhisperStateW::kBusyMsg, params, [&](whisper_full_params p) {
                     return whisper_full_with_state(r.ctx, r.st, p, samples.data(), (int) samples.shape(0));
                 });
             },
             "samples"_a, "params"_a = nb::none());
    bind_results(state_cls);
    bind_pipeline(state_cls);

    // -------------------------------------------------------------------------
    // Standalone voice activity detection
    // -------------------------------------------------------------------------
    nb::class_<WhisperVadContextParamsW>(m, "WhisperVadContextParams")
        .def(nb::init<>())
        .def_prop_rw("n_threads",
            [](WhisperVadContextParamsW& s){ return s.c.n_threads; },
            [](WhisperVadContextParamsW& s, int v){ s.c.n_threads = v; })
        .def_prop_rw("use_gpu",
            [](WhisperVadContextParamsW& s){ return s.c.use_gpu; },
            [](WhisperVadContextParamsW& s, bool v){ s.c.use_gpu = v; })
        .def_prop_rw("gpu_device",
            [](WhisperVadContextParamsW& s){ return s.c.gpu_device; },
            [](WhisperVadContextParamsW& s, int v){ s.c.gpu_device = v; });

    nb::class_<WhisperVadContextW>(m, "WhisperVadContext",
        "Silero voice activity detection on 16 kHz mono samples, without a whisper model. "
        "detect_speech() computes one speech probability per window; segments_* turn "
        "probabilities into (t0, t1) speech segments in centiseconds.")
        .def(nb::init<const std::string&, std::optional<WhisperVadContextParamsW*>>(),
             "model_path"_a, "params"_a = nb::none())
        .def("close", [](WhisperVadContextW& s){
            // a running detect_speech() holds the lock
            inferna::BusyGuard guard(s.busy_lock, WhisperVadContextW::kBusyMsg);
            whisper_vad_free(s.ctx);
            s.ctx = nullptr;
        })
        .def_prop_ro("is_valid", [](WhisperVadContextW& s){ return s.ctx != nullptr; })
        .def_prop_ro("_busy_lock", [](WhisperVadContextW& s){ return s.busy_lock; })
        .def("detect_speech", [](WhisperVadContextW& s, Samples samples, bool reset) {
            whisper_vad_context* ctx = s.get();
            const float* x = samples.data();
            int n = (int) samples.shape(0);
            bool ok = run_locked(s, [&]{
                return reset ? whisper_vad_detect_speech(ctx, x, n) : whisper_vad_detect_speech_no_reset(ctx, x, n);
            });
            if (!ok) throw std::runtime_error("VAD speech detection failed");
        }, "samples"_a, "reset"_a = true,
           "Compute speech probabilities. reset=False keeps the LSTM state from the previous "
           "call, for streaming; call reset_state() between utterances.")
        .def("reset_state", [](WhisperVadContextW& s) {
            whisper_vad_context* ctx = s.get();
            run_locked(s, [&]{ whisper_vad_reset_state(ctx); return 0; });
        })
        .def("probs", [](WhisperVadContextW& s) {
            whisper_vad_context* ctx = s.get();
            size_t n = (size_t) whisper_vad_n_probs(ctx);
            float* out = new float[n];
            std::memcpy(out, whisper_vad_probs(ctx), n * sizeof(float));
            nb::capsule owner(out, [](void* p) noexcept { delete[] (float*) p; });
            return nb::ndarray<nb::numpy, float, nb::ndim<1>>(out, {n}, owner);
        }, "Speech probability per window from the last detect_speech().")
        .def("segments_from_probs", [](WhisperVadContextW& s, std::optional<WhisperVadParamsW*> params) {
            whisper_vad_context* ctx = s.get();
            whisper_vad_params p = vad_params_or_default(params);
            whisper_vad_segments* seg = nullptr;
            run_locked(s, [&]{ seg = whisper_vad_segments_from_probs(ctx, p); return 0; });
            return vad_segments_to_list(seg);
        }, "params"_a = nb::none(), "Speech segments [(t0, t1)] in centiseconds from the last detect_speech().")
        .def("segments_from_samples",
             [](WhisperVadContextW& s, Samples samples, std::optional<WhisperVadParamsW*> params) {
            whisper_vad_context* ctx = s.get();
            whisper_vad_params p = vad_params_or_default(params);
            whisper_vad_segments* seg = nullptr;
            run_locked(s, [&]{
                seg = whisper_vad_segments_from_samples(ctx, p, samples.data(), (int) samples.shape(0));
                return 0;
            });
            return vad_segments_to_list(seg);
        }, "samples"_a, "params"_a = nb::none(), "detect_speech() then segments_from_probs().");

    // -------------------------------------------------------------------------
    // Module-level functions
    // -------------------------------------------------------------------------
    m.def("ggml_backend_load_all", [](){
        inferna::load_all_backends("inferna.whisper._whisper_native");
    }, "Load all available ggml backends (CUDA, Metal, Vulkan, etc.).");

    m.def("disable_logging", [](){
        whisper_log_set(_whisper_no_log_cb, nullptr);
    }, "Suppress all C-level log output from whisper.cpp and ggml.");

    m.def("version",           [](){ return std::string(whisper_version()); },
          "whisper.cpp version string of the linked library.");
    m.def("print_system_info", [](){ return std::string(whisper_print_system_info()); },
          "Multi-line system / build / SIMD capability summary string.");
    m.def("lang_max_id",       [](){ return whisper_lang_max_id(); },
          "Largest valid integer language id known to whisper.");
    m.def("lang_id",  [](const std::string& s){ return whisper_lang_id(s.c_str()); }, "lang"_a,
          "Map an ISO-639 short language code (e.g. 'en') to its integer language id, or -1 if unknown.");
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
