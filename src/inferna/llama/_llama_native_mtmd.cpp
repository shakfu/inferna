// Multimodal (vision + audio) bindings for libmtmd. Registered into the
// _llama_native module via register_mtmd(). Underlying llama_model* /
// llama_context* are recovered from Python handles via the cross-TU
// helpers in _llama_native.hpp.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include "_llama_native.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// ---------------------------------------------------------------------------
// Wrappers
// ---------------------------------------------------------------------------

struct MtmdContextParamsW {
    mtmd_context_params p;
    std::optional<std::string> media_marker_owned;
    MtmdContextParamsW() : p(mtmd_context_params_default()) {}
};

struct MtmdBitmapW {
    mtmd_bitmap* ptr = nullptr;
    bool owner = false;
    ~MtmdBitmapW() {
        if (ptr && owner) { mtmd_bitmap_free(ptr); ptr = nullptr; }
    }
    MtmdBitmapW() = default;
    MtmdBitmapW(const MtmdBitmapW&) = delete;
    MtmdBitmapW& operator=(const MtmdBitmapW&) = delete;
};

struct MtmdInputChunkW {
    const mtmd_input_chunk* ptr = nullptr;
    bool owner = false;
    nb::object parent;  // keep parent MtmdInputChunks alive when non-owning
    ~MtmdInputChunkW() {
        if (ptr && owner) {
            mtmd_input_chunk_free(const_cast<mtmd_input_chunk*>(ptr));
            ptr = nullptr;
        }
    }
    MtmdInputChunkW() = default;
    MtmdInputChunkW(const MtmdInputChunkW&) = delete;
    MtmdInputChunkW& operator=(const MtmdInputChunkW&) = delete;
};

struct MtmdInputChunksW {
    mtmd_input_chunks* ptr = nullptr;
    bool owner = false;
    MtmdInputChunksW() {
        ptr = mtmd_input_chunks_init();
        if (!ptr) throw std::runtime_error("Failed to initialize input chunks");
        owner = true;
    }
    ~MtmdInputChunksW() {
        if (ptr && owner) { mtmd_input_chunks_free(ptr); ptr = nullptr; }
    }
    MtmdInputChunksW(const MtmdInputChunksW&) = delete;
    MtmdInputChunksW& operator=(const MtmdInputChunksW&) = delete;
};

struct MtmdContextW {
    mtmd_context* ptr = nullptr;
    nb::object model_obj;  // hold parent LlamaModel alive
    ~MtmdContextW() {
        if (ptr) { mtmd_free(ptr); ptr = nullptr; }
    }
    MtmdContextW() = default;
    MtmdContextW(const MtmdContextW&) = delete;
    MtmdContextW& operator=(const MtmdContextW&) = delete;

    // Throw a normal Python exception if the context has been closed,
    // instead of letting mtmd dereference a null pointer.
    void ensure_valid() const {
        if (!ptr) throw std::runtime_error(
            "MtmdContext has been closed and is no longer usable");
    }
};

// unwrap_ctx raises ValueError for a closed context; a closed parent is a state error here.
::llama_context* checked_llama_ctx(nb::handle h) {
    try {
        return inferna::unwrap_ctx(h);
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("LlamaContext is closed");
    }
}

// Batch of media chunks encoded in one projector pass. mtmd borrows the
// chunk pointers, so `chunk_refs` keeps each chunk's owner alive.
struct MtmdBatchW {
    mtmd_batch* ptr = nullptr;
    nb::object ctx_ref;
    std::vector<nb::object> chunk_refs;
    ~MtmdBatchW() { if (ptr) mtmd_batch_free(ptr); }
    MtmdBatchW() = default;
    MtmdBatchW(const MtmdBatchW&) = delete;
    MtmdBatchW& operator=(const MtmdBatchW&) = delete;
    void ensure_open() const {
        if (!ptr) throw std::runtime_error("Batch is closed");
        nb::cast<MtmdContextW&>(ctx_ref).ensure_valid();
    }
    void close() {
        if (ptr) { mtmd_batch_free(ptr); ptr = nullptr; }
        chunk_refs.clear();
        ctx_ref = nb::object();
    }
};

struct MtmdVideoW {
    mtmd_helper_video* ptr = nullptr;
    nb::object ctx_ref;
    ~MtmdVideoW() { if (ptr) mtmd_helper_video_free(ptr); }
    MtmdVideoW() = default;
    MtmdVideoW(const MtmdVideoW&) = delete;
    MtmdVideoW& operator=(const MtmdVideoW&) = delete;
    void ensure_open() const {
        if (!ptr) throw std::runtime_error("Video is closed");
        nb::cast<MtmdContextW&>(ctx_ref).ensure_valid();
    }
    void close() {
        if (ptr) { mtmd_helper_video_free(ptr); ptr = nullptr; }
        ctx_ref = nb::object();
    }
};

struct MtmdAudioGeneratorW {
    mtmd_helper_gen_audio* ptr = nullptr;
    ::llama_context* lctx = nullptr;
    const float* h_state = nullptr;  // backbone hidden state fed to the next step_gen()
    bool prompt_done = false;
    nb::object llama_ctx_ref;
    nb::object mtmd_ctx_ref;
    ~MtmdAudioGeneratorW() { if (ptr) mtmd_helper_gen_audio_free(ptr); }
    MtmdAudioGeneratorW() = default;
    MtmdAudioGeneratorW(const MtmdAudioGeneratorW&) = delete;
    MtmdAudioGeneratorW& operator=(const MtmdAudioGeneratorW&) = delete;
    // the parents' close() frees their native contexts whatever references remain
    void ensure_open() const {
        if (!ptr) throw std::runtime_error("Audio generator is closed");
        nb::cast<MtmdContextW&>(mtmd_ctx_ref).ensure_valid();
        checked_llama_ctx(llama_ctx_ref);
    }
    void close() {
        if (ptr) { mtmd_helper_gen_audio_free(ptr); ptr = nullptr; }
        h_state = nullptr;
        lctx = nullptr;
        llama_ctx_ref = nb::object();
        mtmd_ctx_ref = nb::object();
    }
};

mtmd_helper_video_init_params video_params(float fps_target, const std::optional<std::string>& ffmpeg_bin_dir,
                                           int64_t timestamp_interval_ms, bool no_soundtrack) {
    mtmd_helper_video_init_params params = mtmd_helper_video_init_params_default();
    if (fps_target > 0) params.fps_target = fps_target;
    if (timestamp_interval_ms > 0) params.timestamp_interval_ms = timestamp_interval_ms;
    if (ffmpeg_bin_dir) params.ffmpeg_bin_dir = ffmpeg_bin_dir->c_str();
    params.no_soundtrack = no_soundtrack;
    return params;
}

nb::object wrap_video(mtmd_helper_video* vid, nb::object ctx_obj, const std::string& what) {
    if (!vid) throw std::runtime_error(
        "Failed to open video" + what + ". Check that the data is a readable video and that "
        "ffmpeg/ffprobe are installed and on PATH (or pass ffmpeg_bin_dir).");
    auto* w = new MtmdVideoW();
    w->ptr = vid;
    w->ctx_ref = ctx_obj;
    return nb::cast(w, nb::rv_policy::take_ownership);
}

nb::object own_bitmap(mtmd_bitmap* bmp) {
    auto* w = new MtmdBitmapW();
    w->ptr = bmp;
    w->owner = true;
    return nb::cast(w, nb::rv_policy::take_ownership);
}

std::vector<std::string> voice_names(const mtmd_context* ctx) {
    std::vector<std::string> names;
    for (int32_t i = 0;; ++i) {
        const char* n = mtmd_gen_audio_voice_name(ctx, i);
        if (!n) break;
        names.emplace_back(n);
    }
    return names;
}

}  // namespace

void register_mtmd(nb::module_& m) {
    // -------------------------------------------------------------------------
    // Enum: MtmdInputChunkType
    // -------------------------------------------------------------------------
    nb::enum_<mtmd_input_chunk_type>(m, "MtmdInputChunkType")
        .value("TEXT",  MTMD_INPUT_CHUNK_TYPE_TEXT)
        .value("IMAGE", MTMD_INPUT_CHUNK_TYPE_IMAGE)
        .value("AUDIO", MTMD_INPUT_CHUNK_TYPE_AUDIO)
        .export_values();

    nb::enum_<mtmd_gen_audio_type>(m, "MtmdGenAudioType",
        "Audio generation pipeline exposed by an mmproj.")
        .value("NONE",       MTMD_GEN_AUDIO_TYPE_NONE)
        .value("QWEN3TTS",   MTMD_GEN_AUDIO_TYPE_QWEN3TTS)
        .value("POCKETTTS",  MTMD_GEN_AUDIO_TYPE_POCKETTTS)
        .value("SOPRANO",    MTMD_GEN_AUDIO_TYPE_SOPRANO)
        .value("LFM3A",      MTMD_GEN_AUDIO_TYPE_LFM3A)
        .value("CHATTERBOX", MTMD_GEN_AUDIO_TYPE_CHATTERBOX);

    nb::enum_<mtmd_helper_gen_audio_out_mode>(m, "MtmdGenAudioOutMode",
        "What a response that continues a sequence (set_input n_past > 0) outputs.")
        .value("AUTO",  MTMD_HELPER_GEN_AUDIO_OUT_AUTO)
        .value("TEXT",  MTMD_HELPER_GEN_AUDIO_OUT_TEXT)
        .value("AUDIO", MTMD_HELPER_GEN_AUDIO_OUT_AUDIO)
        .value("MIXED", MTMD_HELPER_GEN_AUDIO_OUT_MIXED);

    // -------------------------------------------------------------------------
    // MtmdContextParams
    // -------------------------------------------------------------------------
    nb::class_<MtmdContextParamsW>(m, "MtmdContextParams",
        "Parameters for an MtmdContext: GPU use, thread count, media marker, "
        "flash-attn type, image token budgets, warmup.")
        .def("__init__",
             [](MtmdContextParamsW* self, bool use_gpu, bool print_timings,
                int n_threads, std::optional<std::string> media_marker,
                int flash_attn_type, int image_min_tokens, int image_max_tokens,
                bool warmup, std::optional<int> batch_max_tokens) {
                 new (self) MtmdContextParamsW();
                 self->p.use_gpu          = use_gpu;
                 self->p.print_timings    = print_timings;
                 self->p.n_threads        = n_threads;
                 self->p.flash_attn_type  = (llama_flash_attn_type) flash_attn_type;
                 self->p.image_min_tokens = image_min_tokens;
                 self->p.image_max_tokens = image_max_tokens;
                 self->p.warmup           = warmup;
                 if (batch_max_tokens) self->p.batch_max_tokens = *batch_max_tokens;
                 if (media_marker) {
                     self->media_marker_owned = std::move(*media_marker);
                     self->p.media_marker = self->media_marker_owned->c_str();
                 }
             },
             "use_gpu"_a = true, "print_timings"_a = false, "n_threads"_a = 1,
             "media_marker"_a.none() = nb::none(),
             "flash_attn_type"_a = 0, "image_min_tokens"_a = -1,
             "image_max_tokens"_a = -1, "warmup"_a = true,
             "batch_max_tokens"_a.none() = nb::none())
        .def_prop_rw("use_gpu",
            [](MtmdContextParamsW& s){ return (bool) s.p.use_gpu; },
            [](MtmdContextParamsW& s, bool v){ s.p.use_gpu = v; })
        .def_prop_rw("print_timings",
            [](MtmdContextParamsW& s){ return (bool) s.p.print_timings; },
            [](MtmdContextParamsW& s, bool v){ s.p.print_timings = v; })
        .def_prop_rw("n_threads",
            [](MtmdContextParamsW& s){ return s.p.n_threads; },
            [](MtmdContextParamsW& s, int v){ s.p.n_threads = v; })
        .def_prop_rw("flash_attn_type",
            [](MtmdContextParamsW& s){ return (int) s.p.flash_attn_type; },
            [](MtmdContextParamsW& s, int v){ s.p.flash_attn_type = (llama_flash_attn_type) v; })
        .def_prop_rw("image_min_tokens",
            [](MtmdContextParamsW& s){ return s.p.image_min_tokens; },
            [](MtmdContextParamsW& s, int v){ s.p.image_min_tokens = v; })
        .def_prop_rw("image_max_tokens",
            [](MtmdContextParamsW& s){ return s.p.image_max_tokens; },
            [](MtmdContextParamsW& s, int v){ s.p.image_max_tokens = v; })
        .def_prop_rw("warmup",
            [](MtmdContextParamsW& s){ return (bool) s.p.warmup; },
            [](MtmdContextParamsW& s, bool v){ s.p.warmup = v; })
        .def_prop_rw("batch_max_tokens",
            [](MtmdContextParamsW& s){ return s.p.batch_max_tokens; },
            [](MtmdContextParamsW& s, int v){ s.p.batch_max_tokens = v; },
            "Soft cap on output tokens per MtmdBatch encode; the first chunk is always added.");

    // -------------------------------------------------------------------------
    // MtmdBitmap
    // -------------------------------------------------------------------------
    nb::class_<MtmdBitmapW> bitmap_cls(m, "MtmdBitmap",
        "Decoded image (HxWxRGB) or audio (1D float samples) buffer fed to MtmdContext.tokenize. "
        "Construct via create_image / create_audio / from_file / from_buffer.");
    bitmap_cls
        .def(nb::init<>())
        .def_static("create_image",
            [](int width, int height, nb::bytes data){
                if (width < 0 || height < 0) {
                    PyErr_SetString(PyExc_OverflowError,
                        "image width and height must be non-negative");
                    throw nb::python_error();
                }
                size_t expected = (size_t) width * height * 3;
                if (data.size() != expected)
                    throw std::invalid_argument(
                        "RGB image data must be width*height*3 bytes; got " +
                        std::to_string(data.size()) + " (expected " +
                        std::to_string(expected) + ")");
                auto* w = new MtmdBitmapW();
                w->ptr = mtmd_bitmap_init((uint32_t) width, (uint32_t) height,
                                          (const unsigned char*) data.c_str());
                if (!w->ptr) {
                    delete w;
                    throw std::runtime_error("Failed to create image bitmap");
                }
                w->owner = true;
                return nb::cast(w, nb::rv_policy::take_ownership);
            }, "width"_a, "height"_a, "data"_a)
        .def_static("create_audio",
            [](std::vector<float> samples){
                auto* w = new MtmdBitmapW();
                w->ptr = mtmd_bitmap_init_from_audio(samples.size(), samples.data());
                if (!w->ptr) {
                    delete w;
                    throw std::runtime_error("Failed to create audio bitmap");
                }
                w->owner = true;
                return nb::cast(w, nb::rv_policy::take_ownership);
            }, "samples"_a)
        .def_static("from_file",
            [](nb::object ctx_obj, const std::string& file_path){
                // File-existence check runs first so callers passing None or
                // a real ctx both get FileNotFoundError when the path is
                // missing, before the ctx-required code path.
                nb::module_ os = nb::module_::import_("os");
                if (!nb::cast<bool>(os.attr("path").attr("exists")(file_path))) {
                    PyErr_SetString(PyExc_FileNotFoundError,
                        ("File not found: " + file_path).c_str());
                    throw nb::python_error();
                }
                if (ctx_obj.is_none())
                    throw std::invalid_argument(
                        "mtmd_ctx must be an MtmdContext instance, got None");
                if (!nb::isinstance<MtmdContextW>(ctx_obj)) {
                    PyErr_SetString(PyExc_TypeError, "mtmd_ctx must be an MtmdContext");
                    throw nb::python_error();
                }
                MtmdContextW& ctx = nb::cast<MtmdContextW&>(ctx_obj);
                ctx.ensure_valid();
                auto* w = new MtmdBitmapW();
                w->ptr = mtmd_helper_bitmap_init_from_file(
                    ctx.ptr, file_path.c_str(), /*placeholder=*/false,
                    mtmd_helper_init_opt_default()).bitmap;
                if (!w->ptr) {
                    delete w;
                    throw std::runtime_error("Failed to load bitmap from file: " + file_path);
                }
                w->owner = true;
                return nb::cast(w, nb::rv_policy::take_ownership);
            }, "mtmd_ctx"_a.none(), "file_path"_a)
        .def_static("from_buffer",
            [](MtmdContextW& ctx, nb::bytes data){
                ctx.ensure_valid();
                auto* w = new MtmdBitmapW();
                w->ptr = mtmd_helper_bitmap_init_from_buf(ctx.ptr,
                            (const unsigned char*) data.c_str(), data.size(),
                            /*placeholder=*/false,
                            mtmd_helper_init_opt_default()).bitmap;
                if (!w->ptr) {
                    delete w;
                    throw std::runtime_error("Failed to load bitmap from buffer");
                }
                w->owner = true;
                return nb::cast(w, nb::rv_policy::take_ownership);
            }, "mtmd_ctx"_a, "data"_a)
        .def_prop_ro("width", [](MtmdBitmapW& s){
            if (!s.ptr) throw std::runtime_error("Bitmap not initialized");
            return mtmd_bitmap_get_nx(s.ptr);
        })
        .def_prop_ro("height", [](MtmdBitmapW& s){
            if (!s.ptr) throw std::runtime_error("Bitmap not initialized");
            return mtmd_bitmap_get_ny(s.ptr);
        })
        .def_prop_ro("data", [](MtmdBitmapW& s){
            if (!s.ptr) throw std::runtime_error("Bitmap not initialized");
            const unsigned char* d = mtmd_bitmap_get_data(s.ptr);
            size_t n = mtmd_bitmap_get_n_bytes(s.ptr);
            return nb::bytes((const char*) d, n);
        })
        .def_prop_ro("is_audio", [](MtmdBitmapW& s){
            if (!s.ptr) throw std::runtime_error("Bitmap not initialized");
            return (bool) mtmd_bitmap_is_audio(s.ptr);
        })
        .def_prop_ro("is_voice", [](MtmdBitmapW& s){
            if (!s.ptr) throw std::runtime_error("Bitmap not initialized");
            return (bool) mtmd_bitmap_is_voice(s.ptr);
        }, "True for a predefined-voice bitmap from MtmdContext.gen_audio_voice.")
        .def_prop_rw("id",
            [](MtmdBitmapW& s) -> nb::object {
                if (!s.ptr) throw std::runtime_error("Bitmap not initialized");
                const char* id = mtmd_bitmap_get_id(s.ptr);
                return id ? nb::cast(std::string(id)) : nb::none();
            },
            [](MtmdBitmapW& s, const std::string& v){
                if (!s.ptr) throw std::runtime_error("Bitmap not initialized");
                mtmd_bitmap_set_id(s.ptr, v.c_str());
            });

    // -------------------------------------------------------------------------
    // MtmdInputChunk — non-owning; tied to MtmdInputChunks lifetime.
    // -------------------------------------------------------------------------
    nb::class_<MtmdInputChunkW>(m, "MtmdInputChunk",
        "Single chunk produced by MtmdContext.tokenize: text, image, or audio segment "
        "with token/position counts. Non-owning view tied to its parent MtmdInputChunks.")
        .def(nb::init<>())
        .def_prop_ro("type", [](MtmdInputChunkW& s){
            if (!s.ptr) throw std::runtime_error("Chunk not initialized");
            return (mtmd_input_chunk_type) mtmd_input_chunk_get_type(s.ptr);
        })
        .def_prop_ro("n_tokens", [](MtmdInputChunkW& s){
            if (!s.ptr) throw std::runtime_error("Chunk not initialized");
            return mtmd_input_chunk_get_n_tokens(s.ptr);
        })
        .def_prop_ro("n_pos", [](MtmdInputChunkW& s){
            if (!s.ptr) throw std::runtime_error("Chunk not initialized");
            return mtmd_input_chunk_get_n_pos(s.ptr);
        })
        .def_prop_ro("id", [](MtmdInputChunkW& s) -> nb::object {
            if (!s.ptr) throw std::runtime_error("Chunk not initialized");
            const char* id = mtmd_input_chunk_get_id(s.ptr);
            return id ? nb::cast(std::string(id)) : nb::none();
        })
        .def("get_text_tokens", [](MtmdInputChunkW& s){
            if (!s.ptr) throw std::runtime_error("Chunk not initialized");
            if (mtmd_input_chunk_get_type(s.ptr) != MTMD_INPUT_CHUNK_TYPE_TEXT)
                throw std::invalid_argument("This is not a text chunk");
            size_t n = 0;
            const llama_token* toks = mtmd_input_chunk_get_tokens_text(s.ptr, &n);
            std::vector<int> out;
            out.reserve(n);
            if (toks) for (size_t i = 0; i < n; ++i) out.push_back(toks[i]);
            return out;
        });

    // -------------------------------------------------------------------------
    // MtmdInputChunks — owning container, indexable.
    // -------------------------------------------------------------------------
    nb::class_<MtmdInputChunksW>(m, "MtmdInputChunks",
        "Owning, indexable container of MtmdInputChunk produced by MtmdContext.tokenize. "
        "Pass to MtmdContext.eval_chunks for evaluation against a LlamaContext.")
        .def(nb::init<>())
        .def("__len__", [](MtmdInputChunksW& s){
            return s.ptr ? mtmd_input_chunks_size(s.ptr) : (size_t) 0;
        })
        .def("__getitem__",
            [](nb::object self_obj, int64_t idx){
                MtmdInputChunksW& s = nb::cast<MtmdInputChunksW&>(self_obj);
                if (!s.ptr) throw std::runtime_error("Chunks not initialized");
                int64_t size = (int64_t) mtmd_input_chunks_size(s.ptr);
                if (idx < 0) idx += size;
                if (idx < 0 || idx >= size)
                    throw nb::index_error(("Index out of range: " +
                                            std::to_string(idx)).c_str());
                auto* w = new MtmdInputChunkW();
                w->ptr = mtmd_input_chunks_get(s.ptr, (size_t) idx);
                w->owner = false;
                w->parent = self_obj;
                return nb::cast(w, nb::rv_policy::take_ownership);
            })
        .def_prop_ro("total_tokens", [](MtmdInputChunksW& s){
            return s.ptr ? mtmd_helper_get_n_tokens(s.ptr) : (size_t) 0;
        })
        .def_prop_ro("total_positions", [](MtmdInputChunksW& s){
            return s.ptr ? mtmd_helper_get_n_pos(s.ptr) : (llama_pos) 0;
        });

    // -------------------------------------------------------------------------
    // MtmdContext
    // -------------------------------------------------------------------------
    nb::class_<MtmdContextW>(m, "MtmdContext",
        "Multimodal projector context: tokenizes text + bitmaps into chunks and "
        "evaluates them into a LlamaContext. Construct from a .mmproj file plus a LlamaModel.")
        .def("__init__",
             [](MtmdContextW* self, const std::string& mmproj_path,
                nb::object llama_model, std::optional<MtmdContextParamsW*> params) {
                 nb::module_ os = nb::module_::import_("os");
                 if (!nb::cast<bool>(os.attr("path").attr("exists")(mmproj_path)))
                     throw std::invalid_argument(
                         "Multimodal projector file not found: " + mmproj_path);
                 new (self) MtmdContextW();
                 self->model_obj = llama_model;
                 ::llama_model* model_ptr = inferna::unwrap_model(llama_model);

                 MtmdContextParamsW default_params;
                 MtmdContextParamsW* p = params && *params ? *params : &default_params;
                 self->ptr = mtmd_init_from_file(mmproj_path.c_str(), model_ptr, p->p);
                 if (!self->ptr)
                     throw std::runtime_error(
                         "Failed to initialize mtmd context from: " + mmproj_path);
             },
             "mmproj_path"_a, "llama_model"_a, "params"_a = nb::none())
        .def("close", [](MtmdContextW& s){
            if (s.ptr) { mtmd_free(s.ptr); s.ptr = nullptr; }
        })
        .def_prop_ro("is_valid", [](MtmdContextW& s){ return s.ptr != nullptr; })
        .def_prop_ro("supports_vision", [](MtmdContextW& s){
            s.ensure_valid();
            return (bool) mtmd_support_vision(s.ptr);
        })
        .def_prop_ro("supports_audio", [](MtmdContextW& s){
            s.ensure_valid();
            return (bool) mtmd_support_audio(s.ptr);
        })
        .def_prop_ro("audio_sample_rate", [](MtmdContextW& s){
            s.ensure_valid();
            return mtmd_get_audio_sample_rate(s.ptr);
        })
        .def_prop_ro("uses_non_causal", [](MtmdContextW& s){
            s.ensure_valid();
            return (bool) mtmd_decode_use_non_causal(s.ptr, nullptr);
        })
        .def_prop_ro("uses_mrope", [](MtmdContextW& s){
            s.ensure_valid();
            return (bool) mtmd_decode_use_mrope(s.ptr);
        })
        .def_prop_ro("media_marker", [](MtmdContextW& s){
            s.ensure_valid();
            return std::string(mtmd_get_marker(s.ptr));
        }, "Marker that tokenize() replaces with each bitmap, as set in MtmdContextParams.")
        .def_prop_ro("capabilities", [](MtmdContextW& s){
            s.ensure_valid();
            mtmd_helper_cap c = mtmd_helper_get_capabilities(inferna::unwrap_model(s.model_obj), s.ptr);
            nb::dict d;
            d["chat"] = c.chat; d["vision"] = c.vision; d["audio"] = c.audio; d["video"] = c.video;
            d["audio_stream"] = c.audio_stream; d["text_gen"] = c.text_gen;
            d["audio_gen"] = c.audio_gen; d["audio_gen_spk_ref"] = c.audio_gen_spk_ref;
            return d;
        }, "Model/mmproj capabilities as a dict of bools. ``video`` is also False when "
           "the build lacks MTMD_VIDEO; ``audio_gen_spk_ref`` means voice cloning from a speaker reference.")
        .def("model_can_chat", [](MtmdContextW& s, nb::object llama_ctx){
            s.ensure_valid();
            return mtmd_helper_get_capabilities(llama_get_model(inferna::unwrap_ctx(llama_ctx)), s.ptr).chat;
        }, "llama_ctx"_a, "False for a TTS-only model: one that generates audio and has no chat template.")
        .def_prop_ro("gen_audio_info", [](MtmdContextW& s){
            s.ensure_valid();
            mtmd_gen_audio_info inf = mtmd_gen_audio_get_info(s.ptr);
            nb::dict d;
            d["type"] = inf.type;
            d["sample_rate"] = inf.sample_rate;
            d["model_variant"] = inf.model_variant ? nb::cast(std::string(inf.model_variant)) : nb::none();
            d["n_voices"] = inf.n_voices;
            return d;
        }, "Audio generation pipeline: ``type`` (MtmdGenAudioType), ``sample_rate``, "
           "``model_variant``, ``n_voices``.")
        .def("gen_audio_voices", [](MtmdContextW& s){
            s.ensure_valid();
            return voice_names(s.ptr);
        }, "Names of the predefined voices of the audio generation pipeline; empty if none.")
        .def("gen_audio_voice", [](MtmdContextW& s, const std::string& name){
            s.ensure_valid();
            mtmd_bitmap* bmp = mtmd_gen_audio_get_voice(s.ptr, name.c_str());
            if (!bmp) {
                std::string avail;
                for (const auto& n : voice_names(s.ptr)) avail += (avail.empty() ? "'" : ", '") + n + "'";
                throw std::invalid_argument("unknown voice '" + name + "'; available: [" + avail + "]");
            }
            return own_bitmap(bmp);
        }, "name"_a,
           "Predefined voice as a bitmap, to pass as ``speaker`` to MtmdAudioGenerator.set_input. "
           "Names match case-insensitively. Raises ValueError for an unknown name.")
        .def("batch", [](nb::object self_obj){
            MtmdContextW& s = nb::cast<MtmdContextW&>(self_obj);
            s.ensure_valid();
            auto* w = new MtmdBatchW();
            w->ptr = mtmd_batch_init(s.ptr);
            if (!w->ptr) { delete w; throw std::runtime_error("Failed to create mtmd batch"); }
            w->ctx_ref = self_obj;
            return nb::cast(w, nb::rv_policy::take_ownership);
        }, "Create an MtmdBatch for encoding several media chunks in one pass.")
        .def("open_video",
            [](nb::object self_obj, const std::string& path, float fps_target,
               std::optional<std::string> ffmpeg_bin_dir, int64_t timestamp_interval_ms, bool no_soundtrack){
                MtmdContextW& s = nb::cast<MtmdContextW&>(self_obj);
                s.ensure_valid();
                nb::module_ os = nb::module_::import_("os");
                if (!nb::cast<bool>(os.attr("path").attr("exists")(path))) {
                    PyErr_SetString(PyExc_FileNotFoundError, ("Video file not found: " + path).c_str());
                    throw nb::python_error();
                }
                auto params = video_params(fps_target, ffmpeg_bin_dir, timestamp_interval_ms, no_soundtrack);
                mtmd_helper_video* vid;
                {
                    nb::gil_scoped_release nogil;
                    vid = mtmd_helper_video_init(s.ptr, path.c_str(), params);
                }
                return wrap_video(vid, self_obj, ": " + path);
            },
            "path"_a, "fps_target"_a = 0.0f, "ffmpeg_bin_dir"_a.none() = nb::none(),
            "timestamp_interval_ms"_a = 0, "no_soundtrack"_a = false,
            "Open a video file as an MtmdVideo stream of frames and timestamp text. Needs ffmpeg "
            "and ffprobe. fps_target <= 0 uses the helper default (4 fps); timestamp_interval_ms "
            "<= 0 uses the helper default (5000 ms). no_soundtrack drops the audio that models such "
            "as LFM3-Omni otherwise receive after each frame group.")
        .def("open_video_buffer",
            [](nb::object self_obj, nb::bytes data, float fps_target,
               std::optional<std::string> ffmpeg_bin_dir, int64_t timestamp_interval_ms, bool no_soundtrack){
                MtmdContextW& s = nb::cast<MtmdContextW&>(self_obj);
                s.ensure_valid();
                if (data.size() == 0) throw std::invalid_argument("video data is empty");
                auto params = video_params(fps_target, ffmpeg_bin_dir, timestamp_interval_ms, no_soundtrack);
                mtmd_helper_video* vid;
                {
                    nb::gil_scoped_release nogil;  // mtmd copies the buffer; bytes are immutable
                    vid = mtmd_helper_video_init_from_buf(
                        s.ptr, (const unsigned char*) data.c_str(), data.size(), params);
                }
                return wrap_video(vid, self_obj, " from buffer");
            },
            "data"_a, "fps_target"_a = 0.0f, "ffmpeg_bin_dir"_a.none() = nb::none(),
            "timestamp_interval_ms"_a = 0, "no_soundtrack"_a = false,
            "As open_video, from in-memory video bytes. ffmpeg reads them from a pipe, so seeking "
            "decodes and discards frames.")
        .def("tokenize",
            [](MtmdContextW& s, const std::string& text,
               std::vector<MtmdBitmapW*> bitmaps,
               bool add_special, bool parse_special){
                s.ensure_valid();

                mtmd_input_text input_text{};
                input_text.text          = text.c_str();
                input_text.text_len      = text.size();
                input_text.add_special   = add_special;
                input_text.parse_special = parse_special;

                std::vector<const mtmd_bitmap*> ptrs;
                ptrs.reserve(bitmaps.size());
                for (MtmdBitmapW* b : bitmaps) {
                    if (!b || !b->ptr)
                        throw std::invalid_argument("Bitmap is null or freed");
                    ptrs.push_back(b->ptr);
                }

                auto* chunks = new MtmdInputChunksW();
                int32_t rc = mtmd_tokenize(s.ptr, chunks->ptr, &input_text,
                                           ptrs.empty() ? nullptr : ptrs.data(),
                                           ptrs.size());
                if (rc != 0) {
                    delete chunks;
                    if (rc == 1)
                        throw std::invalid_argument(
                            "Number of bitmaps does not match number of markers in text");
                    if (rc == 2)
                        throw std::runtime_error("Image preprocessing error");
                    throw std::runtime_error(
                        "Tokenization failed with error code: " + std::to_string(rc));
                }
                return nb::cast(chunks, nb::rv_policy::take_ownership);
            },
            "text"_a, "bitmaps"_a, "add_special"_a = true, "parse_special"_a = true)
        .def("encode_chunk", [](MtmdContextW& s, MtmdInputChunkW& chunk){
            s.ensure_valid();
            return mtmd_encode_chunk(s.ptr, chunk.ptr);
        })
        .def("get_output_embeddings",
            [](MtmdContextW& s, int n_tokens, int n_embd){
                // Flat row-major (n_tokens * n_embd) matches LlamaContext.get_logits.
                // Returning nb::list[nb::list[float]] would allocate one PyFloat per
                // element — 2.4M allocations for a typical CLIP projector.
                s.ensure_valid();
                float* embd = mtmd_get_output_embd(s.ptr);
                if (!embd) throw std::runtime_error("No embeddings available");
                size_t total = (size_t) n_tokens * (size_t) n_embd;
                return std::vector<float>(embd, embd + total);
            }, "n_tokens"_a, "n_embd"_a,
            "Output embeddings of the last encoded chunk, flat row-major. n_embd is the "
            "text model's input embedding width, LlamaModel.n_embd_inp.")
        .def("eval_chunks",
            [](MtmdContextW& s, nb::object llama_ctx,
               MtmdInputChunksW& chunks, int n_past, int seq_id,
               std::optional<int32_t> n_batch_opt, bool logits_last){
                s.ensure_valid();
                ::llama_context* ctx_ptr = inferna::unwrap_ctx(llama_ctx);
                int32_t n_batch = n_batch_opt ? *n_batch_opt : (int32_t) llama_n_batch(ctx_ptr);
                if (n_batch <= 0) throw std::invalid_argument("n_batch must be positive");
                // A non-causal media chunk (gemma3) must be decoded in one ubatch:
                // split, its tokens cannot attend to each other; over n_ubatch
                // llama.cpp aborts the process.
                size_t limit = std::min<size_t>((size_t) n_batch, llama_n_ubatch(ctx_ptr));
                for (size_t i = 0; i < mtmd_input_chunks_size(chunks.ptr); ++i) {
                    const mtmd_input_chunk* c = mtmd_input_chunks_get(chunks.ptr, i);
                    if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_TEXT) continue;
                    if (!mtmd_decode_use_non_causal(s.ptr, c)) continue;
                    size_t n = mtmd_input_chunk_get_n_tokens(c);
                    if (n > limit) throw std::invalid_argument(
                        "chunk " + std::to_string(i) + " needs " + std::to_string(n) +
                        " tokens in one batch because this model uses non-causal attention on "
                        "media (n_batch=" + std::to_string(n_batch) + ", n_ubatch=" +
                        std::to_string(llama_n_ubatch(ctx_ptr)) + "); raise both to at least " +
                        std::to_string(n) + ", or lower MtmdContextParams.image_max_tokens");
                }
                llama_pos new_n_past = 0;
                int32_t rc = mtmd_helper_eval_chunks(
                    s.ptr, ctx_ptr, chunks.ptr, (llama_pos) n_past,
                    (llama_seq_id) seq_id, n_batch, logits_last, &new_n_past);
                if (rc != 0)
                    throw std::runtime_error(
                        "Chunk evaluation failed with error code: " + std::to_string(rc));
                return new_n_past;
            },
            "llama_ctx"_a, "chunks"_a, "n_past"_a = 0, "seq_id"_a = 0,
            "n_batch"_a = nb::none(), "logits_last"_a = true,
            "Decode chunks into llama_ctx; n_batch defaults to the context's n_batch.");

    // -------------------------------------------------------------------------
    // MtmdBatch
    // -------------------------------------------------------------------------
    nb::class_<MtmdBatchW>(m, "MtmdBatch",
        "Encodes several media chunks in one mmproj pass. Create via MtmdContext.batch(). "
        "Keeps each added chunk's MtmdInputChunks alive until close().")
        .def("add_chunk", [](MtmdBatchW& s, nb::object chunk_obj){
            s.ensure_open();
            MtmdInputChunkW& chunk = nb::cast<MtmdInputChunkW&>(chunk_obj);
            if (!chunk.ptr) throw std::runtime_error("Chunk not initialized");
            int32_t rc = mtmd_batch_add_chunk(s.ptr, chunk.ptr);
            if (rc == 0) { s.chunk_refs.push_back(chunk_obj); return; }
            if (rc == 2) throw std::invalid_argument(
                "Chunk does not fit in the batch; encode the batch and start a new one");
            if (rc == 3) throw std::invalid_argument(
                "Chunk cannot be batched with the chunks already added "
                "(incompatible media type or geometry); use a separate batch");
            throw std::invalid_argument("Failed to add chunk to batch (error " + std::to_string(rc) +
                                        "); note that text chunks are not accepted");
        }, "chunk"_a, "Add a media chunk. Raises ValueError for a text chunk or one that does not fit.")
        .def("encode", [](MtmdBatchW& s){
            s.ensure_open();
            int32_t rc;
            {
                nb::gil_scoped_release nogil;
                rc = mtmd_batch_encode(s.ptr);
            }
            if (rc != 0) throw std::runtime_error("Batch encoding failed with error code: " + std::to_string(rc));
        }, "Run the projector over every chunk added so far.")
        .def("get_output_embd", [](MtmdBatchW& s, MtmdInputChunkW& chunk, int n_embd){
            s.ensure_open();
            if (!chunk.ptr) throw std::runtime_error("Chunk not initialized");
            if (n_embd <= 0) throw std::invalid_argument("n_embd must be positive, got " + std::to_string(n_embd));
            const float* embd = mtmd_batch_get_output_embd(s.ptr, chunk.ptr);
            if (!embd) throw std::runtime_error(
                "No output embeddings for this chunk; was it added to the batch and was encode() called?");
            size_t total = mtmd_input_chunk_get_n_tokens(chunk.ptr) * (size_t) n_embd;
            return std::vector<float>(embd, embd + total);
        }, "chunk"_a, "n_embd"_a,
           "Embeddings of ``chunk`` from the last encode(), flat row-major (n_tokens * n_embd), "
           "as MtmdContext.get_output_embeddings. n_embd is LlamaModel.n_embd_inp.")
        .def("close", [](MtmdBatchW& s){ s.close(); }, "Release the native batch. Idempotent.")
        .def("__enter__", [](nb::object self){ return self; })
        .def("__exit__", [](MtmdBatchW& s, nb::args){ s.close(); return false; });

    // -------------------------------------------------------------------------
    // MtmdVideo
    // -------------------------------------------------------------------------
    nb::class_<MtmdVideoW>(m, "MtmdVideo",
        "Video file decoded via ffmpeg into image bitmaps plus timestamp text. Create via "
        "MtmdContext.open_video(). Iterating yields ('image' or 'audio', MtmdBitmap) or ('text', str).")
        .def_prop_ro("info", [](MtmdVideoW& s){
            s.ensure_open();
            mtmd_helper_video_info inf = mtmd_helper_video_get_info(s.ptr);
            nb::dict d;
            d["width"] = inf.width; d["height"] = inf.height;
            d["fps"] = inf.fps; d["n_frames"] = inf.n_frames;
            return d;
        }, "``width``, ``height``, effective ``fps``, and estimated ``n_frames`` (-1 if unknown).")
        .def("read_next", [](MtmdVideoW& s) -> nb::object {
            s.ensure_open();
            mtmd_bitmap* bmp = nullptr;
            char* text = nullptr;
            int32_t rc;
            {
                nb::gil_scoped_release nogil;
                rc = mtmd_helper_video_read_next(s.ptr, &bmp, &text);
            }
            if (rc == -1) return nb::none();
            if (rc != 0) throw std::runtime_error("Video read failed with error code: " + std::to_string(rc));
            // LFM3-Omni gets each frame group's soundtrack as an audio bitmap
            if (bmp) return nb::make_tuple(mtmd_bitmap_is_audio(bmp) ? "audio" : "image", own_bitmap(bmp));
            if (text) {
                std::string t(text);
                free(text);  // allocated by strdup/malloc
                return nb::make_tuple("text", t);
            }
            return nb::none();
        }, "Next ``(kind, value)`` item, or None at end of stream.")
        .def("__iter__", [](nb::object self){ return self; })
        .def("__next__", [](nb::object self){
            nb::object item = self.attr("read_next")();
            if (item.is_none()) throw nb::stop_iteration();
            return item;
        })
        .def("close", [](MtmdVideoW& s){ s.close(); }, "Release the native video reader. Idempotent.")
        .def("__enter__", [](nb::object self){ return self; })
        .def("__exit__", [](MtmdVideoW& s, nb::args){ s.close(); return false; });

    // -------------------------------------------------------------------------
    // MtmdAudioGenerator
    // -------------------------------------------------------------------------
    nb::class_<MtmdAudioGeneratorW>(m, "MtmdAudioGenerator",
        "Audio generation (TTS) from a backbone LlamaContext and an mmproj. EXPERIMENTAL upstream. "
        "The caller samples a backbone token per frame and passes it to step_gen(). The "
        "LlamaContext must output embeddings (set_embeddings_mode(True)).")
        .def("__init__", [](MtmdAudioGeneratorW* self, nb::object llama_ctx, nb::object mtmd_ctx){
            MtmdContextW& m_ctx = nb::cast<MtmdContextW&>(mtmd_ctx);
            if (!m_ctx.ptr) throw std::runtime_error("MtmdContext is closed");
            if (mtmd_gen_audio_get_info(m_ctx.ptr).type == MTMD_GEN_AUDIO_TYPE_NONE)
                throw std::invalid_argument("mmproj does not support audio generation");
            ::llama_context* lctx = checked_llama_ctx(llama_ctx);
            new (self) MtmdAudioGeneratorW();
            self->ptr = mtmd_helper_gen_audio_init(lctx, m_ctx.ptr);
            if (!self->ptr) throw std::runtime_error("Failed to initialize audio generator");
            self->lctx = lctx;
            self->llama_ctx_ref = llama_ctx;
            self->mtmd_ctx_ref = mtmd_ctx;
        }, "llama_ctx"_a, "mtmd_ctx"_a)
        .def("set_input",
            [](MtmdAudioGeneratorW& s, const std::string& prompt, nb::object speaker_obj,
               std::optional<std::string> lang, int top_k, float top_p, uint32_t seed,
               bool wav, bool stream, int32_t n_past, mtmd_helper_gen_audio_out_mode out_mode){
                if (n_past < 0) throw std::invalid_argument("n_past must be >= 0");
                // upstream reads out_mode only when continuing a sequence
                if (n_past == 0 && out_mode != MTMD_HELPER_GEN_AUDIO_OUT_AUTO)
                    throw std::invalid_argument("out_mode requires n_past > 0");
                s.ensure_open();
                MtmdBitmapW* speaker = speaker_obj.is_none() ? nullptr : &nb::cast<MtmdBitmapW&>(speaker_obj);
                if (speaker && !speaker->ptr) throw std::invalid_argument("speaker bitmap is freed");
                // start from upstream defaults so fields added upstream are initialized
                mtmd_helper_gen_audio_inp inp = mtmd_helper_gen_audio_inp_default();
                inp.seq_id = 0;
                inp.prompt = prompt.c_str();
                inp.prompt_len = prompt.size();
                inp.speaker_ref = speaker ? speaker->ptr : nullptr;
                inp.lang = (lang && !lang->empty()) ? lang->c_str() : nullptr;
                inp.top_k = top_k;
                inp.top_p = top_p;
                inp.seed = seed;
                inp.out_type = wav ? MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV : MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;
                inp.stream = stream;
                inp.n_past = n_past;
                inp.out_mode = out_mode;
                s.h_state = nullptr;
                s.prompt_done = false;
                int32_t rc;
                {
                    nb::gil_scoped_release nogil;
                    rc = mtmd_helper_gen_audio_set_input(s.ptr, &inp);
                }
                if (rc != 0) throw std::runtime_error(
                    "Audio generator set_input failed with error code: " + std::to_string(rc) +
                    (n_past > 0 ? " (only LFM3-Audio can continue a sequence)" : ""));
            },
            "prompt"_a, "speaker"_a.none() = nb::none(), "lang"_a.none() = nb::none(),
            "top_k"_a = 0, "top_p"_a = 0.0f, "seed"_a = 0xFFFFFFFFu, "wav"_a = true, "stream"_a = false,
            "n_past"_a = 0, "out_mode"_a = MTMD_HELPER_GEN_AUDIO_OUT_AUTO,
            "Start a new utterance; discards any previous state. ``speaker`` is reference audio for "
            "voice cloning or a voice bitmap from MtmdContext.gen_audio_voice. top_k/top_p of 0 use "
            "the pipeline default; seed 0xFFFFFFFF is random. ``wav`` selects 16-bit WAV over float32 "
            "PCM. With ``stream``, each get_output() returns only new audio; call flush() after the "
            "last step_gen(). n_past > 0 continues a sequence of that many positions with ``prompt`` as raw "
            "text, and out_mode picks what the response outputs (LFM3-Audio only).")
        .def("step_prompt", [](MtmdAudioGeneratorW& s, int32_t n_batch){
            s.ensure_open();
            if (n_batch <= 0) throw std::invalid_argument("n_batch must be positive");
            int32_t rc;
            {
                nb::gil_scoped_release nogil;
                rc = mtmd_helper_gen_audio_step_prompt(s.ptr, n_batch);
            }
            if (rc < 0) throw std::runtime_error(
                "Audio generator prompt processing failed with error code: " + std::to_string(rc));
            if (rc == 0) s.prompt_done = true;
            return rc;
        }, "n_batch"_a, "Decode up to n_batch prompt tokens. Returns the number remaining.")
        .def("step_gen", [](MtmdAudioGeneratorW& s, int sampled){
            s.ensure_open();
            if (!s.prompt_done) throw std::runtime_error("step_prompt() must return 0 before step_gen()");
            if (!s.h_state) {
                s.h_state = llama_get_embeddings_ith(s.lctx, -1);
                if (!s.h_state) throw std::runtime_error(
                    "No backbone embeddings; enable embeddings on the LlamaContext");
            }
            const float* h_next = nullptr;
            bool stop = false;
            int32_t rc;
            {
                nb::gil_scoped_release nogil;
                rc = mtmd_helper_gen_audio_step_gen(s.ptr, (llama_token) sampled, s.h_state, &h_next, &stop);
            }
            if (rc != 0) throw std::runtime_error(
                "Audio generator step_gen failed with error code: " + std::to_string(rc));
            s.h_state = h_next;
            return !stop && h_next != nullptr;
        }, "sampled"_a, "Generate one frame from a backbone token. Returns False when generation ends.")
        .def("next_is_text", [](MtmdAudioGeneratorW& s){
            s.ensure_open();
            return (bool) mtmd_helper_gen_audio_next_is_text(s.ptr);
        }, "True if the next step_gen() takes a text token: sample it from the logits, then pass it. "
           "False if the step generates an audio frame (LFM3-Audio continuation).")
        .def("get_logit_bias", [](MtmdAudioGeneratorW& s){
            s.ensure_open();
            size_t n = 0;
            const llama_logit_bias* b = mtmd_helper_gen_audio_get_logit_bias(s.ptr, &n);
            nb::list out;
            for (size_t i = 0; i < n; ++i) out.append(nb::make_tuple(b[i].token, b[i].bias));
            return out;
        }, "``(token, bias)`` pairs to apply when sampling text tokens, in the format of "
           "LlamaSampler.add_logit_bias; set by set_input.")
        .def("get_output", [](MtmdAudioGeneratorW& s){
            s.ensure_open();
            int32_t sample_rate = 0;
            const char* data = nullptr;
            size_t len = 0;
            int32_t rc;
            {
                nb::gil_scoped_release nogil;
                rc = mtmd_helper_gen_audio_get_output(s.ptr, &sample_rate, &data, &len, nullptr);
            }
            if (rc != 0) throw std::runtime_error(
                "Audio generator get_output failed with error code: " + std::to_string(rc));
            return nb::bytes(len > 0 ? data : "", len);
        }, "Vocode the generated frames into WAV or float32 PCM bytes, per set_input(). "
           "In stream mode, returns only new audio.")
        .def("flush", [](MtmdAudioGeneratorW& s){
            s.ensure_open();
            int32_t rc;
            {
                nb::gil_scoped_release nogil;
                rc = mtmd_helper_gen_audio_flush(s.ptr);
            }
            if (rc != 0) throw std::runtime_error(
                "Audio generator flush failed with error code: " + std::to_string(rc));
        }, "Vocode buffered frames now. In stream mode, call before the last get_output().")
        .def("reset", [](MtmdAudioGeneratorW& s){
            s.ensure_open();
            mtmd_helper_gen_audio_reset(s.ptr);
            s.h_state = nullptr;
            s.prompt_done = false;
        }, "Discard the current utterance.")
        .def("close", [](MtmdAudioGeneratorW& s){ s.close(); }, "Release the native generator. Idempotent.")
        .def("__enter__", [](nb::object self){ return self; })
        .def("__exit__", [](MtmdAudioGeneratorW& s, nb::args){ s.close(); return false; });

    // -------------------------------------------------------------------------
    // Module-level
    // -------------------------------------------------------------------------
    m.def("get_default_media_marker", [](){
        return std::string(mtmd_default_marker());
    }, "The default placeholder string (e.g. '<__media__>') marking media positions in input text.");

    m.def("get_mmproj_caps", [](const std::string& mmproj_path){
        nb::module_ os = nb::module_::import_("os");
        if (!nb::cast<bool>(os.attr("path").attr("exists")(mmproj_path))) {
            PyErr_SetString(PyExc_FileNotFoundError,
                ("Multimodal projector file not found: " + mmproj_path).c_str());
            throw nb::python_error();
        }
        mtmd_caps c = mtmd_get_cap_from_file(mmproj_path.c_str());
        nb::dict d;
        d["vision"] = c.inp_vision; d["audio"] = c.inp_audio; d["gen_audio"] = c.gen_audio;
        return d;
    }, "mmproj_path"_a,
       "An mmproj file's ``vision`` and ``audio`` inputs and ``gen_audio`` (TTS) output, "
       "without loading a context.");
}
