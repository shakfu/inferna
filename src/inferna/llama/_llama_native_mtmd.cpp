// Multimodal (vision + audio) bindings for libmtmd. Registered into the
// _llama_native module via register_mtmd(). Underlying llama_model* /
// llama_context* are recovered from Python handles via the cross-TU
// helpers in _llama_native.hpp.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
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
};

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

    // -------------------------------------------------------------------------
    // MtmdContextParams
    // -------------------------------------------------------------------------
    nb::class_<MtmdContextParamsW>(m, "MtmdContextParams")
        .def("__init__",
             [](MtmdContextParamsW* self, bool use_gpu, bool print_timings,
                int n_threads, std::optional<std::string> media_marker,
                int flash_attn_type, int image_min_tokens, int image_max_tokens,
                bool warmup) {
                 new (self) MtmdContextParamsW();
                 self->p.use_gpu          = use_gpu;
                 self->p.print_timings    = print_timings;
                 self->p.n_threads        = n_threads;
                 self->p.flash_attn_type  = (llama_flash_attn_type) flash_attn_type;
                 self->p.image_min_tokens = image_min_tokens;
                 self->p.image_max_tokens = image_max_tokens;
                 self->p.warmup           = warmup;
                 if (media_marker) {
                     self->media_marker_owned = std::move(*media_marker);
                     self->p.media_marker = self->media_marker_owned->c_str();
                 }
             },
             "use_gpu"_a = true, "print_timings"_a = false, "n_threads"_a = 1,
             "media_marker"_a.none() = nb::none(),
             "flash_attn_type"_a = 0, "image_min_tokens"_a = -1,
             "image_max_tokens"_a = -1, "warmup"_a = true)
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
            [](MtmdContextParamsW& s, bool v){ s.p.warmup = v; });

    // -------------------------------------------------------------------------
    // MtmdBitmap
    // -------------------------------------------------------------------------
    nb::class_<MtmdBitmapW> bitmap_cls(m, "MtmdBitmap");
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
                MtmdContextW& ctx = nb::cast<MtmdContextW&>(ctx_obj);
                auto* w = new MtmdBitmapW();
                w->ptr = mtmd_helper_bitmap_init_from_file(ctx.ptr, file_path.c_str());
                if (!w->ptr) {
                    delete w;
                    throw std::runtime_error("Failed to load bitmap from file: " + file_path);
                }
                w->owner = true;
                return nb::cast(w, nb::rv_policy::take_ownership);
            }, "mtmd_ctx"_a.none(), "file_path"_a)
        .def_static("from_buffer",
            [](MtmdContextW& ctx, nb::bytes data){
                auto* w = new MtmdBitmapW();
                w->ptr = mtmd_helper_bitmap_init_from_buf(ctx.ptr,
                            (const unsigned char*) data.c_str(), data.size());
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
    nb::class_<MtmdInputChunkW>(m, "MtmdInputChunk")
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
    nb::class_<MtmdInputChunksW>(m, "MtmdInputChunks")
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
    nb::class_<MtmdContextW>(m, "MtmdContext")
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
        .def_prop_ro("supports_vision", [](MtmdContextW& s){
            return s.ptr ? (bool) mtmd_support_vision(s.ptr) : false;
        })
        .def_prop_ro("supports_audio", [](MtmdContextW& s){
            return s.ptr ? (bool) mtmd_support_audio(s.ptr) : false;
        })
        .def_prop_ro("audio_sample_rate", [](MtmdContextW& s){
            return s.ptr ? mtmd_get_audio_sample_rate(s.ptr) : -1;
        })
        .def_prop_ro("uses_non_causal", [](MtmdContextW& s){
            return s.ptr ? (bool) mtmd_decode_use_non_causal(s.ptr, nullptr) : false;
        })
        .def_prop_ro("uses_mrope", [](MtmdContextW& s){
            return s.ptr ? (bool) mtmd_decode_use_mrope(s.ptr) : false;
        })
        .def("tokenize",
            [](MtmdContextW& s, const std::string& text,
               std::vector<MtmdBitmapW*> bitmaps,
               bool add_special, bool parse_special){
                if (!s.ptr) throw std::runtime_error("Context not initialized");

                mtmd_input_text input_text{};
                input_text.text          = text.c_str();
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
            if (!s.ptr) throw std::runtime_error("Context not initialized");
            return mtmd_encode_chunk(s.ptr, chunk.ptr);
        })
        .def("get_output_embeddings",
            [](MtmdContextW& s, int n_tokens, int n_embd){
                if (!s.ptr) throw std::runtime_error("Context not initialized");
                float* embd = mtmd_get_output_embd(s.ptr);
                if (!embd) throw std::runtime_error("No embeddings available");
                nb::list out;
                for (int i = 0; i < n_tokens; ++i) {
                    nb::list row;
                    for (int j = 0; j < n_embd; ++j)
                        row.append(embd[i * n_embd + j]);
                    out.append(row);
                }
                return out;
            }, "n_tokens"_a, "n_embd"_a)
        .def("eval_chunks",
            [](MtmdContextW& s, nb::object llama_ctx,
               MtmdInputChunksW& chunks, int n_past, int seq_id,
               int32_t n_batch, bool logits_last){
                if (!s.ptr) throw std::runtime_error("Context not initialized");
                ::llama_context* ctx_ptr = inferna::unwrap_ctx(llama_ctx);
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
            "n_batch"_a = 32, "logits_last"_a = true);

    // -------------------------------------------------------------------------
    // Module-level
    // -------------------------------------------------------------------------
    m.def("get_default_media_marker", [](){
        return std::string(mtmd_default_marker());
    });
}
