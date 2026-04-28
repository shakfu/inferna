// nanobind bindings for stable-diffusion.cpp. Produces the `_sd_native`
// extension; the public Python surface lives in `inferna.sd.stable_diffusion`,
// which wraps these raw bindings and adds:
//   - IntEnum classes for the enum types
//   - SDImage helper methods (PPM/BMP I/O, PIL/numpy conversion)
//   - SDContext.generate(...) convenience method
//   - text_to_image / text_to_images / image_to_image helpers
//
// Names exported here correspond to the Python wrapper's `_n.<name>`.
// Renaming is fine but the wrapper module must be updated in lock-step.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "stable-diffusion.h"
#include "stb_image.h"
#include "stb_image_write.h"

#include "common/busy_lock.hpp"

// Forward-declare ggml-backend to avoid header conflicts (same workaround
// used in the whisper module).
extern "C" {
    struct ggml_backend_reg;
    typedef ggml_backend_reg* ggml_backend_reg_t;
    ggml_backend_reg_t ggml_backend_load(const char* path);
    void ggml_backend_load_all(void);
    void ggml_backend_load_all_from_path(const char* dir_path);
}

namespace nb = nanobind;
using namespace nb::literals;

// =============================================================================
// SDImage
// =============================================================================

struct SDImageW {
    sd_image_t img{};
    bool owns = false;

    SDImageW() = default;
    SDImageW(const SDImageW&) = delete;
    SDImageW& operator=(const SDImageW&) = delete;
    SDImageW(SDImageW&& other) noexcept : img(other.img), owns(other.owns) {
        other.img = sd_image_t{};
        other.owns = false;
    }
    ~SDImageW() {
        if (owns && img.data) {
            std::free(img.data);
            img.data = nullptr;
        }
    }
};

// =============================================================================
// SDContextParams
// =============================================================================

struct SDContextParamsW {
    sd_ctx_params_t p{};
    // Owning storage for char* fields. Must outlive any new_sd_ctx() call.
    std::optional<std::string> model_path_s;
    std::optional<std::string> clip_l_path_s;
    std::optional<std::string> clip_g_path_s;
    std::optional<std::string> clip_vision_path_s;
    std::optional<std::string> t5xxl_path_s;
    std::optional<std::string> llm_path_s;
    std::optional<std::string> llm_vision_path_s;
    std::optional<std::string> diffusion_model_path_s;
    std::optional<std::string> high_noise_diffusion_model_path_s;
    std::optional<std::string> vae_path_s;
    std::optional<std::string> taesd_path_s;
    std::optional<std::string> control_net_path_s;
    std::optional<std::string> photo_maker_path_s;
    std::optional<std::string> tensor_type_rules_s;

    SDContextParamsW() {
        sd_ctx_params_init(&p);
    }
};

// Macro for the optional<string>-backed path properties — saves ~250 lines of
// boilerplate. Example: SD_PARAM_PATH(SDContextParamsW, p.model_path, model_path_s, "model_path")
#define SD_PARAM_PATH(WrapperT, FIELD, OWNED_FIELD, NAME)                          \
    .def_prop_rw(NAME,                                                             \
        [](WrapperT& s) -> nb::object {                                            \
            if (!s.FIELD) return nb::none();                                       \
            return nb::cast(std::string(s.FIELD));                                 \
        },                                                                         \
        [](WrapperT& s, std::optional<std::string> v) {                            \
            if (!v || v->empty()) { s.FIELD = nullptr; s.OWNED_FIELD.reset(); }    \
            else { s.OWNED_FIELD = std::move(*v); s.FIELD = s.OWNED_FIELD->c_str(); } \
        })

// Macro for plain primitive struct fields (POD types — int, float, bool, enums-as-int).
#define SD_PARAM_VAL(WrapperT, T, FIELD, NAME)                          \
    .def_prop_rw(NAME,                                                  \
        [](WrapperT& s) { return (T) s.FIELD; },                        \
        [](WrapperT& s, T v) { s.FIELD = (decltype(s.FIELD)) v; })

// =============================================================================
// SDSampleParams
// =============================================================================

struct SDSampleParamsW {
    sd_sample_params_t p{};
    std::vector<int> slg_layers_owned;
    std::vector<float> custom_sigmas_owned;

    SDSampleParamsW() {
        sd_sample_params_init(&p);
    }
};

// =============================================================================
// SDImageGenParams
// =============================================================================

struct SDImageGenParamsW {
    sd_img_gen_params_t p{};
    SDSampleParamsW sample;  // mirror of sample params; synced on demand
    std::optional<std::string> prompt_s;
    std::optional<std::string> negative_prompt_s;
    std::optional<std::string> pm_id_embed_path_s;
    std::optional<std::string> scm_mask_s;
    std::optional<std::string> hires_model_path_s;

    // Owning storage for arrays referenced by the C struct.
    std::vector<sd_image_t> ref_images_buf;
    std::vector<sd_image_t> pm_id_images_buf;
    std::vector<sd_lora_t>  loras_buf;
    std::vector<std::string> lora_paths_owned;  // backs sd_lora_t::path

    // Keep refs to SDImage instances alive (so their data buffers persist).
    nb::object init_image_ref;
    nb::object mask_image_ref;
    nb::object control_image_ref;
    nb::object ref_images_pyref;
    nb::object pm_id_images_pyref;

    SDImageGenParamsW() {
        sd_img_gen_params_init(&p);
        // sd_img_gen_params_init() doesn't touch the hires substruct — it
        // leaves upscaler=NONE, but callers expect LATENT as the documented
        // default (also asserted by tests/test_sd.py::test_hires_defaults).
        // sd_hires_params_init() exists in the header but isn't exported by
        // libstable-diffusion.a, so set the documented defaults inline.
        p.hires.upscaler          = SD_HIRES_UPSCALER_LATENT;
        p.hires.scale             = 2.0f;
        p.hires.denoising_strength = 0.7f;
        p.hires.upscale_tile_size = 128;
        // Inherit the default sample params we just constructed.
        p.sample_params = sample.p;
    }

    void sync_sample() {
        // Caller must invoke before any native call that reads p.sample_params.
        p.sample_params = sample.p;
    }
};

// =============================================================================
// SDContext
// =============================================================================

struct SDContextW {
    sd_ctx_t* ctx = nullptr;
    nb::object busy_lock;

    explicit SDContextW(SDContextParamsW& params) {
        // Validation lives Python-side (inferna.utils.validation) for a
        // consistent typed-error matrix across the project.
        nb::module_ validation = nb::module_::import_("inferna.utils.validation");
        struct PathSpec { const char* attr; const char* label; };
        PathSpec specs[] = {
            {"model_path", "Stable Diffusion model"},
            {"diffusion_model_path", "diffusion model"},
            {"vae_path", "VAE"},
            {"clip_l_path", "CLIP-L"},
            {"clip_g_path", "CLIP-G"},
            {"t5xxl_path", "T5-XXL"},
            {"control_net_path", "ControlNet"},
            {"taesd_path", "TAESD"},
            {"photo_maker_path", "PhotoMaker"},
        };
        const std::optional<std::string>* values[] = {
            &params.model_path_s, &params.diffusion_model_path_s, &params.vae_path_s,
            &params.clip_l_path_s, &params.clip_g_path_s, &params.t5xxl_path_s,
            &params.control_net_path_s, &params.taesd_path_s, &params.photo_maker_path_s
        };
        for (size_t i = 0; i < sizeof(specs)/sizeof(specs[0]); ++i) {
            if (values[i]->has_value()) {
                validation.attr("validate_model_file")(
                    **values[i], "kind"_a = specs[i].label);
            }
        }

        ctx = new_sd_ctx(&params.p);
        if (!ctx) {
            throw std::runtime_error(
                "Failed to create Stable Diffusion context. "
                "All configured paths exist and are readable, but stable-diffusion.cpp "
                "could not initialize. Possible causes: unsupported model format or "
                "version, mismatched companion files (VAE/CLIP/T5), insufficient memory, "
                "or invalid weight type. Check stderr for details from stable-diffusion.cpp.");
        }
        nb::module_ threading = nb::module_::import_("threading");
        busy_lock = threading.attr("Lock")();
    }

    ~SDContextW() { if (ctx) { free_sd_ctx(ctx); ctx = nullptr; } }
    SDContextW(const SDContextW&) = delete;
    SDContextW& operator=(const SDContextW&) = delete;

    static constexpr const char* kBusyMsg =
        "SDContext is currently being used by another thread. "
        "stable-diffusion.cpp contexts are not thread-safe -- "
        "create one SDContext per thread instead of sharing a "
        "single instance across threads.";

    void try_acquire_busy() {
        nb::object acquired = busy_lock.attr("acquire")("blocking"_a = false);
        if (!nb::cast<bool>(acquired)) {
            throw std::runtime_error(kBusyMsg);
        }
    }
    void release_busy() { busy_lock.attr("release")(); }
};

// =============================================================================
// Upscaler
// =============================================================================

struct UpscalerW {
    upscaler_ctx_t* ctx = nullptr;
    UpscalerW(const std::string& model_path, bool offload_to_cpu, bool direct,
              int n_threads, int tile_size) {
        if (n_threads < 0) n_threads = sd_get_num_physical_cores();
        ctx = new_upscaler_ctx(model_path.c_str(), offload_to_cpu, direct,
                                n_threads, tile_size);
        if (!ctx) {
            throw std::runtime_error("Failed to load upscaler model: " + model_path);
        }
    }
    ~UpscalerW() { if (ctx) { free_upscaler_ctx(ctx); ctx = nullptr; } }
    UpscalerW(const UpscalerW&) = delete;
    UpscalerW& operator=(const UpscalerW&) = delete;
};

// =============================================================================
// Module-level callback storage. Held as nb::object to keep the GC root.
// =============================================================================

static nb::object g_log_cb;
static nb::object g_progress_cb;
static nb::object g_preview_cb;

// Acquire the GIL *before* the is_valid/is_none check on each global cb —
// the worker thread cannot race with set_*_callback (which mutates the
// nb::object under the GIL) once the GIL is held.
extern "C" void _c_log_cb(sd_log_level_t level, const char* text, void* /*data*/) {
    nb::gil_scoped_acquire gil;
    if (!g_log_cb.is_valid() || g_log_cb.is_none()) return;
    try {
        g_log_cb((int)level, text ? std::string(text) : std::string());
    } catch (...) {
        // Swallow exceptions in the C log callback — never let them
        // propagate back into native code from a Python handler.
    }
}
extern "C" void _c_progress_cb(int step, int steps, float time, void* /*data*/) {
    nb::gil_scoped_acquire gil;
    if (!g_progress_cb.is_valid() || g_progress_cb.is_none()) return;
    try { g_progress_cb(step, steps, time); }
    catch (...) {}
}
extern "C" void _c_preview_cb(int step, int frame_count, sd_image_t* frames,
                              bool is_noisy, void* /*data*/) {
    nb::gil_scoped_acquire gil;
    if (!g_preview_cb.is_valid() || g_preview_cb.is_none()) return;
    try {
        nb::list py_frames;
        for (int i = 0; i < frame_count; ++i) {
            auto* img = new SDImageW{};
            img->img = frames[i];
            img->owns = false;  // preview only — caller owns the buffer
            py_frames.append(nb::cast(img, nb::rv_policy::take_ownership));
        }
        g_preview_cb(step, py_frames, is_noisy);
    } catch (...) {}
}

// =============================================================================
// Helpers for module-init (enum int constants). Placed here for visibility.
// =============================================================================

static nb::dict make_enum_dict() {
    // Returned dict is read by the Python module at import time to populate
    // its IntEnum classes. Single source of truth for enum values keeps
    // Python and C++ from drifting if upstream stable-diffusion.h ever
    // re-orders an enum.
    nb::dict d;
    auto add = [&](const char* name, int val) { d[name] = val; };
    add("STD_DEFAULT_RNG", STD_DEFAULT_RNG);
    add("CUDA_RNG", CUDA_RNG);
    add("CPU_RNG", CPU_RNG);
    add("RNG_TYPE_COUNT", RNG_TYPE_COUNT);

    add("EULER_SAMPLE_METHOD", EULER_SAMPLE_METHOD);
    add("EULER_A_SAMPLE_METHOD", EULER_A_SAMPLE_METHOD);
    add("HEUN_SAMPLE_METHOD", HEUN_SAMPLE_METHOD);
    add("DPM2_SAMPLE_METHOD", DPM2_SAMPLE_METHOD);
    add("DPMPP2S_A_SAMPLE_METHOD", DPMPP2S_A_SAMPLE_METHOD);
    add("DPMPP2M_SAMPLE_METHOD", DPMPP2M_SAMPLE_METHOD);
    add("DPMPP2Mv2_SAMPLE_METHOD", DPMPP2Mv2_SAMPLE_METHOD);
    add("IPNDM_SAMPLE_METHOD", IPNDM_SAMPLE_METHOD);
    add("IPNDM_V_SAMPLE_METHOD", IPNDM_V_SAMPLE_METHOD);
    add("LCM_SAMPLE_METHOD", LCM_SAMPLE_METHOD);
    add("DDIM_TRAILING_SAMPLE_METHOD", DDIM_TRAILING_SAMPLE_METHOD);
    add("TCD_SAMPLE_METHOD", TCD_SAMPLE_METHOD);
    add("RES_MULTISTEP_SAMPLE_METHOD", RES_MULTISTEP_SAMPLE_METHOD);
    add("RES_2S_SAMPLE_METHOD", RES_2S_SAMPLE_METHOD);
    add("ER_SDE_SAMPLE_METHOD", ER_SDE_SAMPLE_METHOD);
    add("SAMPLE_METHOD_COUNT", SAMPLE_METHOD_COUNT);

    add("DISCRETE_SCHEDULER", DISCRETE_SCHEDULER);
    add("KARRAS_SCHEDULER", KARRAS_SCHEDULER);
    add("EXPONENTIAL_SCHEDULER", EXPONENTIAL_SCHEDULER);
    add("AYS_SCHEDULER", AYS_SCHEDULER);
    add("GITS_SCHEDULER", GITS_SCHEDULER);
    add("SGM_UNIFORM_SCHEDULER", SGM_UNIFORM_SCHEDULER);
    add("SIMPLE_SCHEDULER", SIMPLE_SCHEDULER);
    add("SMOOTHSTEP_SCHEDULER", SMOOTHSTEP_SCHEDULER);
    add("KL_OPTIMAL_SCHEDULER", KL_OPTIMAL_SCHEDULER);
    add("LCM_SCHEDULER", LCM_SCHEDULER);
    add("BONG_TANGENT_SCHEDULER", BONG_TANGENT_SCHEDULER);
    add("SCHEDULER_COUNT", SCHEDULER_COUNT);

    add("EPS_PRED", EPS_PRED);
    add("V_PRED", V_PRED);
    add("EDM_V_PRED", EDM_V_PRED);
    add("FLOW_PRED", FLOW_PRED);
    add("FLUX_FLOW_PRED", FLUX_FLOW_PRED);
    add("FLUX2_FLOW_PRED", FLUX2_FLOW_PRED);
    add("PREDICTION_COUNT", PREDICTION_COUNT);

    add("SD_TYPE_F32", SD_TYPE_F32);
    add("SD_TYPE_F16", SD_TYPE_F16);
    add("SD_TYPE_Q4_0", SD_TYPE_Q4_0);
    add("SD_TYPE_Q4_1", SD_TYPE_Q4_1);
    add("SD_TYPE_Q5_0", SD_TYPE_Q5_0);
    add("SD_TYPE_Q5_1", SD_TYPE_Q5_1);
    add("SD_TYPE_Q8_0", SD_TYPE_Q8_0);
    add("SD_TYPE_Q8_1", SD_TYPE_Q8_1);
    add("SD_TYPE_Q2_K", SD_TYPE_Q2_K);
    add("SD_TYPE_Q3_K", SD_TYPE_Q3_K);
    add("SD_TYPE_Q4_K", SD_TYPE_Q4_K);
    add("SD_TYPE_Q5_K", SD_TYPE_Q5_K);
    add("SD_TYPE_Q6_K", SD_TYPE_Q6_K);
    add("SD_TYPE_Q8_K", SD_TYPE_Q8_K);
    add("SD_TYPE_BF16", SD_TYPE_BF16);
    add("SD_TYPE_COUNT", SD_TYPE_COUNT);

    add("SD_LOG_DEBUG", SD_LOG_DEBUG);
    add("SD_LOG_INFO", SD_LOG_INFO);
    add("SD_LOG_WARN", SD_LOG_WARN);
    add("SD_LOG_ERROR", SD_LOG_ERROR);

    add("PREVIEW_NONE", PREVIEW_NONE);
    add("PREVIEW_PROJ", PREVIEW_PROJ);
    add("PREVIEW_TAE", PREVIEW_TAE);
    add("PREVIEW_VAE", PREVIEW_VAE);

    add("LORA_APPLY_AUTO", LORA_APPLY_AUTO);
    add("LORA_APPLY_IMMEDIATELY", LORA_APPLY_IMMEDIATELY);
    add("LORA_APPLY_AT_RUNTIME", LORA_APPLY_AT_RUNTIME);

    add("SD_HIRES_UPSCALER_NONE", SD_HIRES_UPSCALER_NONE);
    add("SD_HIRES_UPSCALER_LATENT", SD_HIRES_UPSCALER_LATENT);
    add("SD_HIRES_UPSCALER_LATENT_NEAREST", SD_HIRES_UPSCALER_LATENT_NEAREST);
    add("SD_HIRES_UPSCALER_LATENT_NEAREST_EXACT", SD_HIRES_UPSCALER_LATENT_NEAREST_EXACT);
    add("SD_HIRES_UPSCALER_LATENT_ANTIALIASED", SD_HIRES_UPSCALER_LATENT_ANTIALIASED);
    add("SD_HIRES_UPSCALER_LATENT_BICUBIC", SD_HIRES_UPSCALER_LATENT_BICUBIC);
    add("SD_HIRES_UPSCALER_LATENT_BICUBIC_ANTIALIASED", SD_HIRES_UPSCALER_LATENT_BICUBIC_ANTIALIASED);
    add("SD_HIRES_UPSCALER_LANCZOS", SD_HIRES_UPSCALER_LANCZOS);
    add("SD_HIRES_UPSCALER_NEAREST", SD_HIRES_UPSCALER_NEAREST);
    add("SD_HIRES_UPSCALER_MODEL", SD_HIRES_UPSCALER_MODEL);

    add("SD_CACHE_DISABLED", SD_CACHE_DISABLED);
    add("SD_CACHE_EASYCACHE", SD_CACHE_EASYCACHE);
    add("SD_CACHE_UCACHE", SD_CACHE_UCACHE);
    add("SD_CACHE_DBCACHE", SD_CACHE_DBCACHE);
    add("SD_CACHE_TAYLORSEER", SD_CACHE_TAYLORSEER);
    add("SD_CACHE_CACHE_DIT", SD_CACHE_CACHE_DIT);

    return d;
}

// =============================================================================
// NB_MODULE
// =============================================================================

NB_MODULE(_sd_native, m) {
    m.attr("ENUMS") = make_enum_dict();

    // -------------------------------------------------------------------------
    // SDImage
    // -------------------------------------------------------------------------
    nb::class_<SDImageW>(m, "SDImage")
        .def(nb::init<>())
        .def_prop_ro("width",    [](SDImageW& s){ return s.img.width; })
        .def_prop_ro("height",   [](SDImageW& s){ return s.img.height; })
        .def_prop_ro("channels", [](SDImageW& s){ return s.img.channel; })
        .def_prop_ro("is_valid", [](SDImageW& s){
            return s.img.data != nullptr && s.img.width > 0 && s.img.height > 0;
        })
        .def("to_numpy", [](SDImageW& s) -> nb::object {
            if (!s.img.data || s.img.width == 0 || s.img.height == 0)
                throw std::runtime_error("Image has no valid data");
            // Produce a (H, W, C) uint8 ndarray that owns its own copy.
            size_t sz = (size_t)s.img.width * s.img.height * s.img.channel;
            uint8_t* buf = new uint8_t[sz];
            std::memcpy(buf, s.img.data, sz);
            size_t shape[3] = {s.img.height, s.img.width, s.img.channel};
            // The owner capsule deletes buf when the ndarray is gc'd.
            nb::capsule deleter(buf, [](void* p) noexcept { delete[] (uint8_t*)p; });
            return nb::cast(nb::ndarray<uint8_t, nb::numpy, nb::ndim<3>, nb::c_contig>(
                buf, 3, shape, deleter));
        })
        .def_static("from_numpy", [](nb::ndarray<> arr_any) {
            // Accept (H,W,C) or (H,W) uint8 arrays. We make a contiguous copy
            // since SDImage takes ownership of its own malloc'd buffer.
            nb::module_ np = nb::module_::import_("numpy");
            nb::object arr_obj = np.attr("ascontiguousarray")(
                nb::cast(arr_any), "dtype"_a = np.attr("uint8"));
            if (nb::cast<int>(arr_obj.attr("ndim")) == 2) {
                arr_obj = arr_obj.attr("__getitem__")(
                    nb::make_tuple(nb::slice(nb::none(), nb::none(), nb::none()),
                                   nb::slice(nb::none(), nb::none(), nb::none()),
                                   np.attr("newaxis")));
                arr_obj = np.attr("ascontiguousarray")(arr_obj);
            }
            auto arr = nb::cast<nb::ndarray<uint8_t, nb::ndim<3>, nb::c_contig>>(arr_obj);
            size_t h = arr.shape(0), w = arr.shape(1), c = arr.shape(2);
            if (h && w > (SIZE_MAX / h))
                throw std::overflow_error("Image dimensions too large");
            if (h * w && c > (SIZE_MAX / (h * w)))
                throw std::overflow_error("Image dimensions too large");
            size_t sz = h * w * c;
            auto* w_obj = new SDImageW{};
            w_obj->img.height  = (uint32_t)h;
            w_obj->img.width   = (uint32_t)w;
            w_obj->img.channel = (uint32_t)c;
            w_obj->img.data    = (uint8_t*)std::malloc(sz);
            if (!w_obj->img.data) {
                delete w_obj;
                throw std::bad_alloc();
            }
            std::memcpy(w_obj->img.data, arr.data(), sz);
            w_obj->owns = true;
            return nb::cast(w_obj, nb::rv_policy::take_ownership);
        })
        .def("save_png", [](SDImageW& s, const std::string& path) {
            if (!s.img.data) throw std::runtime_error("Image has no valid data");
            int rc = stbi_write_png(path.c_str(), (int)s.img.width, (int)s.img.height,
                                    (int)s.img.channel, s.img.data,
                                    (int)(s.img.width * s.img.channel));
            if (rc == 0) throw std::runtime_error("Failed to write PNG file: " + path);
        })
        .def("save_jpg", [](SDImageW& s, const std::string& path, int quality) {
            if (!s.img.data) throw std::runtime_error("Image has no valid data");
            int rc = stbi_write_jpg(path.c_str(), (int)s.img.width, (int)s.img.height,
                                    (int)s.img.channel, s.img.data, quality);
            if (rc == 0) throw std::runtime_error("Failed to write JPEG file: " + path);
        }, "path"_a, "quality"_a = 90)
        .def_static("load_stb", [](const std::string& path, int desired_channels) {
            int w, h, c;
            unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, desired_channels);
            if (!data) throw std::runtime_error("Failed to load image: " + path);
            int actual_c = desired_channels > 0 ? desired_channels : c;
            size_t sz = (size_t)w * h * actual_c;
            auto* w_obj = new SDImageW{};
            w_obj->img.width   = (uint32_t)w;
            w_obj->img.height  = (uint32_t)h;
            w_obj->img.channel = (uint32_t)actual_c;
            w_obj->img.data    = (uint8_t*)std::malloc(sz);
            if (!w_obj->img.data) {
                stbi_image_free(data);
                delete w_obj;
                throw std::bad_alloc();
            }
            std::memcpy(w_obj->img.data, data, sz);
            stbi_image_free(data);
            w_obj->owns = true;
            return nb::cast(w_obj, nb::rv_policy::take_ownership);
        }, "path"_a, "channels"_a = 0)
        // Used by the Python wrapper's set_init_image / set_mask_image / etc.
        // — opaque pointer tag the wrapper passes back to the C struct setters.
        .def("_c_image_addr", [](SDImageW& s){
            return (uintptr_t) (void*) &s.img;
        });

    // -------------------------------------------------------------------------
    // SDContextParams (sd_ctx_params_t)
    // -------------------------------------------------------------------------
    nb::class_<SDContextParamsW>(m, "SDContextParams")
        .def(nb::init<>())
        SD_PARAM_PATH(SDContextParamsW, p.model_path,                       model_path_s,                       "model_path")
        SD_PARAM_PATH(SDContextParamsW, p.clip_l_path,                      clip_l_path_s,                      "clip_l_path")
        SD_PARAM_PATH(SDContextParamsW, p.clip_g_path,                      clip_g_path_s,                      "clip_g_path")
        SD_PARAM_PATH(SDContextParamsW, p.clip_vision_path,                 clip_vision_path_s,                 "clip_vision_path")
        SD_PARAM_PATH(SDContextParamsW, p.t5xxl_path,                       t5xxl_path_s,                       "t5xxl_path")
        SD_PARAM_PATH(SDContextParamsW, p.llm_path,                         llm_path_s,                         "llm_path")
        SD_PARAM_PATH(SDContextParamsW, p.llm_vision_path,                  llm_vision_path_s,                  "llm_vision_path")
        SD_PARAM_PATH(SDContextParamsW, p.diffusion_model_path,             diffusion_model_path_s,             "diffusion_model_path")
        SD_PARAM_PATH(SDContextParamsW, p.high_noise_diffusion_model_path,  high_noise_diffusion_model_path_s,  "high_noise_diffusion_model_path")
        SD_PARAM_PATH(SDContextParamsW, p.vae_path,                         vae_path_s,                         "vae_path")
        SD_PARAM_PATH(SDContextParamsW, p.taesd_path,                       taesd_path_s,                       "taesd_path")
        SD_PARAM_PATH(SDContextParamsW, p.control_net_path,                 control_net_path_s,                 "control_net_path")
        SD_PARAM_PATH(SDContextParamsW, p.photo_maker_path,                 photo_maker_path_s,                 "photo_maker_path")
        SD_PARAM_PATH(SDContextParamsW, p.tensor_type_rules,                tensor_type_rules_s,                "tensor_type_rules")
        SD_PARAM_VAL(SDContextParamsW, int, p.n_threads, "n_threads")
        SD_PARAM_VAL(SDContextParamsW, int, p.wtype, "wtype")
        SD_PARAM_VAL(SDContextParamsW, int, p.rng_type, "rng_type")
        SD_PARAM_VAL(SDContextParamsW, int, p.sampler_rng_type, "sampler_rng_type")
        SD_PARAM_VAL(SDContextParamsW, int, p.prediction, "prediction")
        SD_PARAM_VAL(SDContextParamsW, int, p.lora_apply_mode, "lora_apply_mode")
        SD_PARAM_VAL(SDContextParamsW, bool, p.vae_decode_only, "vae_decode_only")
        SD_PARAM_VAL(SDContextParamsW, bool, p.free_params_immediately, "free_params_immediately")
        SD_PARAM_VAL(SDContextParamsW, bool, p.offload_params_to_cpu, "offload_params_to_cpu")
        SD_PARAM_VAL(SDContextParamsW, bool, p.enable_mmap, "enable_mmap")
        SD_PARAM_VAL(SDContextParamsW, bool, p.keep_clip_on_cpu, "keep_clip_on_cpu")
        SD_PARAM_VAL(SDContextParamsW, bool, p.keep_control_net_on_cpu, "keep_control_net_on_cpu")
        SD_PARAM_VAL(SDContextParamsW, bool, p.keep_vae_on_cpu, "keep_vae_on_cpu")
        SD_PARAM_VAL(SDContextParamsW, bool, p.flash_attn, "flash_attn")
        SD_PARAM_VAL(SDContextParamsW, bool, p.diffusion_flash_attn, "diffusion_flash_attn")
        SD_PARAM_VAL(SDContextParamsW, bool, p.tae_preview_only, "tae_preview_only")
        SD_PARAM_VAL(SDContextParamsW, bool, p.diffusion_conv_direct, "diffusion_conv_direct")
        SD_PARAM_VAL(SDContextParamsW, bool, p.vae_conv_direct, "vae_conv_direct")
        SD_PARAM_VAL(SDContextParamsW, bool, p.circular_x, "circular_x")
        SD_PARAM_VAL(SDContextParamsW, bool, p.circular_y, "circular_y")
        SD_PARAM_VAL(SDContextParamsW, bool, p.force_sdxl_vae_conv_scale, "force_sdxl_vae_conv_scale")
        SD_PARAM_VAL(SDContextParamsW, bool, p.chroma_use_dit_mask, "chroma_use_dit_mask")
        SD_PARAM_VAL(SDContextParamsW, bool, p.chroma_use_t5_mask, "chroma_use_t5_mask")
        SD_PARAM_VAL(SDContextParamsW, int,  p.chroma_t5_mask_pad, "chroma_t5_mask_pad")
        SD_PARAM_VAL(SDContextParamsW, bool, p.qwen_image_zero_cond_t, "qwen_image_zero_cond_t")
        .def("__str__", [](SDContextParamsW& s){
            char* str = sd_ctx_params_to_str(&s.p);
            std::string out = str ? std::string(str) : std::string("SDContextParams()");
            if (str) std::free(str);
            return out;
        });

    // -------------------------------------------------------------------------
    // SDSampleParams (sd_sample_params_t — note the embedded guidance struct)
    // -------------------------------------------------------------------------
    nb::class_<SDSampleParamsW>(m, "SDSampleParams")
        .def(nb::init<>())
        SD_PARAM_VAL(SDSampleParamsW, int,   p.sample_method, "sample_method")
        SD_PARAM_VAL(SDSampleParamsW, int,   p.scheduler, "scheduler")
        SD_PARAM_VAL(SDSampleParamsW, int,   p.sample_steps, "sample_steps")
        SD_PARAM_VAL(SDSampleParamsW, float, p.eta, "eta")
        SD_PARAM_VAL(SDSampleParamsW, int,   p.shifted_timestep, "shifted_timestep")
        SD_PARAM_VAL(SDSampleParamsW, float, p.flow_shift, "flow_shift")
        SD_PARAM_VAL(SDSampleParamsW, float, p.guidance.txt_cfg, "cfg_scale")
        SD_PARAM_VAL(SDSampleParamsW, float, p.guidance.img_cfg, "img_cfg_scale")
        SD_PARAM_VAL(SDSampleParamsW, float, p.guidance.distilled_guidance, "distilled_guidance")
        SD_PARAM_VAL(SDSampleParamsW, float, p.guidance.slg.scale, "slg_scale")
        SD_PARAM_VAL(SDSampleParamsW, float, p.guidance.slg.layer_start, "slg_layer_start")
        SD_PARAM_VAL(SDSampleParamsW, float, p.guidance.slg.layer_end, "slg_layer_end")
        .def_prop_rw("slg_layers",
            [](SDSampleParamsW& s) {
                std::vector<int> out;
                for (size_t i = 0; i < s.p.guidance.slg.layer_count; ++i)
                    out.push_back(s.p.guidance.slg.layers[i]);
                return out;
            },
            [](SDSampleParamsW& s, const std::vector<int>& v) {
                s.slg_layers_owned = v;
                if (s.slg_layers_owned.empty()) {
                    s.p.guidance.slg.layers = nullptr;
                    s.p.guidance.slg.layer_count = 0;
                } else {
                    s.p.guidance.slg.layers = s.slg_layers_owned.data();
                    s.p.guidance.slg.layer_count = s.slg_layers_owned.size();
                }
            })
        .def_prop_rw("custom_sigmas",
            [](SDSampleParamsW& s) {
                std::vector<float> out;
                for (int i = 0; i < s.p.custom_sigmas_count; ++i)
                    out.push_back(s.p.custom_sigmas[i]);
                return out;
            },
            [](SDSampleParamsW& s, const std::vector<float>& v) {
                s.custom_sigmas_owned = v;
                if (s.custom_sigmas_owned.empty()) {
                    s.p.custom_sigmas = nullptr;
                    s.p.custom_sigmas_count = 0;
                } else {
                    s.p.custom_sigmas = s.custom_sigmas_owned.data();
                    s.p.custom_sigmas_count = (int)s.custom_sigmas_owned.size();
                }
            })
        .def("__str__", [](SDSampleParamsW& s) {
            char* str = sd_sample_params_to_str(&s.p);
            std::string out = str ? std::string(str) : std::string("SDSampleParams()");
            if (str) std::free(str);
            return out;
        });

    // -------------------------------------------------------------------------
    // SDImageGenParams (sd_img_gen_params_t)
    // -------------------------------------------------------------------------
    nb::class_<SDImageGenParamsW>(m, "SDImageGenParams")
        .def(nb::init<>())
        SD_PARAM_PATH(SDImageGenParamsW, p.prompt,            prompt_s,            "prompt")
        SD_PARAM_PATH(SDImageGenParamsW, p.negative_prompt,   negative_prompt_s,   "negative_prompt")
        SD_PARAM_VAL(SDImageGenParamsW, int,    p.clip_skip, "clip_skip")
        SD_PARAM_VAL(SDImageGenParamsW, int,    p.width, "width")
        SD_PARAM_VAL(SDImageGenParamsW, int,    p.height, "height")
        SD_PARAM_VAL(SDImageGenParamsW, int64_t, p.seed, "seed")
        SD_PARAM_VAL(SDImageGenParamsW, int,    p.batch_count, "batch_count")
        SD_PARAM_VAL(SDImageGenParamsW, float,  p.strength, "strength")
        SD_PARAM_VAL(SDImageGenParamsW, float,  p.control_strength, "control_strength")
        SD_PARAM_VAL(SDImageGenParamsW, bool,   p.auto_resize_ref_image, "auto_resize_ref_image")
        SD_PARAM_VAL(SDImageGenParamsW, bool,   p.increase_ref_index, "increase_ref_index")
        // VAE tiling
        SD_PARAM_VAL(SDImageGenParamsW, bool,   p.vae_tiling_params.enabled, "vae_tiling_enabled")
        SD_PARAM_VAL(SDImageGenParamsW, int,    p.vae_tiling_params.tile_size_x, "vae_tile_size_x")
        SD_PARAM_VAL(SDImageGenParamsW, int,    p.vae_tiling_params.tile_size_y, "vae_tile_size_y")
        SD_PARAM_VAL(SDImageGenParamsW, float,  p.vae_tiling_params.target_overlap, "vae_tile_overlap")
        SD_PARAM_VAL(SDImageGenParamsW, float,  p.vae_tiling_params.rel_size_x, "vae_tile_rel_size_x")
        SD_PARAM_VAL(SDImageGenParamsW, float,  p.vae_tiling_params.rel_size_y, "vae_tile_rel_size_y")
        // Cache params
        SD_PARAM_VAL(SDImageGenParamsW, int,    p.cache.mode, "cache_mode")
        SD_PARAM_VAL(SDImageGenParamsW, float,  p.cache.reuse_threshold, "cache_threshold")
        SD_PARAM_VAL(SDImageGenParamsW, float,  p.cache.start_percent, "cache_start_percent")
        SD_PARAM_VAL(SDImageGenParamsW, float,  p.cache.end_percent, "cache_end_percent")
        SD_PARAM_VAL(SDImageGenParamsW, float,  p.cache.error_decay_rate, "cache_error_decay_rate")
        SD_PARAM_VAL(SDImageGenParamsW, bool,   p.cache.use_relative_threshold, "cache_use_relative_threshold")
        SD_PARAM_VAL(SDImageGenParamsW, bool,   p.cache.reset_error_on_compute, "cache_reset_error_on_compute")
        SD_PARAM_VAL(SDImageGenParamsW, int,    p.cache.Fn_compute_blocks, "cache_fn_compute_blocks")
        SD_PARAM_VAL(SDImageGenParamsW, int,    p.cache.Bn_compute_blocks, "cache_bn_compute_blocks")
        SD_PARAM_VAL(SDImageGenParamsW, float,  p.cache.residual_diff_threshold, "cache_residual_diff_threshold")
        SD_PARAM_VAL(SDImageGenParamsW, int,    p.cache.max_warmup_steps, "cache_max_warmup_steps")
        SD_PARAM_VAL(SDImageGenParamsW, int,    p.cache.max_cached_steps, "cache_max_cached_steps")
        SD_PARAM_VAL(SDImageGenParamsW, int,    p.cache.max_continuous_cached_steps, "cache_max_continuous_cached_steps")
        SD_PARAM_VAL(SDImageGenParamsW, int,    p.cache.taylorseer_n_derivatives, "cache_taylorseer_n_derivatives")
        SD_PARAM_VAL(SDImageGenParamsW, int,    p.cache.taylorseer_skip_interval, "cache_taylorseer_skip_interval")
        SD_PARAM_PATH(SDImageGenParamsW, p.cache.scm_mask, scm_mask_s, "cache_scm_mask")
        SD_PARAM_VAL(SDImageGenParamsW, bool,   p.cache.scm_policy_dynamic, "cache_scm_policy_dynamic")
        SD_PARAM_VAL(SDImageGenParamsW, float,  p.cache.spectrum_w, "cache_spectrum_w")
        SD_PARAM_VAL(SDImageGenParamsW, int,    p.cache.spectrum_m, "cache_spectrum_m")
        SD_PARAM_VAL(SDImageGenParamsW, float,  p.cache.spectrum_lam, "cache_spectrum_lam")
        SD_PARAM_VAL(SDImageGenParamsW, int,    p.cache.spectrum_window_size, "cache_spectrum_window_size")
        SD_PARAM_VAL(SDImageGenParamsW, float,  p.cache.spectrum_flex_window, "cache_spectrum_flex_window")
        SD_PARAM_VAL(SDImageGenParamsW, int,    p.cache.spectrum_warmup_steps, "cache_spectrum_warmup_steps")
        SD_PARAM_VAL(SDImageGenParamsW, float,  p.cache.spectrum_stop_percent, "cache_spectrum_stop_percent")
        // Hires-fix
        SD_PARAM_VAL(SDImageGenParamsW, bool,   p.hires.enabled, "hires_enabled")
        SD_PARAM_VAL(SDImageGenParamsW, int,    p.hires.upscaler, "hires_upscaler")
        SD_PARAM_PATH(SDImageGenParamsW, p.hires.model_path, hires_model_path_s, "hires_model_path")
        SD_PARAM_VAL(SDImageGenParamsW, float,  p.hires.scale, "hires_scale")
        SD_PARAM_VAL(SDImageGenParamsW, int,    p.hires.target_width,  "hires_target_width")
        SD_PARAM_VAL(SDImageGenParamsW, int,    p.hires.target_height, "hires_target_height")
        SD_PARAM_VAL(SDImageGenParamsW, int,    p.hires.steps, "hires_steps")
        SD_PARAM_VAL(SDImageGenParamsW, float,  p.hires.denoising_strength, "hires_denoising_strength")
        SD_PARAM_VAL(SDImageGenParamsW, int,    p.hires.upscale_tile_size, "hires_tile_size")
        // Photo maker
        SD_PARAM_PATH(SDImageGenParamsW, p.pm_params.id_embed_path, pm_id_embed_path_s, "pm_id_embed_path")
        SD_PARAM_VAL(SDImageGenParamsW, float,  p.pm_params.style_strength, "pm_style_strength")
        // sample_params is the SDSampleParams handle. Setting it copies; getting
        // returns a reference that the caller can mutate (write-through).
        .def_prop_rw("sample_params",
            [](SDImageGenParamsW& s) -> SDSampleParamsW& { return s.sample; },
            [](SDImageGenParamsW& s, SDSampleParamsW& v) {
                s.sample = SDSampleParamsW();  // re-init owning vectors
                s.sample.p = v.p;
                s.sample.slg_layers_owned = v.slg_layers_owned;
                s.sample.custom_sigmas_owned = v.custom_sigmas_owned;
                if (!s.sample.slg_layers_owned.empty())
                    s.sample.p.guidance.slg.layers = s.sample.slg_layers_owned.data();
                if (!s.sample.custom_sigmas_owned.empty())
                    s.sample.p.custom_sigmas = s.sample.custom_sigmas_owned.data();
                s.sync_sample();
            },
            nb::rv_policy::reference_internal)
        .def("set_init_image", [](SDImageGenParamsW& s, SDImageW& img, nb::object pyref){
            s.p.init_image = img.img;
            s.init_image_ref = pyref;
        })
        .def("set_mask_image", [](SDImageGenParamsW& s, SDImageW& img, nb::object pyref){
            s.p.mask_image = img.img;
            s.mask_image_ref = pyref;
        })
        .def("set_control_image", [](SDImageGenParamsW& s, SDImageW& img, nb::object pyref){
            s.p.control_image = img.img;
            s.control_image_ref = pyref;
        })
        .def("set_ref_images", [](SDImageGenParamsW& s, nb::list images){
            s.ref_images_buf.clear();
            if (images.size() == 0) {
                s.p.ref_images = nullptr;
                s.p.ref_images_count = 0;
                s.ref_images_pyref = nb::object();
                return;
            }
            for (auto h : images) {
                SDImageW& w = nb::cast<SDImageW&>(h);
                s.ref_images_buf.push_back(w.img);
            }
            s.p.ref_images = s.ref_images_buf.data();
            s.p.ref_images_count = (int)s.ref_images_buf.size();
            s.ref_images_pyref = images;  // keep all SDImage refs alive
        })
        .def("set_pm_id_images", [](SDImageGenParamsW& s, nb::list images){
            s.pm_id_images_buf.clear();
            if (images.size() == 0) {
                s.p.pm_params.id_images = nullptr;
                s.p.pm_params.id_images_count = 0;
                s.pm_id_images_pyref = nb::object();
                return;
            }
            for (auto h : images) {
                SDImageW& w = nb::cast<SDImageW&>(h);
                s.pm_id_images_buf.push_back(w.img);
            }
            s.p.pm_params.id_images = s.pm_id_images_buf.data();
            s.p.pm_params.id_images_count = (int)s.pm_id_images_buf.size();
            s.pm_id_images_pyref = images;
        })
        .def("set_loras", [](SDImageGenParamsW& s, nb::list loras){
            s.loras_buf.clear();
            s.lora_paths_owned.clear();
            if (loras.size() == 0) {
                s.p.loras = nullptr;
                s.p.lora_count = 0;
                return;
            }
            s.lora_paths_owned.reserve(loras.size());
            s.loras_buf.reserve(loras.size());
            for (auto h : loras) {
                nb::dict d = nb::cast<nb::dict>(h);
                s.lora_paths_owned.push_back(nb::cast<std::string>(d["path"]));
                sd_lora_t entry{};
                entry.path = s.lora_paths_owned.back().c_str();
                entry.multiplier = d.contains("multiplier")
                    ? nb::cast<float>(d["multiplier"]) : 1.0f;
                entry.is_high_noise = d.contains("is_high_noise")
                    ? nb::cast<bool>(d["is_high_noise"]) : false;
                s.loras_buf.push_back(entry);
            }
            s.p.loras = s.loras_buf.data();
            s.p.lora_count = (uint32_t)s.loras_buf.size();
        })
        .def("__str__", [](SDImageGenParamsW& s) {
            s.sync_sample();
            char* str = sd_img_gen_params_to_str(&s.p);
            std::string out = str ? std::string(str) : std::string("SDImageGenParams()");
            if (str) std::free(str);
            return out;
        });

    // -------------------------------------------------------------------------
    // SDContext
    // -------------------------------------------------------------------------
    nb::class_<SDContextW>(m, "SDContext")
        .def(nb::init<SDContextParamsW&>(), "params"_a)
        .def_prop_ro("_busy_lock", [](SDContextW& s){ return s.busy_lock; })
        .def("_try_acquire_busy", &SDContextW::try_acquire_busy)
        .def("close", [](SDContextW& s){
            if (s.ctx) { free_sd_ctx(s.ctx); s.ctx = nullptr; }
        })
        .def_prop_ro("is_valid", [](SDContextW& s){ return s.ctx != nullptr; })
        .def_prop_ro("supports_image_generation", [](SDContextW& s){
            if (!s.ctx) throw std::runtime_error("Context not initialized");
            return (bool) sd_ctx_supports_image_generation(s.ctx);
        })
        .def_prop_ro("supports_video_generation", [](SDContextW& s){
            if (!s.ctx) throw std::runtime_error("Context not initialized");
            return (bool) sd_ctx_supports_video_generation(s.ctx);
        })
        .def("get_default_sample_method", [](SDContextW& s){
            if (!s.ctx) throw std::runtime_error("Context not initialized");
            return (int) sd_get_default_sample_method(s.ctx);
        })
        .def("get_default_scheduler", [](SDContextW& s, std::optional<int> sm){
            if (!s.ctx) throw std::runtime_error("Context not initialized");
            sample_method_t method = sm ? (sample_method_t)*sm
                                        : sd_get_default_sample_method(s.ctx);
            return (int) sd_get_default_scheduler(s.ctx, method);
        }, "sample_method"_a = nb::none())
        .def("generate_with_params", [](SDContextW& s, SDImageGenParamsW& params){
            if (!s.ctx) throw std::runtime_error("Context not initialized");
            params.sync_sample();
            int batch = params.p.batch_count;
            sd_image_t* result = nullptr;

            {
                inferna::BusyGuard guard(s.busy_lock, SDContextW::kBusyMsg);
                nb::gil_scoped_release rel;
                result = generate_image(s.ctx, &params.p);
            }

            if (!result) throw std::runtime_error("Image generation failed");

            nb::list out;
            int n_invalid = 0;
            for (int i = 0; i < batch; ++i) {
                auto* w = new SDImageW{};
                w->img = result[i];
                w->owns = true;
                if (!w->img.data || w->img.width == 0 || w->img.height == 0)
                    n_invalid++;
                out.append(nb::cast(w, nb::rv_policy::take_ownership));
            }
            std::free(result);

            if (n_invalid == batch) {
                throw std::runtime_error(
                    "Image generation failed: all images have invalid data. "
                    "This usually means GPU memory allocation failed (out of memory). "
                    "Try a smaller model quantization (e.g. Q4_K_M), reduced dimensions, "
                    "or --offload-to-cpu (note: offloading may not work for all model "
                    "architectures, e.g. z-image).");
            }
            if (n_invalid > 0) {
                nb::module_ warnings = nb::module_::import_("warnings");
                std::string msg = std::to_string(n_invalid) + "/"
                    + std::to_string(batch) + " images failed to generate "
                    "(likely GPU memory allocation failure)";
                warnings.attr("warn")(msg);
            }
            return out;
        })
        .def("generate_video_raw", [](SDContextW& s,
                const std::string& prompt,
                const std::string& negative_prompt,
                int width, int height, int video_frames,
                int sample_method, int scheduler,
                int sample_steps, float cfg_scale,
                int64_t seed, int clip_skip, float strength,
                float eta, float moe_boundary, float vace_strength,
                std::optional<SDImageW*> init_image,
                std::optional<SDImageW*> end_image) {
            if (!s.ctx) throw std::runtime_error("Context not initialized");
            sd_vid_gen_params_t vid_params;
            sd_vid_gen_params_init(&vid_params);
            vid_params.prompt = prompt.c_str();
            vid_params.negative_prompt = negative_prompt.c_str();
            vid_params.width = width;
            vid_params.height = height;
            vid_params.video_frames = video_frames;
            vid_params.clip_skip = clip_skip;
            vid_params.strength = strength;
            vid_params.seed = seed;
            vid_params.sample_params.sample_method = (sample_method_t) sample_method;
            vid_params.sample_params.scheduler = (scheduler_t) scheduler;
            vid_params.sample_params.sample_steps = sample_steps;
            vid_params.sample_params.guidance.txt_cfg = cfg_scale;
            vid_params.sample_params.eta = eta;
            vid_params.moe_boundary = moe_boundary;
            vid_params.vace_strength = vace_strength;
            if (init_image && *init_image) vid_params.init_image = (*init_image)->img;
            if (end_image  && *end_image)  vid_params.end_image  = (*end_image)->img;

            int num_frames_out = 0;
            sd_image_t* result = nullptr;
            {
                inferna::BusyGuard guard(s.busy_lock, SDContextW::kBusyMsg);
                nb::gil_scoped_release rel;
                result = generate_video(s.ctx, &vid_params, &num_frames_out);
            }

            if (!result) throw std::runtime_error("Video generation failed");

            nb::list out;
            int n_invalid = 0;
            for (int i = 0; i < num_frames_out; ++i) {
                auto* w = new SDImageW{};
                w->img = result[i];
                w->owns = true;
                if (!w->img.data) n_invalid++;
                out.append(nb::cast(w, nb::rv_policy::take_ownership));
            }
            std::free(result);
            if (num_frames_out > 0 && n_invalid == num_frames_out)
                throw std::runtime_error(
                    "Video generation failed: all frames have invalid data. "
                    "This usually means GPU memory allocation failed (out of memory).");
            if (n_invalid > 0) {
                nb::module_ warnings = nb::module_::import_("warnings");
                std::string msg = std::to_string(n_invalid) + "/"
                    + std::to_string(num_frames_out) + " video frames failed to generate";
                warnings.attr("warn")(msg);
            }
            return out;
        });

    // -------------------------------------------------------------------------
    // Upscaler
    // -------------------------------------------------------------------------
    nb::class_<UpscalerW>(m, "Upscaler")
        .def(nb::init<const std::string&, bool, bool, int, int>(),
             "model_path"_a, "offload_to_cpu"_a = false, "direct"_a = false,
             "n_threads"_a = -1, "tile_size"_a = 0)
        .def_prop_ro("is_valid", [](UpscalerW& s){ return s.ctx != nullptr; })
        .def_prop_ro("upscale_factor", [](UpscalerW& s){
            return s.ctx ? get_upscale_factor(s.ctx) : 0;
        })
        .def("upscale", [](UpscalerW& s, SDImageW& image, int factor){
            if (!s.ctx) throw std::runtime_error("Upscaler not initialized");
            if (factor == 0) factor = get_upscale_factor(s.ctx);
            sd_image_t result = ::upscale(s.ctx, image.img, (uint32_t)factor);
            if (!result.data) throw std::runtime_error("Upscaling failed");
            auto* w = new SDImageW{};
            w->img = result;
            w->owns = true;
            return nb::cast(w, nb::rv_policy::take_ownership);
        }, "image"_a, "factor"_a = 0);

    // -------------------------------------------------------------------------
    // Module-level functions
    // -------------------------------------------------------------------------
    m.def("get_num_cores", [](){ return sd_get_num_physical_cores(); });
    m.def("get_system_info", [](){
        const char* info = sd_get_system_info();
        return info ? std::string(info) : std::string();
    });
    m.def("type_name", [](int t){
        const char* n = sd_type_name((sd_type_t)t);
        return n ? std::string(n) : std::string();
    });
    m.def("sample_method_name", [](int t){
        const char* n = sd_sample_method_name((sample_method_t)t);
        return n ? std::string(n) : std::string();
    });
    m.def("scheduler_name", [](int t){
        const char* n = sd_scheduler_name((scheduler_t)t);
        return n ? std::string(n) : std::string();
    });

    m.def("convert_native", [](const std::string& input_path,
                               const std::string& output_path,
                               int output_type,
                               std::optional<std::string> vae_path,
                               std::optional<std::string> tensor_type_rules,
                               bool convert_name) {
        const char* vae_ptr = vae_path ? vae_path->c_str() : nullptr;
        const char* rules_ptr = tensor_type_rules ? tensor_type_rules->c_str() : nullptr;
        return (bool) convert(input_path.c_str(), vae_ptr, output_path.c_str(),
                              (sd_type_t)output_type, rules_ptr, convert_name);
    });

    m.def("preprocess_canny", [](SDImageW& img, float high_threshold, float low_threshold,
                                  float weak, float strong, bool inverse) {
        return (bool) preprocess_canny(img.img, high_threshold, low_threshold,
                                       weak, strong, inverse);
    });

    m.def("set_log_callback", [](nb::object cb){
        g_log_cb = cb;
        if (cb.is_none()) sd_set_log_callback(nullptr, nullptr);
        else              sd_set_log_callback(_c_log_cb, nullptr);
    }, "cb"_a.none());
    m.def("set_progress_callback", [](nb::object cb){
        g_progress_cb = cb;
        if (cb.is_none()) sd_set_progress_callback(nullptr, nullptr);
        else              sd_set_progress_callback(_c_progress_cb, nullptr);
    }, "cb"_a.none());
    m.def("set_preview_callback", [](nb::object cb, int mode, int interval,
                                      bool denoised, bool noisy){
        g_preview_cb = cb;
        if (cb.is_none())
            sd_set_preview_callback(nullptr, PREVIEW_NONE, 1, false, false, nullptr);
        else
            sd_set_preview_callback(_c_preview_cb, (preview_t)mode, interval,
                                    denoised, noisy, nullptr);
    }, "cb"_a.none(), "mode"_a, "interval"_a, "denoised"_a, "noisy"_a);

    m.def("ggml_backend_load_all", [](){
        nb::module_ os = nb::module_::import_("os");
        nb::module_ backend_dl = nb::module_::import_("inferna._internal.backend_dl");
        nb::object __file__ = nb::module_::import_("inferna.sd._sd_native").attr("__file__");
        std::string this_file = nb::cast<std::string>(__file__);
        std::string this_dir = nb::cast<std::string>(os.attr("path").attr("dirname")(
            os.attr("path").attr("abspath")(this_file)));
        std::string llama_dir = nb::cast<std::string>(os.attr("path").attr("join")(
            os.attr("path").attr("dirname")(this_dir), "llama"));
        if (nb::cast<bool>(os.attr("path").attr("isdir")(llama_dir)))
            ggml_backend_load_all_from_path(llama_dir.c_str());
        else
            ggml_backend_load_all_from_path(this_dir.c_str());

        std::string site = nb::cast<std::string>(os.attr("path").attr("dirname")(
            os.attr("path").attr("dirname")(this_dir)));
        nb::object paths = backend_dl.attr("libs_to_load")(site);
        for (nb::handle p : paths) {
            std::string sp = nb::cast<std::string>(p);
            ggml_backend_load(sp.c_str());
        }
    });
}
