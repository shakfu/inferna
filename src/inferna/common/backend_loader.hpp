// Shared helper for the three native modules' `ggml_backend_load_all`
// bindings. Each module previously inlined the same ~15 lines of path
// resolution + Python-side `libs_to_load` plumbing.
//
// The TU including this header is responsible for declaring the
// `ggml_backend_load*` C symbols (either by including ggml-backend.h, as
// llama does, or via the extern "C" forward decls used by whisper/sd to
// avoid header conflicts). The helper itself only references those
// symbols and the nanobind/python C API.

#pragma once

#include <nanobind/nanobind.h>
#include <string>

namespace inferna {

// Discover and load ggml backend plugins for the calling extension.
//
// Resolution order:
//   1. Sibling `inferna/llama/` directory (where the llama wheel ships
//      its `ggml-*` plugin shared libs). For the llama extension itself
//      this happens to be its own directory.
//   2. Fallback: the calling extension's own directory.
// Then any extras reported by `inferna._internal.backend_dl.libs_to_load`
// (delvewheel-relocated copies on Windows, etc.) are loaded individually.
//
// `extension_module_name` is the dotted import path of the calling
// extension — e.g. "inferna.llama._llama_native". It is imported solely
// to read `__file__` for path resolution.
inline void load_all_backends(const char* extension_module_name) {
    namespace nb = nanobind;
    nb::module_ os         = nb::module_::import_("os");
    nb::module_ backend_dl = nb::module_::import_("inferna._internal.backend_dl");
    nb::object  __file__   = nb::module_::import_(extension_module_name).attr("__file__");

    std::string this_file = nb::cast<std::string>(__file__);
    std::string this_dir  = nb::cast<std::string>(
        os.attr("path").attr("dirname")(os.attr("path").attr("abspath")(this_file)));
    std::string parent    = nb::cast<std::string>(os.attr("path").attr("dirname")(this_dir));
    std::string llama_dir = nb::cast<std::string>(os.attr("path").attr("join")(parent, "llama"));

    if (nb::cast<bool>(os.attr("path").attr("isdir")(llama_dir))) {
        ggml_backend_load_all_from_path(llama_dir.c_str());
    } else {
        ggml_backend_load_all_from_path(this_dir.c_str());
    }

    std::string site = nb::cast<std::string>(os.attr("path").attr("dirname")(parent));
    nb::object  paths = backend_dl.attr("libs_to_load")(site);
    for (nb::handle p : paths) {
        std::string sp = nb::cast<std::string>(p);
        ggml_backend_load(sp.c_str());
    }
}

} // namespace inferna
