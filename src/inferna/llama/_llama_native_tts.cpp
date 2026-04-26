// TTS helper bindings (xterm color, WAV writer, hann window, FFT/IRFFT,
// number-to-words, OuteTTS text preprocessor). Registered into the
// _llama_native module via register_tts(). The C++ implementation lives
// in helpers/tts.cpp (linked into the same target).

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <string>
#include <vector>

#include "helpers/tts.h"

namespace nb = nanobind;
using namespace nb::literals;

void register_tts(nb::module_& m) {
    m.def("rgb2xterm256", &rgb2xterm256, "r"_a, "g"_a, "b"_a);

    m.def("set_xterm256_foreground", &set_xterm256_foreground, "r"_a, "g"_a, "b"_a);

    m.def("save_wav16", [](const std::string& fname,
                             std::vector<float> data, int sample_rate) {
        return save_wav16(fname, data, sample_rate);
    }, "fname"_a, "data"_a, "sample_rate"_a);

    m.def("save_wav16_from_list", [](const std::string& fname,
                                       std::vector<float> data, int sample_rate) {
        return save_wav16(fname, data, sample_rate);
    }, "fname"_a, "data"_a, "sample_rate"_a);

    m.def("fill_hann_window", [](int length, bool periodic) {
        std::vector<float> out(length);
        fill_hann_window(length, periodic, out.data());
        return out;
    }, "length"_a, "periodic"_a);

    m.def("twiddle_factors", [](float real, float imag, int k, int N) {
        twiddle(&real, &imag, k, N);
        return nb::make_tuple(real, imag);
    }, "real"_a, "imag"_a, "k"_a, "N"_a);

    m.def("irfft", [](std::vector<float> inp_cplx) {
        int n = (int) inp_cplx.size();
        std::vector<float> out(n);
        irfft(n, inp_cplx.data(), out.data());
        return out;
    }, "inp_cplx"_a);

    m.def("fold", [](std::vector<float> data, int64_t n_out, int64_t n_win,
                       int64_t n_hop, int64_t n_pad) {
        std::vector<float> out;
        fold(data, n_out, n_win, n_hop, n_pad, out);
        return out;
    }, "data"_a, "n_out"_a, "n_win"_a, "n_hop"_a, "n_pad"_a);

    m.def("convert_less_than_thousand", &convert_less_than_thousand, "num"_a);

    m.def("number_to_words", &number_to_words, "number_str"_a);

    m.def("replace_numbers_with_words", &replace_numbers_with_words, "input_text"_a);

    m.def("process_text", [](const std::string& text, int tts_version) {
        return process_text(text, (outetts_version) tts_version);
    }, "text"_a, "tts_version"_a);
}
