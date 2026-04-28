// RAII guard for the per-context Python `threading.Lock` used by the
// whisper and sd nanobind wrappers to serialize entry into non-thread-safe
// native calls. Construction acquires the lock (raising on contention),
// destruction releases it -- so error paths and `gil_scoped_release` can
// no longer drift out of sync between wrappers.

#pragma once

#include <nanobind/nanobind.h>
#include <stdexcept>
#include <string>

namespace inferna {

class BusyGuard {
public:
    BusyGuard(nanobind::object& lock, const char* contention_msg)
        : lock_(lock)
    {
        namespace nb = nanobind;
        using namespace nb::literals;
        nb::object acquired = lock_.attr("acquire")("blocking"_a = false);
        if (!nb::cast<bool>(acquired)) {
            throw std::runtime_error(contention_msg);
        }
    }

    ~BusyGuard() {
        try { lock_.attr("release")(); } catch (...) {}
    }

    BusyGuard(const BusyGuard&) = delete;
    BusyGuard& operator=(const BusyGuard&) = delete;

private:
    nanobind::object& lock_;
};

} // namespace inferna
