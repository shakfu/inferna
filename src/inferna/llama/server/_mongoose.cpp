// Minimal nanobind bindings for mongoose's mg_mgr lifecycle.
//
// Exposes a `Manager` class plus opaque `Connection` integer handles.
// The `embedded.py` module layered above this holds the EmbeddedServer /
// MongooseConnection / handle_http_request logic in pure Python.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>

#include <cstring>
#include <memory>
#include <string>

#include "mongoose.h"

// Re-declare the inferna_* shims (defined in mongoose_wrapper.c) so the
// C++ TU links against them without needing a header.
extern "C" {
    void inferna_mg_mgr_init(struct mg_mgr* mgr);
    void inferna_mg_mgr_free(struct mg_mgr* mgr);
    void inferna_mg_mgr_poll(struct mg_mgr* mgr, int timeout_ms);
    struct mg_connection* inferna_mg_http_listen(struct mg_mgr* mgr, const char* url,
                                                 mg_event_handler_t fn, void* fn_data);
    void inferna_mg_http_reply(struct mg_connection* c, int status_code,
                                const char* headers, const char* body_fmt, ...);
}

namespace nb = nanobind;
using namespace nb::literals;

// =============================================================================
// Manager — owns mg_mgr, dispatches HTTP requests to a Python callback.
// =============================================================================

struct Manager {
    mg_mgr mgr{};
    mg_connection* listener = nullptr;
    nb::object handler;  // Python callable: (conn_id:int, method:str, uri:str, body:str) -> None

    Manager() {
        inferna_mg_mgr_init(&mgr);
        mgr.userdata = this;
    }
    ~Manager() {
        inferna_mg_mgr_free(&mgr);
    }
    Manager(const Manager&) = delete;
    Manager& operator=(const Manager&) = delete;
};

// HTTP event handler — bridges mongoose's C callback into the registered
// Python handler. We pull `Manager*` out of mgr.userdata, then dispatch.
extern "C" void _http_event_handler(mg_connection* c, int ev, void* ev_data) {
    if (!c || !c->mgr) return;
    Manager* self = static_cast<Manager*>(c->mgr->userdata);
    if (!self) return;

    if (ev == MG_EV_HTTP_MSG) {
        mg_http_message* hm = static_cast<mg_http_message*>(ev_data);
        if (!hm) return;

        nb::gil_scoped_acquire gil;
        if (!self->handler.is_valid() || self->handler.is_none()) return;

        try {
            std::string method(hm->method.buf, hm->method.len);
            std::string uri(hm->uri.buf, hm->uri.len);
            std::string body = hm->body.len > 0
                ? std::string(hm->body.buf, hm->body.len) : std::string();
            uintptr_t conn_id = reinterpret_cast<uintptr_t>(c);
            self->handler(conn_id, method, uri, body);
        } catch (...) {
            // Logging is the Python handler's responsibility; we just send
            // a plain 500 if it threw.
            inferna_mg_http_reply(c, 500, "Content-Type: text/plain\r\n",
                                   "%s", "Internal Server Error");
        }
    }
}

// =============================================================================
// Module
// =============================================================================

NB_MODULE(_mongoose, m) {
    nb::class_<Manager>(m, "Manager")
        .def(nb::init<>())
        .def("set_handler", [](Manager& s, nb::object h){ s.handler = h; },
             "handler"_a.none(),
             "Register a Python callable invoked on every HTTP request as "
             "(conn_id, method, uri, body).")
        .def("listen", [](Manager& s, const std::string& url) {
            // null fn_data — the handler reads back through mgr.userdata,
            // which already points at our Manager instance (set in ctor).
            s.listener = inferna_mg_http_listen(&s.mgr, url.c_str(),
                                                 _http_event_handler, nullptr);
            return s.listener != nullptr;
        }, "url"_a)
        .def("poll", [](Manager& s, int timeout_ms){
            // Release the GIL for the blocking poll — Python signal handlers
            // and other threads keep running while mongoose ticks.
            nb::gil_scoped_release rel;
            inferna_mg_mgr_poll(&s.mgr, timeout_ms);
        }, "timeout_ms"_a)
        .def("close_all_connections", [](Manager& s){
            int n = 0;
            for (mg_connection* c = s.mgr.conns; c; c = c->next) {
                c->is_closing = 1;
                ++n;
            }
            return n;
        })
        .def_prop_ro("is_listening", [](Manager& s){ return s.listener != nullptr; });

    // Reply primitives — operate on the opaque connection id we surfaced
    // in the handler callback.
    m.def("send_reply", [](uintptr_t conn_id, int status_code,
                            const std::string& headers, const std::string& body){
        mg_connection* c = reinterpret_cast<mg_connection*>(conn_id);
        if (!c) return false;
        inferna_mg_http_reply(c, status_code, headers.c_str(), "%s", body.c_str());
        return true;
    }, "conn_id"_a, "status_code"_a, "headers"_a, "body"_a);
}
