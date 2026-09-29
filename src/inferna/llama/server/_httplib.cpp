// nanobind bindings for a cpp-httplib server that routes every request to one
// Python callable. Routing, auth and response building stay in embedded.py.
//
// The handler is called as handler(method, path, headers, body) on an httplib
// worker thread, with the GIL held, and returns
// (status, content_type, extra_headers, body). A bytes body is sent whole; any
// other body is an iterator of bytes, sent chunked.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cctype>
#include <memory>
#include <string>

#include "httplib.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// Owns a Python iterator. Closing it runs a generator's finally blocks, so a
// stream that ends early or never starts still releases what it holds.
struct PyStream {
    nb::object it;

    ~PyStream() {
        nb::gil_scoped_acquire gil;
        try {
            if (nb::hasattr(it, "close")) it.attr("close")();
        } catch (nb::python_error &e) {
            e.discard_as_unraisable("inferna http stream close");
        }
        it.reset();
    }
};

nb::dict header_dict(const httplib::Headers &headers) {
    // Header names are case-insensitive (RFC 9110), so keys are lowercased.
    nb::dict out;
    for (const auto &kv : headers) {
        std::string name = kv.first;
        for (auto &ch : name) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        out[name.c_str()] = kv.second;
    }
    return out;
}

void set_error(httplib::Response &res) {
    res.status = 500;
    res.set_content("Internal Server Error", "text/plain");
}

void apply(nb::handle result, httplib::Response &res) {
    nb::tuple t = nb::cast<nb::tuple>(result);
    if (t.size() != 4) throw nb::value_error("handler must return (status, content_type, headers, body)");
    res.status = nb::cast<int>(t[0]);
    std::string content_type = nb::cast<std::string>(t[1]);
    for (auto kv : nb::cast<nb::dict>(t[2])) {
        res.set_header(nb::cast<std::string>(kv.first), nb::cast<std::string>(kv.second));
    }
    nb::handle body = t[3];
    if (nb::isinstance<nb::bytes>(body)) {
        nb::bytes b = nb::borrow<nb::bytes>(body);
        res.set_content(b.c_str(), b.size(), content_type);
        return;
    }
    // httplib copies the provider; the shared_ptr closes the iterator once, after the last copy.
    auto stream = std::make_shared<PyStream>();
    stream->it = nb::iter(body);
    res.set_chunked_content_provider(content_type, [stream](size_t, httplib::DataSink &sink) {
        std::string chunk;
        {
            nb::gil_scoped_acquire gil;
            PyObject *item = PyIter_Next(stream->it.ptr());
            if (item == nullptr) {
                if (PyErr_Occurred()) {
                    // Headers are already sent; aborting truncates the chunked body.
                    nb::python_error e;
                    e.discard_as_unraisable("inferna http stream");
                    return false;
                }
                sink.done();
                return true;
            }
            nb::object o = nb::steal(item);
            if (!nb::isinstance<nb::bytes>(o)) return false;
            nb::bytes b = nb::borrow<nb::bytes>(o);
            chunk.assign(b.c_str(), b.size());
        }
        // A failed write means the client is gone; httplib then drops the provider.
        return sink.write(chunk.data(), chunk.size());
    });
}

struct Server {
    httplib::Server svr;
    nb::object handler;

    explicit Server(size_t max_body) {
        // httplib's default sets SO_REUSEPORT, which lets a second process bind the same port.
        svr.set_socket_options([](socket_t sock) {
#ifdef _WIN32
            httplib::set_socket_opt(sock, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, 1);
#else
            httplib::set_socket_opt(sock, SOL_SOCKET, SO_REUSEADDR, 1);
#endif
        });
        // Checked against Content-Length before the body is read.
        svr.set_payload_max_length(max_body);
        // llama-server's defaults (common_params::timeout_read/_write); httplib's are 5 s.
        svr.set_read_timeout(3600);
        svr.set_write_timeout(3600);

        auto route = [this](const httplib::Request &req, httplib::Response &res) {
            nb::gil_scoped_acquire gil;
            nb::object h = handler;
            if (!h.is_valid() || h.is_none()) {
                res.status = 503;
                return;
            }
            try {
                nb::object r = h(req.method, req.path, header_dict(req.headers),
                                 nb::bytes(req.body.data(), req.body.size()));
                apply(r, res);
            } catch (nb::python_error &e) {
                e.discard_as_unraisable("inferna http handler");
                set_error(res);
            } catch (const std::exception &) {
                set_error(res);
            }
        };
        svr.Get(".*", route);
        svr.Post(".*", route);
        svr.Put(".*", route);
        svr.Patch(".*", route);
        svr.Delete(".*", route);
        svr.Options(".*", route);
    }
    Server(const Server &) = delete;
    Server &operator=(const Server &) = delete;
};

}  // namespace

NB_MODULE(_httplib, m) {
    nb::class_<Server>(m, "Server")
        .def(nb::init<size_t>(), "max_body"_a)
        .def(
            "set_handler", [](Server &s, nb::object h) { s.handler = std::move(h); }, "handler"_a.none(),
            "Set the callable that answers every request, or None to answer 503.")
        .def(
            "bind",
            [](Server &s, const std::string &host, int port, bool ipv6) {
                s.svr.set_address_family(ipv6 ? AF_INET6 : AF_INET);
                return s.svr.bind_to_port(host, port);
            },
            "host"_a, "port"_a, "ipv6"_a,
            "Bind exactly `host`; the address family is fixed so it is never widened. Returns False on failure.")
        .def(
            "listen", [](Server &s) { return s.svr.listen_after_bind(); }, nb::call_guard<nb::gil_scoped_release>(),
            "Serve until stop(). Blocks.")
        .def(
            "stop", [](Server &s) { s.svr.stop(); }, nb::call_guard<nb::gil_scoped_release>(),
            "Close the listener; listen() returns once in-flight requests finish.");
}
