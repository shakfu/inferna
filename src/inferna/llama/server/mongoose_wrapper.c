/*
 * C wrapper for mongoose.c to handle compilation separately from C++ code
 * This allows us to compile mongoose.c with C flags and the rest with C++
 */

#include <stdarg.h>
#include <stdio.h>
#include "mongoose.h"

/* Re-export all mongoose functions we need for Cython */

void inferna_mg_mgr_init(struct mg_mgr *mgr) {
    mg_mgr_init(mgr);
}

void inferna_mg_mgr_free(struct mg_mgr *mgr) {
    mg_mgr_free(mgr);
}

void inferna_mg_mgr_poll(struct mg_mgr *mgr, int timeout_ms) {
    mg_mgr_poll(mgr, timeout_ms);
}

struct mg_connection *inferna_mg_http_listen(struct mg_mgr *mgr, const char *url,
                                            mg_event_handler_t fn, void *fn_data) {
    return mg_http_listen(mgr, url, fn, fn_data);
}

void inferna_mg_http_reply(struct mg_connection *c, int status_code, const char *headers,
                          const char *body_fmt, ...) {
    va_list ap;
    va_start(ap, body_fmt);

    /* Create a formatted string from the body_fmt and args */
    char body[4096];
    vsnprintf(body, sizeof(body), body_fmt, ap);
    va_end(ap);

    mg_http_reply(c, status_code, headers, "%s", body);
}

struct mg_str *inferna_mg_http_get_header(struct mg_http_message *hm, const char *name) {
    return mg_http_get_header(hm, name);
}

int inferna_mg_http_get_var(const struct mg_str *buf, const char *name, char *dst, size_t dst_len) {
    return mg_http_get_var(buf, name, dst, dst_len);
}

/* Helper functions for string handling */
struct mg_str inferna_mg_str(const char *s) {
    return mg_str(s);
}

struct mg_str inferna_mg_str_n(const char *s, size_t n) {
    return mg_str_n(s, n);
}

/* Send a binary response with explicit length (NUL-safe; needed for gzipped
 * static assets). Mirrors mg_http_reply but without the printf path that would
 * truncate at the first embedded zero byte. */
void inferna_mg_http_reply_bytes(struct mg_connection *c, int status_code,
                                 const char *headers, const char *body,
                                 size_t body_len) {
    /* Mongoose's mg_printf has its own format-specifier subset and does
     * NOT support %zu — using it produces an empty Content-Length value,
     * which curl rejects ("Invalid Content-Length") and browsers treat
     * as a 0-length body. Cast to unsigned long long and use %llu, which
     * mongoose does support. */
    mg_printf(c, "HTTP/1.1 %d %s\r\n%sContent-Length: %llu\r\n\r\n",
              status_code,
              status_code == 200 ? "OK" : "Error",
              headers ? headers : "",
              (unsigned long long) body_len);
    if (body_len > 0) {
        mg_send(c, body, body_len);
    }
    c->is_resp = 0;
}

/* Begin a chunked (Transfer-Encoding: chunked) response. Caller follows with
 * one or more inferna_mg_http_write_chunk() calls and finally
 * inferna_mg_http_end_chunk(). Used by the SSE streaming path. */
void inferna_mg_begin_chunked(struct mg_connection *c, int status_code,
                              const char *headers) {
    mg_printf(c, "HTTP/1.1 %d %s\r\n%sTransfer-Encoding: chunked\r\n\r\n",
              status_code,
              status_code == 200 ? "OK" : "Error",
              headers ? headers : "");
}

void inferna_mg_http_write_chunk(struct mg_connection *c, const char *buf,
                                 size_t len) {
    mg_http_write_chunk(c, buf, len);
}

void inferna_mg_http_end_chunk(struct mg_connection *c) {
    /* Empty chunk terminates a chunked body per RFC 9112. */
    mg_http_printf_chunk(c, "");
    c->is_resp = 0;
}

int inferna_mg_is_closing(struct mg_connection *c) {
    return (c->is_closing || c->is_draining) ? 1 : 0;
}

/* Set mongoose's global log level. Maps directly to MG_LL_NONE..VERBOSE
 * (0..4). Default at process start is 3 (DEBUG), which prints every
 * accept/read/write/close. Embedded server callers usually want 1 (ERROR)
 * plus their own Python-side access log. */
void inferna_mg_log_set(int level) {
    mg_log_set(level);
}