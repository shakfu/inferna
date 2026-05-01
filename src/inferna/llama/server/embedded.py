"""Embedded HTTP server for inferna.

This module used to be ``embedded.pyx`` — a Cython file that mostly held
Python code with a thin native shim around mongoose. The C++ side now lives
in ``_mongoose.cpp`` and exposes only the mg_mgr lifecycle + HTTP reply
primitives. Everything else (routing, request parsing, slot management,
chat completion handling, signal wiring, and the EmbeddedServer class
itself) stays Python.

Public API (preserved for callers and tests):
    - ``EmbeddedServer(config)`` with ``start()``, ``stop()``,
      ``wait_for_shutdown()``, ``handle_http_request()``, context-manager
      support.
    - ``MongooseConnection`` wrapper for per-request response writing.
    - ``start_embedded_server(model_path, **kwargs)`` convenience.
"""

from __future__ import annotations

import json
import logging
import signal
import time
from importlib.resources import files as _resource_files
from types import FrameType
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

from . import _mongoose as _mg  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Web UI assets
#
# The build hook in scripts/manage.py copies llama.cpp's prebuilt server SPA
# into ``inferna/llama/server/assets/webui/*.gz`` (gzipped at build time —
# the package always ships the compressed form). We load the bytes once at
# import and serve them with ``Content-Encoding: gzip``.
#
# Asset names mirror upstream's tools/server/public/. ``index.html`` is
# also exposed at ``/`` so a bare visit to the server lands on the UI.
# ---------------------------------------------------------------------------

_WEBUI_ASSET_TYPES: dict[str, str] = {
    "index.html": "text/html; charset=utf-8",
    "bundle.css": "text/css; charset=utf-8",
    "bundle.js": "application/javascript; charset=utf-8",
    "loading.html": "text/html; charset=utf-8",
}


def _load_webui_assets() -> dict[str, bytes]:
    """Read the gzipped UI bundle into memory (called once per process).

    Returns ``{"index.html": <gz bytes>, ...}``. Missing files are silently
    omitted — at request time we 404 the corresponding route. This lets a
    dev who hasn't run ``make`` yet still use the JSON API endpoints.
    """
    out: dict[str, bytes] = {}
    base = _resource_files("inferna.llama.server").joinpath("assets", "webui")
    for name in _WEBUI_ASSET_TYPES:
        gz = base.joinpath(f"{name}.gz")
        try:
            out[name] = gz.read_bytes()
        except (FileNotFoundError, OSError):
            continue
    return out


_WEBUI_ASSETS: dict[str, bytes] = _load_webui_assets()

if TYPE_CHECKING:
    from ...rag.embedder import Embedder
    from ..llama_cpp import LlamaModel

from .python import (
    ServerConfig,
    ServerSlot,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatChoice,
)

# Signal handler return type — accept any of the three forms Python's
# signal module returns from `signal.signal(...)`: None, an int (e.g.
# SIG_DFL), or a callable.
_SignalHandler = Union[Callable[[int, Optional[FrameType]], Any], int, None]


# Module-level shutdown flag (matches pymongoose pattern from the old .pyx)
_shutdown_requested = False


class MongooseConnection:
    """Per-request response writer; wraps an opaque mongoose connection id."""

    __slots__ = ("_conn_id", "_mgr")

    def __init__(self, mgr: Optional[Any] = None, conn_id: int = 0) -> None:
        self._mgr = mgr
        self._conn_id = conn_id

    @property
    def is_valid(self) -> bool:
        return self._conn_id != 0 and self._mgr is not None

    def send_json(self, data: dict[str, Any], status_code: int = 200) -> bool:
        if not self.is_valid or self._mgr is None:
            return False
        body = json.dumps(data)
        return self._mgr.send_reply(self._conn_id, status_code, "Content-Type: application/json\r\n", body)

    def send_error(self, status_code: int, message: str) -> bool:
        return self.send_json(
            {"error": {"type": "invalid_request_error", "message": message}},
            status_code,
        )

    def send_gzipped(self, body: bytes, content_type: str, status_code: int = 200) -> bool:
        """Send a precompressed (gzip) payload — used for the UI bundle.

        We add ``Vary: Accept-Encoding`` for correctness even though every
        modern browser accepts gzip; ``Cache-Control`` is short on the
        HTML shell (so model/template changes show up on reload) and long
        on the immutable CSS/JS bundles (content-hashed by the upstream
        build).
        """
        if not self.is_valid or self._mgr is None:
            return False
        cache = "no-cache" if content_type.startswith("text/html") else "public, max-age=3600"
        headers = (
            f"Content-Type: {content_type}\r\n"
            f"Content-Encoding: gzip\r\n"
            f"Vary: Accept-Encoding\r\n"
            f"Cache-Control: {cache}\r\n"
        )
        return self._mgr.send_bytes(self._conn_id, status_code, headers, body)


class EmbeddedServer:
    """High-performance embedded HTTP server for LLM inference using Mongoose."""

    def __init__(self, config: ServerConfig) -> None:
        self._config = config
        self._model: Optional["LlamaModel"] = None
        self._embedder: Optional["Embedder"] = None
        self._slots: List[ServerSlot] = []
        self._logger = logging.getLogger(__name__)
        self._mgr = _mg.Manager()
        self._running = False
        self._signal_received = 0
        # Saved by _setup_signal_handlers, restored by stop(). Without
        # this, the bound `self._signal_handler` method registered with
        # signal.signal() retains a strong reference to `self`, which in
        # turn pins _model / _mgr / _slots[*].sampler past stop() —
        # leaking those native objects all the way to interpreter
        # shutdown and tripping a Metal GGML_ASSERT (rsets not empty).
        self._prev_sigint: _SignalHandler = None
        self._prev_sigterm: _SignalHandler = None

    # ------------------------------------------------------------------ props

    @property
    def signal_received(self) -> int:
        """Get the received signal number (0 if no signal)."""
        return self._signal_received

    @signal_received.setter
    def signal_received(self, value: int) -> None:
        self._signal_received = int(value)

    # ------------------------------------------------------------- lifecycle

    def __enter__(self) -> "EmbeddedServer":
        if self.start():
            return self
        raise RuntimeError("Failed to start embedded server")

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        self._logger.info("Context manager __exit__ called - starting graceful shutdown")
        self.stop()
        self._logger.info("Context manager __exit__ completed")

    # ------------------------------------------------------------ model load

    def load_model(self) -> bool:
        try:
            self._logger.info(f"Loading model: {self._config.model_path}")
            from ..llama_cpp import LlamaModel

            self._model = LlamaModel(path_model=self._config.model_path)
            self._slots = [ServerSlot(i, self._model, self._config) for i in range(self._config.n_parallel)]
            self._logger.info(f"Model loaded successfully with {len(self._slots)} slots")

            if self._config.embedding:
                from ...rag.embedder import Embedder

                emb_model = self._config.embedding_model_path or self._config.model_path
                self._embedder = Embedder(
                    model_path=emb_model,
                    n_ctx=self._config.embedding_n_ctx,
                    n_batch=self._config.embedding_n_batch,
                    n_gpu_layers=self._config.embedding_n_gpu_layers,
                    pooling=self._config.embedding_pooling,
                    normalize=self._config.embedding_normalize,
                )
                self._logger.info(f"Embedder loaded: dim={self._embedder.dimension}, pooling={self._embedder.pooling}")
            self._logger.info("About to return True from load_model()")
            return True
        except Exception as e:
            self._logger.error(f"Failed to load model: {e}")
            return False

    def get_available_slot(self) -> Optional[ServerSlot]:
        for slot in self._slots:
            if not slot.is_processing:
                return slot
        return None

    # ----------------------------------------------------------- signal API

    def _signal_handler(self, signum: int, frame: Optional[FrameType]) -> None:
        global _shutdown_requested
        self._logger.info(f"Received signal {signum}, requesting graceful shutdown...")
        _shutdown_requested = True
        self._signal_received = signum

    def _setup_signal_handlers(self) -> None:
        # Save the previous handlers so stop() can restore them. If we
        # didn't restore, the signal module would keep our bound
        # `self._signal_handler` alive — and through it, the entire
        # EmbeddedServer instance + Manager + LlamaModel + every slot's
        # LlamaContext + LlamaSampler — until interpreter shutdown.
        self._prev_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        self._prev_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
        self._logger.debug("Signal handlers registered for SIGINT and SIGTERM")

    def _restore_signal_handlers(self) -> None:
        # Only restore if we actually installed our handler. Calling
        # stop() without a prior start() should be a no-op.
        if self._prev_sigint is not None:
            try:
                signal.signal(signal.SIGINT, self._prev_sigint)
            except (ValueError, TypeError):
                # signal.signal raises ValueError when called from a
                # non-main thread, and TypeError on some prev-handler
                # sentinel values. Both are fatal-only at process exit,
                # which is exactly when we don't care.
                pass
            self._prev_sigint = None
        if self._prev_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, self._prev_sigterm)
            except (ValueError, TypeError):
                pass
            self._prev_sigterm = None

    # ------------------------------------------------------------- start/stop

    def start(self) -> bool:
        global _shutdown_requested
        _shutdown_requested = False

        success = False
        try:
            if not self.load_model():
                return False

            self._setup_signal_handlers()

            host = self._config.host
            if host in ("127.0.0.1", "localhost"):
                listen_addr = f"http://0.0.0.0:{self._config.port}"
            else:
                listen_addr = f"http://{host}:{self._config.port}"
            self._logger.info(f"Attempting to bind to: {listen_addr}")

            # Wire the request dispatcher before listening.
            self._mgr.set_handler(self._dispatch)
            self._logger.info("Calling inferna_mg_http_listen...")
            ok = self._mgr.listen(listen_addr)
            self._logger.info(f"inferna_mg_http_listen ok={ok}")
            if not ok:
                self._logger.error("Failed to create HTTP listener")
                return False
            self._running = True
            success = True
            self._logger.info(f"Embedded server started on {listen_addr}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to start server: {e}")
            return False
        finally:
            if not success:
                # Undo every retention path installed during start() so a
                # failed bring-up does not pin native state to interpreter
                # shutdown — leaked LlamaContext + Metal teardown order
                # trips a [rsets count]==0 assertion.
                self._mgr.set_handler(None)
                self._restore_signal_handlers()
                self._model = None
                self._slots = []
                self._embedder = None

    def stop(self) -> None:
        self._logger.info("Stop method called")
        if self._running:
            self._logger.info("Stopping embedded server...")
            self._running = False
            if self._signal_received == 0:
                self._signal_received = signal.SIGTERM
            self._close_all_connections_from_main_thread()
            self._mgr.set_handler(None)
            self._logger.info("Embedded server stopped")
        # Always restore signal handlers, even if stop() is called twice
        # or before a successful start. The bound-method handler is the
        # main retention path keeping the server (and its native
        # children) alive past test scope.
        self._restore_signal_handlers()

    def wait_for_shutdown(self) -> None:
        """Pump the event loop until SIGINT/SIGTERM is delivered."""
        global _shutdown_requested
        self._logger.info("Starting embedded server event loop...")
        while not _shutdown_requested:
            self._mgr.poll(100)  # 100ms — same cadence as the prior pyx
        self._logger.info(f"Exiting on signal {self._signal_received}")
        self._close_all_connections_from_main_thread()

    def _close_all_connections_from_main_thread(self) -> None:
        self._logger.info("Closing all Mongoose connections from main thread...")
        n = self._mgr.close_all_connections()
        self._logger.info(f"Set closing flag on {n} connections")

    # --------------------------------------------------------- HTTP dispatch

    def _dispatch(self, conn_id: int, method: str, uri: str, body: str) -> None:
        """Bridge from the C event handler into our Python routing."""
        try:
            conn = MongooseConnection(self._mgr, conn_id)
            self.handle_http_request(conn, method, uri, headers={}, body=body)
        except Exception as e:
            self._logger.error(f"Event handler error: {e}")
            self._mgr.send_reply(conn_id, 500, "Content-Type: text/plain\r\n", "Internal Server Error")

    def handle_http_request(
        self, conn: MongooseConnection, method: str, uri: str, headers: dict[str, str], body: str
    ) -> None:
        # Strip query string — mongoose hands us the raw URI. We don't act
        # on query params yet, but we don't want them to defeat path matching.
        path = uri.split("?", 1)[0]
        try:
            if method == "GET":
                if path == "/" or path == "/index.html":
                    self._handle_webui_asset(conn, "index.html")
                elif path == "/bundle.css":
                    self._handle_webui_asset(conn, "bundle.css")
                elif path == "/bundle.js":
                    self._handle_webui_asset(conn, "bundle.js")
                elif path == "/loading.html":
                    self._handle_webui_asset(conn, "loading.html")
                elif path == "/health":
                    conn.send_json({"status": "ok"})
                elif path == "/props":
                    self._handle_props(conn)
                elif path == "/slots":
                    self._handle_slots(conn)
                elif path == "/metrics":
                    # Prometheus scrape endpoint. The webui calls this but
                    # tolerates an empty exposition; we return 200 with no
                    # series rather than a 404 (which would log noise).
                    if conn._mgr is not None:
                        conn._mgr.send_reply(conn._conn_id, 200,
                                              "Content-Type: text/plain; version=0.0.4\r\n", "")
                elif path == "/v1/models":
                    self._handle_models(conn)
                else:
                    conn.send_error(404, "Not Found")
            elif method == "POST":
                if path == "/v1/chat/completions":
                    self._handle_chat_completions(conn, body)
                elif path == "/v1/embeddings":
                    self._handle_embeddings(conn, body)
                else:
                    conn.send_error(404, "Not Found")
            else:
                conn.send_error(405, "Method Not Allowed")
        except Exception as e:
            self._logger.error(f"Request handling error: {e}")
            conn.send_error(500, "Internal Server Error")

    def _handle_webui_asset(self, conn: MongooseConnection, name: str) -> None:
        body = _WEBUI_ASSETS.get(name)
        if body is None:
            conn.send_error(404, f"UI asset {name} not bundled — rebuild with 'make'")
            return
        conn.send_gzipped(body, _WEBUI_ASSET_TYPES[name])

    def _handle_props(self, conn: MongooseConnection) -> None:
        """Bootstrap payload consumed by the upstream webui at load time."""
        n_ctx = self._config.n_ctx
        gen_defaults = {
            "n_ctx": n_ctx,
            "temperature": 0.8,
            "top_p": 0.9,
            "min_p": 0.05,
        }
        conn.send_json({
            "default_generation_settings": gen_defaults,
            "total_slots": self._config.n_parallel,
            "model_path": self._config.model_path,
            "model_alias": self._config.model_alias,
            "chat_template": "",  # TODO Phase 4: surface tokenizer's template
            "build_info": "inferna",
            "n_ctx": n_ctx,
            "n_ctx_train": n_ctx,
        })

    def _handle_slots(self, conn: MongooseConnection) -> None:
        conn.send_json([
            {
                "id": s.id,
                "is_processing": s.is_processing,
                "task_id": s.task_id,
            }
            for s in self._slots
        ])

    def _handle_models(self, conn: MongooseConnection) -> None:
        models_data = {
            "object": "list",
            "data": [
                {
                    "id": self._config.model_alias,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "inferna",
                }
            ],
        }
        conn.send_json(models_data)

    def _handle_chat_completions(self, conn: MongooseConnection, body: str) -> None:
        try:
            if not body.strip():
                conn.send_error(400, "Empty request body")
                return
            data = json.loads(body)
            messages_data = data.get("messages", [])
            messages = [ChatMessage(role=m["role"], content=m["content"]) for m in messages_data]
            request = ChatRequest(
                messages=messages,
                model=data.get("model", self._config.model_alias),
                max_tokens=data.get("max_tokens"),
                temperature=data.get("temperature", 0.8),
                top_p=data.get("top_p", 0.9),
                stream=data.get("stream", False),
                stop=data.get("stop"),
            )
            if request.stream:
                self._stream_chat_completion(conn, request)
                return
            response = self._process_chat_completion(request)
            response_data = {
                "id": response.id,
                "object": response.object,
                "created": response.created,
                "model": response.model,
                "choices": [
                    {
                        "index": c.index,
                        "message": {"role": c.message.role, "content": c.message.content},
                        "finish_reason": c.finish_reason,
                    }
                    for c in response.choices
                ],
                "usage": response.usage,
            }
            conn.send_json(response_data)
        except json.JSONDecodeError:
            conn.send_error(400, "Invalid JSON")
        except Exception as e:
            self._logger.error(f"Chat completion error: {e}")
            conn.send_error(500, str(e))

    def _handle_embeddings(self, conn: MongooseConnection, body: str) -> None:
        if not self._config.embedding or self._embedder is None:
            conn.send_error(400, "Embeddings not enabled")
            return
        try:
            if not body.strip():
                conn.send_error(400, "Empty request body")
                return
            data = json.loads(body)
            input_data = data.get("input")
            if input_data is None:
                conn.send_error(400, "Missing 'input' field")
                return
            if isinstance(input_data, str):
                texts = [input_data]
            elif isinstance(input_data, list):
                texts = [str(t) for t in input_data]
            else:
                conn.send_error(400, "Invalid 'input' field: must be string or list of strings")
                return
            model_name = data.get("model", self._config.model_alias)
            results = []
            total_tokens = 0
            for i, text in enumerate(texts):
                result = self._embedder.embed_with_info(text)
                results.append({"object": "embedding", "embedding": result.embedding, "index": i})
                total_tokens += result.token_count
            conn.send_json(
                {
                    "object": "list",
                    "data": results,
                    "model": model_name,
                    "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
                }
            )
        except json.JSONDecodeError:
            conn.send_error(400, "Invalid JSON")
        except Exception as e:
            self._logger.error(f"Embeddings error: {e}")
            conn.send_error(500, str(e))

    def _stream_chat_completion(self, conn: MongooseConnection, request: ChatRequest) -> None:
        """SSE streaming branch for ``POST /v1/chat/completions``.

        Emits OpenAI-style chat-completion-chunk JSON deltas, one per
        token, terminated with ``data: [DONE]\\n\\n``. Cancellation:
        between every token we ask mongoose whether the client is still
        connected — if not, we break the loop and reset the slot. This
        accepts up-to-one-token latency on cancel detection (see Phase 1
        design notes).
        """
        import uuid

        slot = self.get_available_slot()
        if slot is None:
            # No chunked-stream framing yet — return a regular JSON error.
            conn.send_error(503, "No available slots")
            return

        task_id = str(uuid.uuid4())
        slot.task_id = task_id
        slot.is_processing = True

        # The first chunk in OpenAI's SSE shape carries the assistant role
        # delta (no content), subsequent chunks carry content deltas, and
        # the final non-DONE chunk carries finish_reason.
        chunk_id = f"chatcmpl-{task_id}"
        created = int(time.time())
        mgr = conn._mgr
        conn_id = conn._conn_id

        def _frame(delta: dict[str, Any], finish: Optional[str] = None) -> bytes:
            payload = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return b"data: " + json.dumps(payload).encode() + b"\n\n"

        try:
            assert mgr is not None
            ok = mgr.begin_chunked(
                conn_id, 200,
                "Content-Type: text/event-stream\r\n"
                "Cache-Control: no-cache\r\n"
                "Connection: keep-alive\r\n",
            )
            if not ok:
                return  # connection already gone

            # Role-only opening delta (matches OpenAI behavior).
            mgr.send_chunk(conn_id, _frame({"role": "assistant"}))

            prompt = self._messages_to_prompt(request.messages)
            max_tokens = request.max_tokens or 100

            stop_hit = False
            buffered = ""  # accumulated to detect stop sequences across token boundaries
            finish_reason = "stop"
            generated_count = 0
            for piece in slot.iter_tokens(prompt, max_tokens, request):
                # Disconnect check — accepts up-to-one-token latency.
                if not mgr.is_connection_alive(conn_id):
                    return  # bail without [DONE]; client is gone
                if _shutdown_requested:
                    finish_reason = "stop"
                    break

                # Stop-sequence handling: if the user supplied stop strings
                # we accumulate the tail of recent output, look for any of
                # them, and truncate to the prefix before the match.
                if request.stop:
                    buffered += piece
                    matched_at = -1
                    for sw in request.stop:
                        idx = buffered.find(sw)
                        if idx != -1 and (matched_at == -1 or idx < matched_at):
                            matched_at = idx
                    if matched_at != -1:
                        # Emit only the portion of `buffered` before the
                        # stop word that hadn't been streamed in earlier
                        # pieces. Conservatively, we just emit nothing
                        # more after the match — the previous deltas have
                        # already gone over the wire, so partial inclusion
                        # of the stop string is acceptable (matches
                        # llama-server's behavior).
                        stop_hit = True
                        break

                if not mgr.send_chunk(conn_id, _frame({"content": piece})):
                    return
                generated_count += 1

            if not stop_hit and generated_count >= max_tokens:
                finish_reason = "length"

            # Closing chunk with finish_reason, then [DONE], then end the
            # chunked transfer.
            mgr.send_chunk(conn_id, _frame({}, finish=finish_reason))
            mgr.send_chunk(conn_id, b"data: [DONE]\n\n")
            mgr.end_chunked(conn_id)
        except Exception as e:
            # Mid-stream errors can't be turned into a clean HTTP error
            # (we've already sent 200 + headers). Best-effort: emit an
            # error event and close.
            self._logger.error(f"Streaming chat completion error: {e}")
            try:
                if mgr is not None:
                    mgr.send_chunk(conn_id, b"data: " + json.dumps(
                        {"error": {"type": "internal_error", "message": str(e)}}
                    ).encode() + b"\n\n")
                    mgr.end_chunked(conn_id)
            except Exception:
                pass
        finally:
            slot.reset()

    def _process_chat_completion(self, request: ChatRequest) -> ChatResponse:
        slot = self.get_available_slot()
        if slot is None:
            raise RuntimeError("No available slots")
        try:
            import uuid

            task_id = str(uuid.uuid4())
            slot.task_id = task_id
            slot.is_processing = True

            prompt = self._messages_to_prompt(request.messages)
            max_tokens = request.max_tokens or 100
            generated_text = slot.process_and_generate(prompt, max_tokens)

            if request.stop and generated_text:
                for stop_word in request.stop:
                    if stop_word in generated_text:
                        generated_text = generated_text.split(stop_word)[0]
                        break

            assert self._model is not None  # _generate_chat_response is only entered after load_model succeeded
            vocab = self._model.get_vocab()
            prompt_tokens = len(vocab.tokenize(prompt, add_special=True, parse_special=True))
            completion_tokens = len(vocab.tokenize(generated_text, add_special=False, parse_special=False))

            choice = ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=generated_text),
                finish_reason="stop",
            )
            return ChatResponse(
                id=task_id,
                model=request.model,
                choices=[choice],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            )
        finally:
            slot.reset()

    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        parts = []
        for m in messages:
            if m.role == "system":
                parts.append(f"System: {m.content}")
            elif m.role == "user":
                parts.append(f"User: {m.content}")
            elif m.role == "assistant":
                parts.append(f"Assistant: {m.content}")
        parts.append("Assistant:")
        return "\n".join(parts)


def start_embedded_server(model_path: str, **kwargs: Any) -> EmbeddedServer:
    """Convenience: build a config + EmbeddedServer and start it."""
    config = ServerConfig(model_path=model_path, **kwargs)
    server = EmbeddedServer(config)
    if not server.start():
        raise RuntimeError("Failed to start embedded server")
    return server
