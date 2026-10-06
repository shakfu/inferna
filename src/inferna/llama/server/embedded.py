"""Embedded HTTP server for inferna.

The native side (``_httplib.cpp``) wraps a cpp-httplib server that runs on
its own thread pool and hands every request to :meth:`EmbeddedServer._dispatch`.
Routing, request parsing, slot management and chat completion handling are
Python.

Public API:
    - ``EmbeddedServer(config)`` with ``start()``, ``stop()``,
      ``wait_for_shutdown()``, ``handle_http_request()``, context-manager
      support.
    - ``HttpResponse``, the per-request response a handler fills in.
    - ``start_embedded_server(model_path, **kwargs)`` convenience.
"""

from __future__ import annotations

import json
import logging
import queue
import signal
import threading
import time
import uuid
from importlib.resources import files as _resource_files
from types import FrameType
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from . import _httplib  # type: ignore[attr-defined]
from .python import (
    ChatChoice,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatRole,
    ServerConfig,
    ServerSlot,
    is_authorized,
    stop_at,
    warn_if_not_loopback,
)


# ---------------------------------------------------------------------------
# Web UI assets
#
# A gzipped snapshot of llama.cpp's web UI is committed under
# ``inferna/llama/server/assets/webui/*.gz`` (vendored — upstream stopped
# shipping the prebuilt SPA in-tree as of b9352; refresh with
# ``manage.py fetch_webui``). We load the bytes once at import and serve them
# with ``Content-Encoding: gzip``.
#
# Asset names mirror upstream's tools/ui/ build output. ``index.html`` is
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
    base = _resource_files("inferna.llama.server").joinpath("assets").joinpath("webui")
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
    from ..decision import DecisionModel
    from ..llama_cpp import LlamaModel

# Signal handler return type — accept any of the three forms Python's
# signal module returns from `signal.signal(...)`: None, an int (e.g.
# SIG_DFL), or a callable.
_SignalHandler = Union[Callable[[int, Optional[FrameType]], Any], int, None]

# (status, content_type, extra_headers, body); body is bytes or an iterator of bytes.
_Reply = Tuple[int, str, Dict[str, str], Union[bytes, Iterator[bytes]]]

_SSE_DONE = b"data: [DONE]\n\n"

# Module-level shutdown flag, set by the signal handler.
_shutdown_requested = False


def _bind_address(host: str) -> Tuple[str, bool]:
    """Return (host, is_ipv6): brackets stripped, family taken from the literal.

    A fixed family keeps "localhost" on 127.0.0.1, as PythonServer binds it.
    """
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host, ":" in host


class HttpResponse:
    """The response to one request. The first ``send_*`` call wins.

    ``status_code`` and ``body_size`` feed the dispatcher's access-log line.
    """

    __slots__ = ("status_code", "body_size", "content_type", "headers", "body")

    def __init__(self) -> None:
        self.status_code: int = 0  # 0 = no response set yet
        self.body_size: int = 0
        self.content_type = "text/plain"
        self.headers: Dict[str, str] = {}
        self.body: Union[bytes, Iterator[bytes]] = b""

    def _set(
        self,
        status_code: int,
        content_type: str,
        body: Union[bytes, Iterator[bytes]],
        headers: Optional[Dict[str, str]] = None,
    ) -> bool:
        if self.status_code:
            return False
        self.status_code = status_code
        self.content_type = content_type
        self.headers = headers or {}
        self.body = body
        self.body_size = len(body) if isinstance(body, bytes) else 0
        return True

    def send_json(self, data: Any, status_code: int = 200) -> bool:
        return self._set(status_code, "application/json", json.dumps(data).encode("utf-8"))

    def send_error(self, status_code: int, message: str) -> bool:
        return self.send_json(
            {"error": {"type": "invalid_request_error", "message": message}},
            status_code,
        )

    def send_text(self, body: str, content_type: str = "text/plain", status_code: int = 200) -> bool:
        return self._set(status_code, content_type, body.encode("utf-8"))

    def send_gzipped(self, body: bytes, content_type: str, status_code: int = 200) -> bool:
        """Send a precompressed (gzip) payload — used for the UI bundle.

        ``Cache-Control`` is short on the HTML shell (so model/template
        changes show up on reload) and long on the CSS/JS bundles.
        """
        cache = "no-cache" if content_type.startswith("text/html") else "public, max-age=3600"
        headers = {"Content-Encoding": "gzip", "Vary": "Accept-Encoding", "Cache-Control": cache}
        return self._set(status_code, content_type, body, headers)

    def send_stream(self, chunks: Iterator[bytes], content_type: str = "text/event-stream") -> bool:
        """Send the bytes ``chunks`` yields, chunked. The server closes ``chunks`` when the response ends."""
        return self._set(200, content_type, chunks, {"Cache-Control": "no-cache"})

    def reply(self) -> _Reply:
        return self.status_code or 500, self.content_type, self.headers, self.body


class EmbeddedServer:
    """Embedded HTTP server for LLM inference using cpp-httplib."""

    def __init__(self, config: ServerConfig) -> None:
        self._config = config
        self._model: Optional["LlamaModel"] = None
        self._embedder: Optional["Embedder"] = None
        self._decision: Optional["DecisionModel"] = None  # set when the model is a supported decision model
        self._slots: List[ServerSlot] = []
        self._free_slots: "queue.Queue[ServerSlot]" = queue.Queue()
        self._logger = logging.getLogger(__name__)
        self._access_logger = logging.getLogger(f"{__name__}.access")
        self._srv = _httplib.Server(config.max_body_bytes)
        self._listen_thread: Optional[threading.Thread] = None
        self._running = False
        # Set by stop(); open streams and slot waits end on it.
        self._stopping = False
        self._signal_received = 0
        # Saved by _setup_signal_handlers, restored by stop(). Without
        # this, the bound `self._signal_handler` method registered with
        # signal.signal() retains a strong reference to `self`, which in
        # turn pins _model / _srv / _slots[*].sampler past stop() —
        # leaking those native objects all the way to interpreter
        # shutdown and tripping a Metal GGML_ASSERT (rsets not empty).
        self._prev_sigint: _SignalHandler = None
        self._prev_sigterm: _SignalHandler = None
        # Windows only; None everywhere else. See _setup_signal_handlers.
        self._prev_sigbreak: _SignalHandler = None

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
            from ..llama_cpp import LlamaModel, ggml_backend_load_all

            # Backends live in separate shared objects in the published wheels; see Embedder.__init__.
            ggml_backend_load_all()
            self._model = LlamaModel(path_model=self._config.model_path)

            from ..decision import load_decision_model

            self._decision = load_decision_model(self._model, self._config.n_ctx, self._logger)
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
            return True
        except Exception as e:
            self._logger.error(f"Failed to load model: {e}")
            return False

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
        # EmbeddedServer instance + LlamaModel + every slot's
        # LlamaContext + LlamaSampler — until interpreter shutdown.
        self._prev_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        self._prev_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
        registered = "SIGINT and SIGTERM"
        # On Windows neither SIGINT nor SIGTERM can be delivered to this
        # process by another one: CTRL_C_EVENT goes to the whole console
        # group, and terminate() is a hard TerminateProcess that runs no
        # handler. Ctrl+Break -- which arrives as SIGBREAK -- is the only
        # signal a supervisor can target at us, so it is the sole route
        # to a graceful shutdown there. SIGBREAK does not exist on POSIX.
        sigbreak = getattr(signal, "SIGBREAK", None)
        if sigbreak is not None:
            self._prev_sigbreak = signal.signal(sigbreak, self._signal_handler)
            registered = "SIGINT, SIGTERM and SIGBREAK"
        self._logger.debug(f"Signal handlers registered for {registered}")

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
        sigbreak = getattr(signal, "SIGBREAK", None)
        if self._prev_sigbreak is not None and sigbreak is not None:
            try:
                signal.signal(sigbreak, self._prev_sigbreak)
            except (ValueError, TypeError):
                pass
            self._prev_sigbreak = None

    # ------------------------------------------------------------- start/stop

    def start(self) -> bool:
        """Load the model, bind, and serve on a background thread."""
        global _shutdown_requested
        if self._running:
            return True
        _shutdown_requested = False
        self._stopping = False

        success = False
        try:
            if not self.load_model():
                return False

            self._setup_signal_handlers()

            warn_if_not_loopback(self._config.host, self._logger, self._config.api_key)
            host, ipv6 = _bind_address(self._config.host)
            self._srv.set_handler(self._dispatch)
            if not self._srv.bind(host, self._config.port, ipv6):
                self._logger.error(f"Failed to bind {self._config.host}:{self._config.port}")
                return False

            self._free_slots = queue.Queue()
            for slot in self._slots:
                self._free_slots.put(slot)

            self._listen_thread = threading.Thread(target=self._srv.listen, name="inferna-http", daemon=True)
            self._listen_thread.start()
            self._running = True
            success = True
            self._logger.info(f"Embedded server started on {self._config.host}:{self._config.port}")
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
                self._srv.set_handler(None)
                self._restore_signal_handlers()
                self._model = None
                self._slots = []
                self._embedder = None

    def stop(self) -> None:
        """Stop serving. Waits for in-flight requests; open streams end at their next token."""
        self._logger.info("Stop method called")
        if self._running:
            self._logger.info("Stopping embedded server...")
            self._stopping = True
            if self._signal_received == 0:
                self._signal_received = signal.SIGTERM
            self._srv.stop()
            if self._listen_thread is not None:
                self._listen_thread.join()
                self._listen_thread = None
            self._running = False
            self._logger.info("Embedded server stopped")
        # The handler is a bound method: the native server would pin self.
        self._srv.set_handler(None)
        # Always restore signal handlers, even if stop() is called twice
        # or before a successful start. The bound-method handler is the
        # main retention path keeping the server (and its native
        # children) alive past test scope.
        self._restore_signal_handlers()

    def wait_for_shutdown(self) -> None:
        """Block until SIGINT/SIGTERM is delivered or stop() is called."""
        self._logger.info("Waiting for shutdown signal...")
        while not _shutdown_requested and self._running:
            time.sleep(0.1)
        self._logger.info(f"Exiting on signal {self._signal_received}")

    # --------------------------------------------------------- HTTP dispatch

    def _dispatch(self, method: str, path: str, headers: dict[str, str], body: bytes) -> _Reply:
        """Answer one request. Runs on an httplib worker thread.

        Emits one access-log line per request:

            GET /props 200 285B 0.4ms
        """
        start = time.monotonic()
        conn = HttpResponse()
        try:
            try:
                text = body.decode("utf-8")
            except UnicodeDecodeError:
                conn.send_error(400, "Request body is not valid UTF-8")
            else:
                self.handle_http_request(conn, method, path, headers=headers, body=text)
        except Exception as e:
            self._logger.exception(f"Event handler error: {e}")
            conn = HttpResponse()
            conn.send_text("Internal Server Error", status_code=500)
        elapsed_ms = (time.monotonic() - start) * 1000.0
        self._access_logger.info("%s %s %d %dB %.1fms", method, path, conn.status_code, conn.body_size, elapsed_ms)
        return conn.reply()

    def handle_http_request(
        self, conn: HttpResponse, method: str, uri: str, headers: dict[str, str], body: str
    ) -> None:
        # Match on the path alone; the webui requests assets with a cache-busting query.
        path = uri.split("?", 1)[0]
        try:
            # ``headers`` keys are lowercase; see _httplib.cpp.
            if not is_authorized(self._config.api_key, path, headers.get("authorization")):
                conn.send_json({"error": {"type": "authentication_error", "message": "Invalid API Key"}}, 401)
                return
            if method == "GET":
                webui = self._config.serve_webui
                if path == "/health":
                    conn.send_json({"status": "ok"})
                elif path == "/v1/models":
                    self._handle_models(conn)
                elif webui and (path == "/" or path == "/index.html"):
                    self._handle_webui_asset(conn, "index.html")
                elif webui and path == "/bundle.css":
                    self._handle_webui_asset(conn, "bundle.css")
                elif webui and path == "/bundle.js":
                    self._handle_webui_asset(conn, "bundle.js")
                elif webui and path == "/loading.html":
                    self._handle_webui_asset(conn, "loading.html")
                elif webui and path == "/props":
                    self._handle_props(conn)
                elif webui and path == "/slots":
                    self._handle_slots(conn)
                elif webui and path == "/metrics":
                    # Prometheus scrape endpoint. The webui calls this but
                    # tolerates an empty exposition; we return 200 with no
                    # series rather than a 404 (which would log noise).
                    conn.send_text("", content_type="text/plain; version=0.0.4")
                else:
                    conn.send_error(404, "Not Found")
            elif method == "POST":
                if path == "/v1/chat/completions":
                    self._handle_chat_completions(conn, body)
                elif path == "/v1/embeddings":
                    self._handle_embeddings(conn, body)
                elif path == "/v1/systemone":
                    self._handle_systemone(conn, body)
                else:
                    conn.send_error(404, "Not Found")
            else:
                conn.send_error(405, "Method Not Allowed")
        except Exception as e:
            self._logger.error(f"Request handling error: {e}")
            conn.send_error(500, "Internal Server Error")

    def _handle_webui_asset(self, conn: HttpResponse, name: str) -> None:
        body = _WEBUI_ASSETS.get(name)
        if body is None:
            conn.send_error(404, f"UI asset {name} not bundled — rebuild with 'make'")
            return
        conn.send_gzipped(body, _WEBUI_ASSET_TYPES[name])

    def _handle_props(self, conn: HttpResponse) -> None:
        """Bootstrap payload consumed by the upstream webui at load time."""
        n_ctx = self._config.n_ctx
        gen_defaults = {
            "n_ctx": n_ctx,
            "temperature": 0.8,
            "top_p": 0.9,
            "min_p": 0.05,
        }
        conn.send_json(
            {
                "default_generation_settings": gen_defaults,
                "total_slots": self._config.n_parallel,
                "model_path": self._config.model_path,
                "model_alias": self._config.model_alias,
                "chat_template": "",  # TODO Phase 4: surface tokenizer's template
                "build_info": "inferna",
                "n_ctx": n_ctx,
                "n_ctx_train": n_ctx,
            }
        )

    def _handle_slots(self, conn: HttpResponse) -> None:
        conn.send_json(
            [
                {
                    "id": s.id,
                    "is_processing": s.is_processing,
                    "task_id": s.task_id,
                }
                for s in self._slots
            ]
        )

    def _handle_models(self, conn: HttpResponse) -> None:
        models_data: Dict[str, Any] = {
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
        if self._decision is not None:
            # clients detect decision models this way, as with llama-server
            models_data["data"][0]["architecture"] = {"output_modalities": ["decisions"]}
        conn.send_json(models_data)

    def _handle_systemone(self, conn: HttpResponse, body: str) -> None:
        """Handle /v1/systemone (decision models)."""
        from ..decision import systemone_response

        try:
            data = json.loads(body) if body.strip() else None
        except json.JSONDecodeError:
            conn.send_error(400, "Invalid JSON")
            return
        status, payload = systemone_response(self._decision, data, self._config.model_alias)
        conn.send_json(payload, status)

    def _handle_chat_completions(self, conn: HttpResponse, body: str) -> None:
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
                min_p=data.get("min_p", 0.05),
                stream=data.get("stream", False),
                stop=data.get("stop"),
                seed=data.get("seed"),
            )
            if request.stream:
                conn.send_stream(self._sse(request))
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
            # The message can carry model and filesystem paths; log it
            # server-side and tell the client nothing specific.
            self._logger.error(f"Chat completion error: {e}")
            conn.send_error(500, "Internal Server Error")

    def _handle_embeddings(self, conn: HttpResponse, body: str) -> None:
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
            conn.send_error(500, "Internal Server Error")

    def _resolve_max_tokens(self, request: ChatRequest) -> int:
        """Map the request's ``max_tokens`` to a concrete cap.

        Honors the llama-server / OpenAI convention: a missing field, ``0``,
        or any negative value (canonically ``-1``) means "generate until EOS
        or context limit." We translate that to ``n_ctx`` as the effective
        cap — the per-slot decode loop already breaks at
        ``n_past >= context.n_ctx - 1``, so this is a hard upper bound that
        the caller will rarely reach. A positive value is honored verbatim.
        """
        n = request.max_tokens
        if n is None or n <= 0:
            return self._config.n_ctx
        return n

    def _acquire_slot(self) -> ServerSlot:
        """Wait for a free slot. Raises RuntimeError if the server stops first."""
        while True:
            try:
                return self._free_slots.get(timeout=0.1)
            except queue.Empty:
                if self._stopping:
                    raise RuntimeError("Server is stopping")

    def _release_slot(self, slot: ServerSlot) -> None:
        slot.reset()
        self._free_slots.put(slot)

    def _sse(self, request: ChatRequest) -> Iterator[bytes]:
        """Yield an OpenAI ``chat.completion.chunk`` event stream.

        Runs as the response body, one chunk per ``next()`` on an httplib
        worker thread. The slot is taken on the first token, so a stream the
        server never starts holds none; closing the generator releases it.
        """
        chunk_id = f"chatcmpl-{uuid.uuid4()}"
        created = int(time.time())
        started_at = time.monotonic()
        body_size = 0
        cancelled = False

        def frame(payload: dict[str, Any]) -> bytes:
            nonlocal body_size
            data = b"data: " + json.dumps(payload).encode() + b"\n\n"
            body_size += len(data)
            return data

        def event(delta: dict[str, Any], finish: Optional[str] = None) -> bytes:
            return frame(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
            )

        max_tokens = self._resolve_max_tokens(request)
        slot: Optional[ServerSlot] = None
        try:
            yield event({"role": ChatRole.ASSISTANT})
            try:
                slot = self._acquire_slot()
                slot.task_id = chunk_id
                slot.is_processing = True
                prompt = self._messages_to_prompt(request.messages)
                stop_hit = False
                generated = 0

                def counted() -> Iterator[str]:
                    nonlocal generated
                    for piece in tokens:
                        generated += 1
                        yield piece

                def content() -> Iterator[str]:
                    nonlocal stop_hit
                    stop_hit = yield from stop_at(counted(), request.stop)

                tokens = slot.iter_tokens(prompt, max_tokens, request)
                try:
                    for piece in content():
                        if self._stopping or _shutdown_requested:
                            break
                        yield event({"content": piece})
                finally:
                    tokens.close()
                finish = "length" if not stop_hit and generated >= max_tokens else "stop"
                yield event({}, finish)
                body_size += len(_SSE_DONE)
                yield _SSE_DONE
            except GeneratorExit:
                cancelled = True
                raise
            except Exception as e:
                self._logger.exception(f"Streaming error: {e}")
                yield frame({"error": {"type": "internal_error", "message": "Internal Server Error"}})
        finally:
            if slot is not None:
                self._release_slot(slot)
            elapsed_ms = (time.monotonic() - started_at) * 1000.0
            self._access_logger.info(
                "%s %s model=%s bytes=%d elapsed=%.1fms",
                "stream-cancel" if cancelled else "stream-done",
                chunk_id,
                request.model,
                body_size,
                elapsed_ms,
            )

    def _process_chat_completion(self, request: ChatRequest) -> ChatResponse:
        slot = self._acquire_slot()
        try:
            task_id = str(uuid.uuid4())
            slot.task_id = task_id
            slot.is_processing = True

            prompt = self._messages_to_prompt(request.messages)
            max_tokens = self._resolve_max_tokens(request)
            generated_text = slot.process_and_generate(prompt, max_tokens, request)
            generated_text = "".join(stop_at([generated_text], request.stop))

            assert self._model is not None  # _generate_chat_response is only entered after load_model succeeded
            vocab = self._model.get_vocab()
            prompt_tokens = len(vocab.tokenize(prompt, add_special=True, parse_special=True))
            completion_tokens = len(vocab.tokenize(generated_text, add_special=False, parse_special=False))

            choice = ChatChoice(
                index=0,
                message=ChatMessage(role=ChatRole.ASSISTANT, content=generated_text),
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
            self._release_slot(slot)

    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        parts = []
        for m in messages:
            if m.role == ChatRole.SYSTEM:
                parts.append(f"System: {m.content}")
            elif m.role == ChatRole.USER:
                parts.append(f"User: {m.content}")
            elif m.role == ChatRole.ASSISTANT:
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
