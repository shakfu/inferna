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
from types import FrameType
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

from . import _mongoose as _mg  # type: ignore[attr-defined]

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

        if not self.load_model():
            return False

        self._setup_signal_handlers()

        try:
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
            self._logger.info(f"Embedded server started on {listen_addr}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to start server: {e}")
            return False

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
        try:
            if method == "GET":
                if uri == "/health":
                    conn.send_json({"status": "ok"})
                elif uri == "/v1/models":
                    self._handle_models(conn)
                else:
                    conn.send_error(404, "Not Found")
            elif method == "POST":
                if uri == "/v1/chat/completions":
                    self._handle_chat_completions(conn, body)
                elif uri == "/v1/embeddings":
                    self._handle_embeddings(conn, body)
                else:
                    conn.send_error(404, "Not Found")
            else:
                conn.send_error(405, "Method Not Allowed")
        except Exception as e:
            self._logger.error(f"Request handling error: {e}")
            conn.send_error(500, "Internal Server Error")

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
