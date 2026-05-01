# Inferna Server Usage Examples

Inferna ships an embedded OpenAI-compatible HTTP server with a built-in chat web UI (a rebrand of llama.cpp's reference webui), plus a pure-Python fallback for environments without the compiled mongoose extension.

## Server Types

### 1. Embedded Server (`EmbeddedServer`)

- C networking via the [Mongoose](https://mongoose.ws/) library, bound to Python through nanobind
- Single-threaded poll loop on the main thread; per-stream worker threads for concurrent token generation
- Serves the upstream [llama-server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server/webui) web UI at `GET /` (gzipped at build time, served with `Content-Encoding: gzip`)
- Compiled as part of the standard `make build`

### 2. Python Server (`PythonServer`)

- Pure Python HTTP server (stdlib `http.server`)
- No compiled mongoose dependency
- Useful for development and environments where the embedded extension can't be loaded

## Basic Usage

### Start the Embedded Server (default)

```bash
python -m inferna.llama.server -m models/Llama-3.2-1B-Instruct-Q8_0.gguf
```

Open http://127.0.0.1:8080/ in a browser to use the chat UI.

### Use the Python Server

```bash
python -m inferna.llama.server -m models/Llama-3.2-1B-Instruct-Q8_0.gguf --server-type python
```

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `-m, --model` | (required) | Path to a `.gguf` model file |
| `--host` | `127.0.0.1` | Bind address (use `0.0.0.0` to expose on the LAN) |
| `--port` | `8080` | Port to listen on |
| `--ctx-size` | `2048` | Context window size in tokens |
| `--gpu-layers` | `-1` | GPU layers to offload (-1 = all) |
| `--n-parallel` | `1` | Number of concurrent processing slots |
| `--model-alias` | (filename stem) | Identifier shown in the UI's "Model" field and `/v1/models[].id`. Defaults to the model filename without extension. |
| `--mongoose-log-level` | `1` (errors only) | Mongoose internal log verbosity. `0`=none, `1`=errors only (default), `2`=info, `3`=debug (every accept/read/write/close — useful for HTTP-level debugging), `4`=verbose |
| `--server-type` | `embedded` | `embedded` or `python` |

## Advanced Configuration

### LAN-accessible server with multiple slots

```bash
python -m inferna.llama.server \
    -m models/Llama-3.2-1B-Instruct-Q8_0.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    --n-parallel 4 \
    --ctx-size 4096
```

### GPU offload + custom display name

```bash
python -m inferna.llama.server \
    -m models/Llama-3.2-1B-Instruct-Q8_0.gguf \
    --gpu-layers 32 \
    --ctx-size 4096 \
    --model-alias my-llama
```

### Verbose mongoose tracing for HTTP debugging

```bash
python -m inferna.llama.server -m models/llama.gguf --mongoose-log-level 3
```

## Logging

By default the server emits one access-log line per HTTP request on the `inferna.llama.server.embedded.access` stdlib logger:

```
INFO:inferna.llama.server.embedded.access:GET /props 200 285B 0.1ms
INFO:inferna.llama.server.embedded.access:POST /v1/chat/completions 200 242B 0.4ms
INFO:inferna.llama.server.embedded.access:stream-done conn=14a82e4f0 model=Llama-3.2-1B-Instruct-Q8_0 bytes=8563 elapsed=917.2ms
```

Streaming chat completions emit two lines: one when the dispatcher returns (covers headers + the role-only opener), and a second `stream-done` (or `stream-cancel` if the client dropped) when the SSE stream finishes, with the cumulative byte count and end-to-end timing.

Mongoose's built-in tracer is silenced by default — raise it with `--mongoose-log-level 3` if you need to see the underlying connection events.

## Web UI

Open the server's root URL (`/`) in a browser. The UI is the upstream llama-server SPA, served from the inferna package as gzipped static assets:

| Route | Content |
|---|---|
| `GET /` and `/index.html` | UI shell (HTML) |
| `GET /bundle.css` | UI stylesheet |
| `GET /bundle.js` | UI bundle (Svelte SPA, ~6.5 MB raw / ~1.7 MB gzipped) |
| `GET /loading.html` | UI loading screen |

The UI calls `/props`, `/v1/models`, and `/v1/chat/completions` (with `stream: true`) to operate. Cancellation, conversation history (in IndexedDB), and per-conversation parameter overrides are upstream features that work out of the box.

## API Endpoints

### Health

```bash
curl http://127.0.0.1:8080/health
```

### Server properties (UI bootstrap)

```bash
curl http://127.0.0.1:8080/props | python3 -m json.tool
```

Returns model metadata the UI uses to render its sidebar:

```json
{
  "default_generation_settings": {"n_ctx": 2048, "temperature": 0.8, "top_p": 0.9, "min_p": 0.05},
  "total_slots": 1,
  "model_path": "models/Llama-3.2-1B-Instruct-Q8_0.gguf",
  "model_alias": "Llama-3.2-1B-Instruct-Q8_0",
  "chat_template": "",
  "build_info": "inferna",
  "n_ctx": 2048,
  "n_ctx_train": 2048
}
```

### Slots

```bash
curl http://127.0.0.1:8080/slots
```

```json
[{"id": 0, "is_processing": false, "task_id": null}]
```

### Metrics

```bash
curl http://127.0.0.1:8080/metrics
```

Returns an empty Prometheus exposition (200 with `Content-Type: text/plain; version=0.0.4`). The UI calls this; populating it with real series is a future enhancement.

### Models

```bash
curl http://127.0.0.1:8080/v1/models
```

### Chat completion (non-streaming)

```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 100
  }'
```

### Chat completion (streaming)

```bash
curl -N -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'
```

The response is OpenAI-shape Server-Sent Events:

```
data: {"id":"chatcmpl-...","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}],...}

data: {"id":"chatcmpl-...","choices":[{"index":0,"delta":{"content":" 1"},"finish_reason":null}],...}

data: {"id":"chatcmpl-...","choices":[{"index":0,"delta":{"content":", 2"},"finish_reason":null}],...}

...

data: {"id":"chatcmpl-...","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],...}

data: [DONE]
```

Tokens arrive on the wire as they're generated (worker-thread + main-poll-loop drain), so SSE clients see real-time streaming.

`max_tokens` defaults to "until EOS or context limit" when omitted, `null`, `0`, or any negative value (the upstream webui's `-1` "unlimited" convention is honored). Pass a positive integer to cap.

### Embeddings

The `/v1/embeddings` endpoint is available when the server is started with `embedding=True` in `ServerConfig` (Python API; the CLI does not yet expose this flag).

```bash
# Single text
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "hello world", "model": "nomic-embed-text"}'

# Batch input
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["first text", "second text"], "model": "nomic-embed-text"}'
```

Response format:

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "embedding": [0.1, 0.2, ...], "index": 0}
  ],
  "model": "nomic-embed-text",
  "usage": {"prompt_tokens": 3, "total_tokens": 3}
}
```

## Embedding Server Configuration

To serve embeddings, enable the `embedding` flag on `ServerConfig` and optionally specify a dedicated embedding model:

```python
from inferna.llama.server.python import ServerConfig
from inferna.llama.server.embedded import EmbeddedServer

config = ServerConfig(
    model_path="models/Llama-3.2-1B-Instruct-Q8_0.gguf",
    embedding=True,

    # Optional: dedicated embedding model (defaults to model_path)
    embedding_model_path="models/bge-small-en-v1.5-q8_0.gguf",

    # Embedding-specific parameters (same options as the Embedder class)
    embedding_n_ctx=512,          # Context size (match model training)
    embedding_n_batch=512,
    embedding_n_gpu_layers=-1,    # -1 = all
    embedding_pooling="mean",     # mean, cls, last, none
    embedding_normalize=True,
)

with EmbeddedServer(config) as server:
    server.wait_for_shutdown()
```

The same config works with `PythonServer`:

```python
from inferna.llama.server.python import PythonServer

with PythonServer(config) as server:
    import time
    while True:
        time.sleep(1)
```

### Embedding Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding` | `False` | Enable `/v1/embeddings` endpoint |
| `embedding_model_path` | `None` | Path to embedding model (uses `model_path` if `None`) |
| `embedding_n_ctx` | `512` | Context size for embedding model |
| `embedding_n_batch` | `512` | Batch size for embedding model |
| `embedding_n_gpu_layers` | `-1` | GPU layers for embedding model (-1 = all) |
| `embedding_pooling` | `"mean"` | Pooling strategy: `mean`, `cls`, `last`, `none` |
| `embedding_normalize` | `True` | L2 normalize output embeddings |

## When to Use Each Server

### Use `EmbeddedServer` (default) when

- You want the built-in web UI
- Multiple concurrent users / streams
- Production deployments
- Throughput matters

### Use `PythonServer` when

- The compiled mongoose extension isn't available (sdist install on a platform without a wheel, etc.)
- Debugging the HTTP layer with stdlib tooling
- You don't need the web UI

The two servers expose the same JSON API (`/v1/models`, `/v1/chat/completions`, `/v1/embeddings`, `/health`); only `EmbeddedServer` serves the web UI and the `/props` / `/slots` / `/metrics` endpoints the UI requires.
