"""Live HTTP tests: a real server process answering real requests.

Everything else under `tests/test_server*.py` is in-process. `ServerConfig`
round-trips, slots are mocked, and the web UI snapshot is checked as gzip
blobs against the route table. Two tests do start `EmbeddedServer` on a real
socket -- and then sleep and stop it, without sending a request. So the wire
was never exercised: a route that 500s, a content type that regressed, an SSE
stream that opens and never delivers a token, or a shutdown that trips
Metal's `[rsets count]==0` assertion all pass that suite.

These tests run the shipped CLI (`python -m inferna.llama.server`) as a
subprocess and talk to it over TCP, which is also the only way to cover the
mongoose event loop: `wait_for_shutdown()` pumps `mg_mgr_poll` until a signal
arrives, so an in-process test would have to reimplement the loop and would
never touch the CLI's `try/finally: server.stop()` teardown.

Subprocess rather than a poll thread is deliberate for a second reason: the
server holds a llama context and Metal state, and the project's cleanup rule
(CLAUDE.md, `docs/dev/test-cleanup.md`) exists because leaked native contexts
crash the suite on macOS. A child process cannot leak into the pytest process
no matter how it exits, and its exit output becomes evidence --
`test_shutdown_releases_native_state` asserts on it directly.
"""

import json
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from conftest import DEFAULT_MODEL

pytestmark = [pytest.mark.integration, pytest.mark.slow]

ROOT = Path(__file__).resolve().parent.parent

# Model load plus Metal warmup. Generous: a cold first run on a loaded machine
# is much slower than a warm one, and a flaky timeout here reads as a server
# bug to whoever hits it.
STARTUP_TIMEOUT = 180.0
SHUTDOWN_TIMEOUT = 60.0


def _free_port() -> int:
    """Ask the OS for an unused port and hand it straight to the server.

    Inherently racy -- the port is released before the child binds it -- but
    the alternative is a hard-coded port, which collides with a developer's
    own server and fails as EADDRINUSE ("Failed to create HTTP listener").
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _get(url: str, timeout: float = 30.0):
    """GET returning (status, headers, body-bytes); HTTP errors are results."""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def _post_json(url: str, payload: dict, timeout: float = 120.0):
    """POST a JSON body, returning (status, headers, body-bytes)."""
    req = urllib.request.Request(
        url,
        method="POST",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def _spawn(model_path: str, log_path: Path):
    """Start the CLI on a free port, logging to `log_path`.

    The log goes to a file rather than a PIPE: llama.cpp writes megabytes of
    load diagnostics to stderr, which fills a pipe buffer and deadlocks the
    child while the test waits on a request that will never be served.
    """
    port = _free_port()
    cmd = [
        sys.executable,
        "-m",
        "inferna.llama.server",
        "--model",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        "512",
        "--server-type",
        "embedded",
        "--webui",
    ]
    log = open(log_path, "wb")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
    return proc, f"http://127.0.0.1:{port}", log


def _await_ready(proc, base: str, log_path: Path) -> None:
    """Block until /health answers 200, or fail with the child's log tail."""
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        try:
            if _get(f"{base}/health", timeout=5.0)[0] == 200:
                return
        except OSError:
            time.sleep(0.5)
    proc.kill()
    proc.wait(timeout=30)
    tail = log_path.read_text(errors="replace")[-4000:]
    pytest.fail(f"server did not become ready within {STARTUP_TIMEOUT}s\n--- log tail ---\n{tail}")


def _interrupt_and_wait(proc) -> int:
    """Deliver SIGINT (what Ctrl+C sends) and wait for the process to exit."""
    if proc.poll() is not None:
        return proc.returncode
    proc.send_signal(signal.SIGINT)
    try:
        return proc.wait(timeout=SHUTDOWN_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)
        pytest.fail(f"server did not exit within {SHUTDOWN_TIMEOUT}s of SIGINT")


@pytest.fixture(scope="module")
def server(model_path, tmp_path_factory):
    """A running server shared by every request test in this module."""
    if not Path(model_path).exists():
        pytest.skip(f"test model not found: {model_path}")

    log_path = tmp_path_factory.mktemp("server") / "server.log"
    proc, base, log = _spawn(model_path, log_path)
    try:
        _await_ready(proc, base, log_path)
        yield base
    finally:
        _interrupt_and_wait(proc)
        log.close()


class TestJsonEndpoints:
    """The OpenAI-compatible and health routes, over the wire."""

    def test_health(self, server):
        status, _, body = _get(f"{server}/health")
        assert status == 200
        assert json.loads(body) == {"status": "ok"}

    def test_models_lists_the_loaded_model(self, server):
        status, _, body = _get(f"{server}/v1/models")
        assert status == 200
        payload = json.loads(body)
        assert payload["object"] == "list"
        assert len(payload["data"]) == 1
        entry = payload["data"][0]
        # --model-alias defaults to the model file's basename without extension.
        assert entry["id"] == Path(DEFAULT_MODEL).stem
        assert entry["object"] == "model"

    def test_props_reports_slot_and_context_config(self, server):
        """The web UI reads /props on load; it must reflect the real config."""
        status, _, body = _get(f"{server}/props")
        assert status == 200
        payload = json.loads(body)
        assert payload["total_slots"] >= 1
        assert payload["default_generation_settings"]["n_ctx"] == 512

    def test_slots_returns_a_list(self, server):
        status, _, body = _get(f"{server}/slots")
        assert status == 200
        assert isinstance(json.loads(body), list)

    def test_metrics_is_a_prometheus_exposition(self, server):
        """Returned as 200-with-no-series rather than 404, so the UI's scrape
        does not fill the log with route-not-found noise."""
        status, headers, _ = _get(f"{server}/metrics")
        assert status == 200
        assert "text/plain" in headers["Content-Type"]

    def test_unknown_route_is_404(self, server):
        assert _get(f"{server}/no/such/route")[0] == 404


class TestWebUIOverTheWire:
    """The vendored SPA snapshot as a browser actually fetches it.

    `TestWebUIAssets` in test_mserver_embedded.py proves the gzip blobs are
    bundled and that the route table names them. It cannot prove the bytes
    reach a client with the right status, encoding, and content type.
    """

    def test_every_asset_route_serves_its_snapshot(self, server):
        from inferna.llama.server.embedded import _WEBUI_ASSET_TYPES

        routes = {
            "/": "index.html",
            "/index.html": "index.html",
            "/bundle.css": "bundle.css",
            "/bundle.js": "bundle.js",
            "/loading.html": "loading.html",
        }
        for route, asset in routes.items():
            status, headers, body = _get(f"{server}{route}")
            assert status == 200, f"{route} returned {status}"
            assert headers["Content-Type"] == _WEBUI_ASSET_TYPES[asset], route
            assert len(body) > 0, route

    def test_assets_are_served_gzipped(self, server):
        """The snapshot is stored gzipped and sent as-is. urllib does not
        request or inflate gzip on its own, so the declared encoding and a
        gzip magic number together prove the stored blob went out intact."""
        status, headers, body = _get(f"{server}/bundle.css")
        assert status == 200
        assert headers.get("Content-Encoding") == "gzip"
        assert body[:2] == b"\x1f\x8b", "body is not gzip data"

    def test_index_references_the_bundles_it_is_served_with(self, server):
        """index.html hard-references ./bundle.js and ./bundle.css. If a
        future snapshot renames them, the routes still 200 and the UI is a
        blank page -- this is what catches that."""
        import gzip

        body = gzip.decompress(_get(f"{server}/")[2]).decode("utf-8", errors="replace")
        assert "bundle.js" in body
        assert "bundle.css" in body

    def test_asset_routes_tolerate_the_cache_busting_query(self, server):
        """index.html requests `./bundle.js?<hash>`. The router splits the
        query off before matching; without that every page load 404s."""
        status, headers, body = _get(f"{server}/bundle.js?v=deadbeef")
        assert status == 200
        assert headers["Content-Type"] == "application/javascript; charset=utf-8"
        assert len(body) > 0


class TestChatCompletions:
    """Generation through the HTTP layer, in both response shapes."""

    def test_non_streaming_completion(self, server):
        status, _, body = _post_json(
            f"{server}/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": 8,
                "temperature": 0.0,
            },
        )
        assert status == 200
        payload = json.loads(body)
        assert payload["object"] == "chat.completion"
        choice = payload["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)
        assert choice["finish_reason"] in ("stop", "length")
        usage = payload["usage"]
        assert usage["prompt_tokens"] > 0
        assert 0 < usage["completion_tokens"] <= 8
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_streaming_completion_delivers_tokens_and_terminates(self, server):
        """The regression this is really for: streaming runs generation on a
        worker thread that queues SSE frames for the poll thread to flush
        (see `_StreamingState`). If that handoff breaks, the response opens,
        headers arrive, and the body never completes -- which every
        in-process test still passes.
        """
        req = urllib.request.Request(
            f"{server}/v1/chat/completions",
            method="POST",
            data=json.dumps(
                {
                    "messages": [{"role": "user", "content": "Count to three."}],
                    "max_tokens": 12,
                    "temperature": 0.0,
                    "stream": True,
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120.0) as r:
            assert r.status == 200
            raw = r.read().decode("utf-8", errors="replace")

        frames = [ln[len("data: ") :] for ln in raw.splitlines() if ln.startswith("data: ")]
        assert frames, "no SSE data frames received"
        assert frames[-1] == "[DONE]", "stream did not terminate with [DONE]"

        chunks = [json.loads(f) for f in frames[:-1]]
        assert all(c["object"] == "chat.completion.chunk" for c in chunks)
        # The first frame opens the message with the role; the rest carry text.
        assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        assert text.strip(), "streamed deltas carried no content"
        assert chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")

    def test_embeddings_route_rejects_when_disabled(self, server):
        """Embeddings are off in this config; the route must answer with a
        client error rather than 404 or a stack trace."""
        status, _, _ = _post_json(f"{server}/v1/embeddings", {"input": "hello"})
        assert status == 400


def test_shutdown_releases_native_state(model_path, tmp_path):
    """SIGINT must free the llama context before the interpreter exits.

    Runs its own server rather than sharing the module fixture, because the
    assertions are about what the process writes on the way out -- a test
    cannot observe the teardown of a fixture it is using.

    The failure guarded here is not a hang, and exit status does not reveal
    it. `wait_for_shutdown()` returns on signal but does not itself free the
    model, so a teardown that stops calling `stop()` still "exits" -- with
    nanobind reporting leaked LlamaModel/Manager types and ggml-metal
    aborting on `GGML_ASSERT([rsets->data count] == 0)` from a destructor
    that runs after Metal has already torn down. Only the log shows it.
    """
    if not Path(model_path).exists():
        pytest.skip(f"test model not found: {model_path}")

    log_path = tmp_path / "shutdown.log"
    proc, base, log = _spawn(model_path, log_path)
    try:
        _await_ready(proc, base, log_path)
        assert _get(f"{base}/health")[0] == 200
    finally:
        _interrupt_and_wait(proc)
        log.close()

    output = log_path.read_text(errors="replace")
    assert "GGML_ASSERT" not in output, (
        f"ggml assertion fired during shutdown -- a native context outlived backend teardown:\n{output[-2000:]}"
    )
    assert "nanobind: leaked" not in output, f"nanobind reported leaked native objects at exit:\n{output[-2000:]}"
