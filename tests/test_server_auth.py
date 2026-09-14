"""Model-free tests for the optional API key on the inference servers."""

import json
import socket
import sys
import urllib.error
import urllib.request
from unittest.mock import Mock

import pytest

from inferna.llama.server.python import PUBLIC_PATHS, PythonServer, ServerConfig, is_authorized

KEY = "s3cret"


@pytest.mark.parametrize(
    "api_key, path, authorization, expected",
    [
        (None, "/v1/models", None, True),
        ("", "/v1/models", None, True),
        (KEY, "/v1/models", None, False),
        (KEY, "/v1/models", "", False),
        (KEY, "/v1/models", f"Bearer {KEY}", True),
        (KEY, "/v1/models", f"bearer {KEY}", True),
        (KEY, "/v1/models", f"Bearer  {KEY} ", True),
        (KEY, "/v1/models", "Bearer wrong", False),
        (KEY, "/v1/models", f"Basic {KEY}", False),
        (KEY, "/v1/models", KEY, False),
        (KEY, "/v1/chat/completions", "Bearer é", False),
        (KEY, "/props", None, False),
        (KEY, "/health", None, True),
        (KEY, "/bundle.js", None, True),
    ],
)
def test_is_authorized(api_key, path, authorization, expected):
    assert is_authorized(api_key, path, authorization) is expected


def test_public_paths_hold_no_model_data():
    # /props exposes the model path and chat template; /v1/models the alias.
    assert not PUBLIC_PATHS & {"/props", "/slots", "/metrics", "/v1/models"}


class TestEmbeddedServerAuth:
    def _server(self, api_key):
        from inferna.llama.server.embedded import EmbeddedServer

        server = EmbeddedServer.__new__(EmbeddedServer)
        server._config = ServerConfig(model_path="unused.gguf", model_alias="m", api_key=api_key)
        server._logger = Mock()
        return server

    def test_missing_key_is_401_before_routing(self):
        conn = Mock()
        self._server(KEY).handle_http_request(conn, "POST", "/v1/chat/completions", {}, "{}")
        conn.send_json.assert_called_once()
        assert conn.send_json.call_args.args[1] == 401
        assert conn.send_json.call_args.args[0]["error"]["type"] == "authentication_error"

    def test_valid_key_reaches_the_route(self):
        conn = Mock()
        headers = {"authorization": f"Bearer {KEY}"}
        self._server(KEY).handle_http_request(conn, "GET", "/v1/models?x=1", headers, "")
        payload = conn.send_json.call_args.args[0]
        assert payload["data"][0]["id"] == "m"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture
def python_server(monkeypatch):
    port = _free_port()
    server = PythonServer(ServerConfig(model_path="unused.gguf", port=port, api_key=KEY))
    monkeypatch.setattr(server, "load_model", lambda: True)
    assert server.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.stop()


class TestPythonServerAuth:
    def test_get_requires_key(self, python_server):
        status, body = _get(f"{python_server}/v1/models")
        assert status == 401
        assert body["error"]["type"] == "authentication_error"

    def test_get_with_key(self, python_server):
        assert _get(f"{python_server}/v1/models", {"Authorization": f"Bearer {KEY}"})[0] == 200

    def test_health_is_public(self, python_server):
        assert _get(f"{python_server}/health")[0] == 200

    def test_post_requires_key(self, python_server):
        req = urllib.request.Request(f"{python_server}/v1/chat/completions", data=b"{}", method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        assert exc.value.code == 401


class TestCliApiKey:
    def _run(self, monkeypatch, *argv):
        """Run the server CLI with a fake EmbeddedServer; return the config it received."""
        from inferna.llama.server import __main__ as cli
        from inferna.llama.server import embedded

        seen = {}

        class _FakeServer:
            def __init__(self, config):
                seen["config"] = config

            def start(self):
                return False

        monkeypatch.setattr(embedded, "EmbeddedServer", _FakeServer)
        monkeypatch.setattr(sys, "argv", ["inferna server", "-m", "unused.gguf", *argv])
        assert cli.main() == 1
        return seen["config"]

    def test_default_has_no_key(self, monkeypatch):
        assert self._run(monkeypatch).api_key is None

    def test_api_key_flag(self, monkeypatch):
        assert self._run(monkeypatch, "--api-key", KEY).api_key == KEY

    def test_api_key_file_strips_trailing_newline(self, monkeypatch, tmp_path):
        f = tmp_path / "key"
        f.write_text(KEY + "\n")
        assert self._run(monkeypatch, "--api-key-file", str(f)).api_key == KEY

    @pytest.mark.parametrize("content", ["", "\n", "a\nb\n"])
    def test_api_key_file_rejects_empty_or_multiline(self, monkeypatch, tmp_path, content):
        f = tmp_path / "key"
        f.write_text(content)
        with pytest.raises(SystemExit) as exc:
            self._run(monkeypatch, "--api-key-file", str(f))
        assert exc.value.code == 2

    def test_flags_are_mutually_exclusive(self, monkeypatch, tmp_path):
        with pytest.raises(SystemExit):
            self._run(monkeypatch, "--api-key", KEY, "--api-key-file", str(tmp_path / "key"))


def test_no_exposure_warning_when_key_is_set(caplog):
    import logging

    from inferna.llama.server.python import warn_if_not_loopback

    logger = logging.getLogger("test_auth_warning")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        warn_if_not_loopback("0.0.0.0", logger, api_key=KEY)
    assert not caplog.records
