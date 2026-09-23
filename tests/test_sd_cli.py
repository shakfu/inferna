"""Tests for the stable-diffusion CLI's context-param and log-level mapping."""

import argparse

import pytest

mod = pytest.importorskip("inferna.sd.__main__")
sd = pytest.importorskip("inferna.sd.stable_diffusion")


def _params(argv):
    parser = argparse.ArgumentParser()
    mod.add_common_model_args(parser)
    mod.add_common_memory_args(parser)
    return mod.create_context_params(parser.parse_args(argv))


class TestMemoryFlags:
    def test_defaults_follow_upstream(self):
        params = _params([])
        assert params.auto_fit is True  # upstream default since master-845 (#1942)
        assert params.disable_prefetch is False
        assert params.disable_segmented_compute is False
        assert params.tokenizer is None

    def test_auto_fit_on_off(self):
        assert _params(["--auto-fit", "off"]).auto_fit is False
        assert _params(["--auto-fit", "on"]).auto_fit is True
        assert _params(["--auto-fit"]).auto_fit is True

    def test_segmented_compute_flags(self):
        params = _params(["--disable-prefetch", "--disable-segmented-compute"])
        assert params.disable_prefetch is True
        assert params.disable_segmented_compute is True

    def test_tokenizer(self):
        assert _params(["--tokenizer", "main=tok.json"]).tokenizer == "main=tok.json"


class TestLogLevels:
    """Upstream inserted SD_LOG_VERBOSE after DEBUG, shifting INFO..ERROR by one."""

    @pytest.fixture
    def emit(self, monkeypatch):
        """Register the CLI's callback and return the wrapper sd.cpp would call with raw ints."""
        registered = []
        monkeypatch.setattr(sd._n, "set_log_callback", registered.append)
        monkeypatch.setattr(sd._n, "set_progress_callback", lambda cb: None)

        def setup(verbose):
            mod.setup_logging(argparse.Namespace(verbose=verbose, progress=False))
            return registered[-1]

        return setup

    def test_warnings_only_drops_info(self, emit, capsys):
        wrap = emit(verbose=False)
        wrap(int(sd.LogLevel.INFO), "info line\n")
        wrap(int(sd.LogLevel.WARN), "warn line\n")
        wrap(int(sd.LogLevel.ERROR), "error line\n")
        out = capsys.readouterr().out
        assert "info line" not in out
        assert "[WARN] warn line" in out
        assert "[ERROR] error line" in out

    def test_verbose_labels_every_level(self, emit, capsys):
        wrap = emit(verbose=True)
        for level in sd.LogLevel:
            wrap(int(level), f"{level.name.lower()}\n")
        out = capsys.readouterr().out
        for level in sd.LogLevel:
            assert f"[{level.name}] {level.name.lower()}" in out
