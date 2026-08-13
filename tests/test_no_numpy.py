"""numpy must never be required to run inferna.

inferna declares no runtime dependencies (``dependencies = []``), so every
import path and every code path has to work on a bare install. numpy is
supported as *input* -- callers who have it can pass ndarrays -- but it is
never imported as a hard requirement, and never needed to produce a result.

These tests run each check in a subprocess with numpy masked at the import
hook, which is the only faithful way to simulate a numpy-less install from
inside an environment that has numpy for the rest of the suite.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Installs an import hook that makes `import numpy` fail, then runs the body.
_MASK = """
import builtins
_real = builtins.__import__
def _guard(name, *a, **k):
    if name == 'numpy' or name.startswith('numpy.'):
        raise ImportError("No module named 'numpy'")
    return _real(name, *a, **k)
builtins.__import__ = _guard
import sys
sys.modules.pop('numpy', None)
"""


def _run_without_numpy(body: str) -> subprocess.CompletedProcess:
    script = _MASK + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )


# Every public subpackage. A hard `import numpy` anywhere in these chains
# breaks a bare install, which is exactly the regression being guarded.
IMPORT_TARGETS = [
    "inferna",
    "inferna.api",
    "inferna.batching",
    "inferna.agents",
    "inferna.integrations",
    "inferna.llama.llama_cpp",
    "inferna.llama.mtmd.multimodal",
    "inferna.llama.server.embedded",
    "inferna.rag.embedder",
    "inferna.sd.stable_diffusion",
    "inferna.whisper.whisper_cpp",
    "inferna.whisper.streaming",
    "inferna.whisper.cli",
]


@pytest.mark.parametrize("module", IMPORT_TARGETS)
def test_module_imports_without_numpy(module):
    """No inferna module may import numpy at module scope."""
    proc = _run_without_numpy(f"""
        import {module}
        print('ok')
    """)
    assert proc.returncode == 0, f"{module} failed to import without numpy:\n{proc.stderr}"
    assert "ok" in proc.stdout


def test_numpy_is_not_a_declared_runtime_dependency():
    """The package metadata must keep runtime dependencies empty."""
    import tomllib

    root = Path(__file__).resolve().parent.parent
    with open(root / "pyproject.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    assert cfg["project"]["dependencies"] == [], (
        "inferna must declare no runtime dependencies; numpy in particular "
        "is optional input, not a requirement"
    )


def test_wav_load_and_resample_without_numpy():
    """The whisper CLI audio path produces real output with numpy absent."""
    proc = _run_without_numpy("""
        from inferna.whisper.cli import load_wav_file, resample_audio
        samples, sr = load_wav_file('tests/samples/jfk.wav')
        assert len(samples) > 0, 'no samples decoded'
        assert sr == 16000, sr
        # Same rate: passthrough. Different rate: interpolating path.
        assert len(resample_audio(samples, sr, sr)) == len(samples)
        half = resample_audio(samples, sr, sr // 2)
        assert abs(len(half) - len(samples) // 2) <= 1, len(half)
        print('ok')
    """)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_streamer_accepts_stdlib_buffers_without_numpy():
    """WhisperStreamer coerces array.array/memoryview with no numpy present."""
    proc = _run_without_numpy("""
        from array import array
        from inferna.whisper.streaming import WhisperStreamer
        s = WhisperStreamer.__new__(WhisperStreamer)   # no model load needed
        for src in (array('f', [0.25, -0.5, 1.0]), memoryview(array('f', [0.25, -0.5, 1.0]))):
            buf = s._coerce_samples(src)
            assert len(buf) == 3, len(buf)
            assert abs(buf[1] + 0.5) < 1e-6, buf[1]
        # Integer input is converted rather than rejected.
        assert len(s._coerce_samples(array('i', [1, 2, 3]))) == 3
        # 2-D input is still rejected.
        try:
            s._coerce_samples(memoryview(bytearray(16)).cast('f', (2, 2)))
        except ValueError:
            pass
        else:
            raise AssertionError('2-D input should be rejected')
        print('ok')
    """)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_numpy_input_still_supported():
    """numpy remains a supported *input* type when it is installed."""
    np = pytest.importorskip("numpy")
    from inferna.whisper.streaming import WhisperStreamer

    streamer = WhisperStreamer.__new__(WhisperStreamer)

    f32 = np.array([0.25, -0.5, 1.0], dtype=np.float32)
    buf = streamer._coerce_samples(f32)
    assert len(buf) == 3
    # float32 input must pass through without a copy.
    assert isinstance(buf, memoryview)

    # float64 is converted rather than rejected.
    f64 = np.array([0.25, -0.5, 1.0], dtype=np.float64)
    assert len(streamer._coerce_samples(f64)) == 3

    with pytest.raises(ValueError):
        streamer._coerce_samples(np.zeros((2, 2), dtype=np.float32))
