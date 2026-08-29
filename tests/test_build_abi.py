"""Guards on the ggml ABI that inferna's build has to keep consistent.

stable-diffusion.cpp, llama.cpp and whisper.cpp are compiled separately but the
shipped extensions link *one* ggml, so anything that changes `struct
ggml_tensor`'s layout has to be identical on every side. `GGML_MAX_NAME` does:
`name` is an inline `char[GGML_MAX_NAME]` in the struct, and `extra` -- the
field right after it, and the last one -- moves with it. SD writes
`tensor->extra` on every graph-cut segment boundary, so a mismatch puts those
writes on top of the next `ggml_object` header. Nothing fails to compile or
link; the arena is simply corrupted at runtime.

The pin went stale once (128 while upstream had moved to 160), so these tests
check every direction: the value manage.py propagates to the follower trees,
and the value inferna's own CMakeLists and xcframework script compile against.
"""

import importlib.util
import inspect
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MANAGE_PY = ROOT / "scripts" / "manage.py"


@pytest.fixture(scope="module")
def manage():
    """Import scripts/manage.py as a module (it is stdlib-only by design)."""
    spec = importlib.util.spec_from_file_location("inferna_manage", MANAGE_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["inferna_manage"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def shared_ggml(monkeypatch):
    """Select the shared-ggml configuration every GPU build actually uses."""
    monkeypatch.setenv("SD_USE_VENDORED_GGML", "0")


def test_every_tree_that_links_the_shared_ggml_gets_the_define(manage):
    """llama.cpp and whisper.cpp must both be built with SD's value.

    inferna's CMakeLists takes the ggml libs from ``${LLAMACPP_LIB}`` on every
    platform except the macOS static build, so llama.cpp's and whisper.cpp's
    object code calls *llama.cpp's* ggml. A tree left on the upstream default
    of 64 is compiled against a ``ggml_tensor`` 96 bytes shorter than the one
    being allocated.
    """
    # stable-diffusion.cpp is deliberately absent: it *sets* the value in its
    # own CMakeLists, and `_verify_ggml_max_name()` checks inferna's pin against
    # it. These two have to follow.
    followers = [manage.LlamaCppBuilder(), manage.WhisperCppBuilder()]
    for builder in followers:
        # The value reaches cmake as a raw -D in CMAKE_{C,CXX}_FLAGS; assert the
        # builder is wired to emit it rather than re-deriving how.
        source = inspect.getsource(type(builder).build)
        assert "GGML_MAX_NAME" in source, (
            f"{builder.name} does not propagate GGML_MAX_NAME to its cmake configure; "
            f"it will compile against a different struct ggml_tensor than the ggml it links"
        )


def test_cmakelists_matches_the_propagated_value(manage):
    """inferna's own translation units must use the value manage.py ships."""
    pinned = manage.StableDiffusionCppBuilder.GGML_MAX_NAME
    text = (ROOT / "CMakeLists.txt").read_text()
    found = re.findall(r"add_definitions\(-DGGML_MAX_NAME=(\d+)\)", text)
    assert found, "inferna's CMakeLists.txt no longer defines GGML_MAX_NAME"
    assert [int(v) for v in found] == [pinned] * len(found)


def test_xcframework_matches_the_propagated_value(manage):
    """The xcframework build configures ggml itself and must agree too."""
    pinned = manage.StableDiffusionCppBuilder.GGML_MAX_NAME
    text = (ROOT / "scripts" / "make_xcframework.py").read_text()
    found = re.findall(r"-DGGML_MAX_NAME=(\d+)", text)
    assert found, "make_xcframework.py no longer sets GGML_MAX_NAME"
    assert [int(v) for v in found] == [pinned] * len(found)


def test_verify_rejects_a_stale_pin(manage, tmp_path, monkeypatch, shared_ggml):
    """A drifted upstream value must fail the build, not warn."""
    builder = manage.StableDiffusionCppBuilder()
    (tmp_path / "CMakeLists.txt").write_text("add_definitions(-DGGML_MAX_NAME=%d)\n" % (builder.GGML_MAX_NAME + 32))
    monkeypatch.setattr(type(builder), "src_dir", property(lambda self: tmp_path))
    with pytest.raises(RuntimeError, match="GGML_MAX_NAME"):
        builder._verify_ggml_max_name()


def test_verify_accepts_a_current_pin(manage, tmp_path, monkeypatch, shared_ggml):
    builder = manage.StableDiffusionCppBuilder()
    (tmp_path / "CMakeLists.txt").write_text("add_definitions(-DGGML_MAX_NAME=%d)\n" % builder.GGML_MAX_NAME)
    monkeypatch.setattr(type(builder), "src_dir", property(lambda self: tmp_path))
    builder._verify_ggml_max_name()  # must not raise


def test_verify_is_skipped_for_a_vendored_ggml(manage, tmp_path, monkeypatch):
    """With SD on its own ggml there is no shared struct to keep in step."""
    monkeypatch.setenv("SD_USE_VENDORED_GGML", "1")
    builder = manage.StableDiffusionCppBuilder()
    (tmp_path / "CMakeLists.txt").write_text("add_definitions(-DGGML_MAX_NAME=9999)\n")
    monkeypatch.setattr(type(builder), "src_dir", property(lambda self: tmp_path))
    builder._verify_ggml_max_name()  # must not raise


def test_pin_matches_the_vendored_stable_diffusion_checkout(manage):
    """When the SD tree is on disk, the pin must describe it."""
    sd_cmakelists = ROOT / "build" / "stable-diffusion.cpp" / "CMakeLists.txt"
    if not sd_cmakelists.exists():
        pytest.skip("stable-diffusion.cpp is not checked out")
    match = re.search(r"add_definitions\(\s*-DGGML_MAX_NAME=(\d+)\s*\)", sd_cmakelists.read_text())
    assert match, "stable-diffusion.cpp no longer sets GGML_MAX_NAME"
    assert int(match.group(1)) == manage.StableDiffusionCppBuilder.GGML_MAX_NAME


# ---------------------------------------------------------------------------
# CMake build-dir invalidation
#
# A CMake cache persists across configures and a re-configure only overwrites
# the entries it is handed, so an option dropped from the command line keeps its
# old value. `build()` and `build_shared()` share one build dir per builder.
# ---------------------------------------------------------------------------


@pytest.fixture
def configured(manage, tmp_path):
    """A build dir carrying a cache and the stamp for a given signature."""

    def _make(signature: str, *, stamp: bool = True):
        build_dir = tmp_path / "build"
        build_dir.mkdir(exist_ok=True)
        (build_dir / "CMakeCache.txt").write_text("GGML_BACKEND_DL:BOOL=ON\n")
        if stamp:
            (build_dir / manage.ShellCmd.CMAKE_ARGS_STAMP).write_text(signature)
        return build_dir

    return _make


def _shell(manage):
    """A bare ShellCmd with a logger attached (the class only declares one)."""
    import logging

    sh = manage.ShellCmd()
    sh.log = logging.getLogger("test")
    return sh


def test_build_dir_survives_an_unchanged_configure(manage, configured):
    build_dir = configured("abc123")
    _shell(manage)._invalidate_stale_build_dir(build_dir, "abc123")
    assert (build_dir / "CMakeCache.txt").exists()


def test_build_dir_is_discarded_when_arguments_change(manage, configured):
    """The static-after-dynamic case: GGML_BACKEND_DL would survive otherwise."""
    build_dir = configured("abc123")
    _shell(manage)._invalidate_stale_build_dir(build_dir, "def456")
    assert not build_dir.exists()


def test_build_dir_without_a_stamp_is_discarded(manage, configured):
    """Predates the check, so its arguments are unknown and not trustworthy."""
    build_dir = configured("abc123", stamp=False)
    _shell(manage)._invalidate_stale_build_dir(build_dir, "abc123")
    assert not build_dir.exists()


def test_unconfigured_build_dir_is_left_alone(manage, tmp_path):
    """Nothing to invalidate before the first configure."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "scratch.txt").write_text("keep me")
    _shell(manage)._invalidate_stale_build_dir(build_dir, "abc123")
    assert (build_dir / "scratch.txt").exists()


def test_signature_separates_the_two_link_modes(manage, tmp_path):
    """build() and build_shared() must not share a signature.

    They differ by BUILD_SHARED_LIBS and GGML_BACKEND_DL; either alone is
    enough, and the signature covers the whole option set rather than a
    hand-picked flag.
    """
    sh = _shell(manage)
    src = tmp_path / "src"
    src.mkdir()
    seen = set()
    for opts in (
        {"BUILD_SHARED_LIBS": False},
        {"BUILD_SHARED_LIBS": True},
        {"BUILD_SHARED_LIBS": True, "GGML_BACKEND_DL": True},
    ):
        captured = {}
        sh.cmd = lambda c, cwd=".", _c=captured: _c.setdefault("cmd", c)  # noqa: ARG005
        sh.cmake_config(src_dir=src, build_dir=tmp_path / "b", **opts)
        seen.add((tmp_path / "b" / manage.ShellCmd.CMAKE_ARGS_STAMP).read_text())
    assert len(seen) == 3


def test_stamp_is_not_written_when_configure_fails(manage, tmp_path):
    """A failed configure leaves a partial cache that must not look current."""
    sh = _shell(manage)
    src = tmp_path / "src"
    src.mkdir()
    build_dir = tmp_path / "b"

    def _boom(cmd, cwd="."):
        build_dir.mkdir(parents=True, exist_ok=True)
        (build_dir / "CMakeCache.txt").write_text("partial\n")
        raise SystemExit(1)

    sh.cmd = _boom
    with pytest.raises(SystemExit):
        sh.cmake_config(src_dir=src, build_dir=build_dir, BUILD_SHARED_LIBS=True)
    assert not (build_dir / manage.ShellCmd.CMAKE_ARGS_STAMP).exists()
