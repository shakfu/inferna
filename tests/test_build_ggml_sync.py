"""Tests for how `scripts/manage.py` selects stable-diffusion.cpp's ggml.

`SD_USE_VENDORED_GGML` is the only switch. `StableDiffusionCppBuilder` derives
upstream's `SD_USE_UPSTREAM_GGML` and `SD_GGML_SOURCE_DIR` from it and passes
both in every mode, because both are CMake cache variables: omitting them lets
a value cached by an earlier configure override the switch.

These tests run against a synthetic checkout, without cloning or building.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANAGE_PY = PROJECT_ROOT / "scripts" / "manage.py"


def _load_manage():
    """Import scripts/manage.py as a module (it is not on the package path)."""
    if "manage" in sys.modules:
        return sys.modules["manage"]
    spec = importlib.util.spec_from_file_location("manage", MANAGE_PY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["manage"] = module
    spec.loader.exec_module(module)
    return module


manage = _load_manage()


@pytest.fixture
def builder(tmp_path, monkeypatch):
    """A SD builder on a post-#1999 checkout, with a built cmake dir and a llama ggml."""
    src = tmp_path / "build"
    sd_dir = src / "stable-diffusion.cpp"
    (src / "llama.cpp" / "ggml" / "src").mkdir(parents=True)
    (sd_dir / "cmake").mkdir(parents=True)
    (sd_dir / "cmake" / "ggml.cmake").write_text("# SD_GGML_SOURCE_DIR\n")
    (sd_dir / "build").mkdir()
    (sd_dir / "build" / "ggml-metal-device.m.o").write_text("stale object")

    b = manage.StableDiffusionCppBuilder()
    b.project.src = src
    assert b.src_dir == sd_dir
    monkeypatch.delenv("SD_USE_VENDORED_GGML", raising=False)
    return b


def test_shared_ggml_points_sd_at_llama_ggml(builder):
    """SD must compile llama.cpp's ggml, with fork-only calls compiled out."""
    assert builder._ggml_options() == {
        "SD_USE_UPSTREAM_GGML": True,
        "SD_GGML_SOURCE_DIR": str(builder.project.src / "llama.cpp" / "ggml"),
    }


def test_vendored_ggml_overrides_cached_options(builder, monkeypatch):
    """Both options are CMake cache variables; omitting them keeps a shared configure's values."""
    monkeypatch.setenv("SD_USE_VENDORED_GGML", "1")

    assert builder._ggml_options() == {
        "SD_USE_UPSTREAM_GGML": False,
        "SD_GGML_SOURCE_DIR": str(builder.src_dir / "ggml"),
    }


def test_missing_llama_ggml_is_rejected(builder):
    """The extension links llama.cpp's ggml, so SD must not fall back to its fork."""
    (builder.project.src / "llama.cpp").rename(builder.project.src / "llama.cpp.gone")

    with pytest.raises(RuntimeError, match="--sd-vendored-ggml"):
        builder._ggml_options()


def test_pre_1999_pin_is_rejected(builder):
    """A pin older than master-883 ignores SD_GGML_SOURCE_DIR and builds the fork ggml."""
    (builder.src_dir / "cmake" / "ggml.cmake").unlink()

    with pytest.raises(RuntimeError, match="master-883"):
        builder._ggml_options()


def test_build_dir_survives_unchanged_ggml_options(builder):
    """Same ggml as the last configure: keep the objects for an incremental build."""
    builder._drop_build_dir_on_ggml_change(builder._ggml_options())
    (builder.build_dir / "sd.o").write_text("object")

    builder._drop_build_dir_on_ggml_change(builder._ggml_options())

    assert (builder.build_dir / "sd.o").exists()


def test_build_dir_is_dropped_when_ggml_options_change(builder, monkeypatch):
    """Objects from the previous ggml tree must not be relinked against the new one."""
    builder._drop_build_dir_on_ggml_change(builder._ggml_options())
    (builder.build_dir / "ggml-metal-device.m.o").write_text("shared-mode object")
    monkeypatch.setenv("SD_USE_VENDORED_GGML", "1")

    builder._drop_build_dir_on_ggml_change(builder._ggml_options())

    assert not (builder.build_dir / "ggml-metal-device.m.o").exists()


def test_unstamped_build_dir_is_dropped(builder):
    """A build dir from before the stamp existed has unknown ggml provenance."""
    builder._drop_build_dir_on_ggml_change(builder._ggml_options())

    assert not (builder.build_dir / "ggml-metal-device.m.o").exists()


# First sd.cpp master counter with SD_USE_UPSTREAM_GGML and SD_GGML_SOURCE_DIR
# (leejet/stable-diffusion.cpp#1999). See the FLOOR comment in manage.py.
SDCPP_FIRST_SHARED_GGML_MASTER = 883


def test_sd_pin_supports_shared_ggml():
    """Shared-ggml builds need a pin that can compile against upstream ggml."""
    match = re.fullmatch(r"master-(\d+)-[0-9a-f]+", manage.SDCPP_VERSION)
    assert match, (
        f"SDCPP_VERSION {manage.SDCPP_VERSION!r} is not a master-<n>-<sha> pin; "
        "re-check it against the shared-ggml floor by hand."
    )
    assert int(match.group(1)) >= SDCPP_FIRST_SHARED_GGML_MASTER
