"""Tests for the ggml sync/restore logic in `scripts/manage.py`.

`StableDiffusionCppBuilder` swaps stable-diffusion.cpp's vendored ggml for
llama.cpp's copy when SD is configured to share llama.cpp's ggml dylibs
(`SD_USE_VENDORED_GGML=0`, which is every `--dynamic` GPU wheel build). The
swap has to be reversible: `verify_checkout()` only compares HEAD shas, so a
dynamic build followed by a static one on the same checkout would otherwise
compile SD against the wrong ggml. Since sd.cpp vendors leejet's ggml *fork*,
which carries ops upstream ggml lacks, that mismatch is a hard compile error
rather than a subtle one.

These tests drive the swap against a synthetic checkout so the state machine
(sync, re-sync, restore, stale backup, missing backup) is covered without
cloning or building anything.
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


FORK_FILE = "vendored-fork.txt"
LLAMA_FILE = "from-llama.txt"


@pytest.fixture
def builder(tmp_path):
    """A SD builder pointed at a synthetic build tree with two ggml copies."""
    src = tmp_path / "build"
    sd_ggml = src / "stable-diffusion.cpp" / "ggml" / "include"
    llama_ggml = src / "llama.cpp" / "ggml" / "include"
    sd_ggml.mkdir(parents=True)
    llama_ggml.mkdir(parents=True)
    (sd_ggml.parent / FORK_FILE).write_text("leejet ggml fork\n")
    (sd_ggml / "ggml.h").write_text("ggml_quantize_i8_convrot\n")
    (llama_ggml.parent / LLAMA_FILE).write_text("llama.cpp ggml\n")
    (llama_ggml / "ggml.h").write_text("upstream ggml\n")

    b = manage.StableDiffusionCppBuilder()
    b.project.src = src
    assert b.src_dir == src / "stable-diffusion.cpp"
    return b


def state(b):
    """(vendored fork in place, llama copy in place, marker, backup)."""
    ggml = b._ggml_dir
    return (
        (ggml / FORK_FILE).exists(),
        (ggml / LLAMA_FILE).exists(),
        b._ggml_provenance_marker.exists(),
        b._vendored_ggml_backup.exists(),
    )


VENDORED = (True, False, False, False)
SYNCED = (False, True, True, True)


def test_starts_vendored(builder):
    assert state(builder) == VENDORED


def test_restore_is_noop_on_pristine_checkout(builder):
    builder._restore_vendored_ggml()
    assert state(builder) == VENDORED


def test_sync_installs_llama_ggml_and_parks_the_fork(builder):
    builder._sync_ggml_abi()
    assert state(builder) == SYNCED
    # The fork is parked, not deleted.
    assert (builder._vendored_ggml_backup / FORK_FILE).exists()


def test_resync_keeps_the_fork_backup(builder):
    """A second dynamic build must not overwrite the backup with llama's ggml."""
    builder._sync_ggml_abi()
    builder._sync_ggml_abi()
    assert state(builder) == SYNCED
    assert (builder._vendored_ggml_backup / FORK_FILE).exists()
    assert not (builder._vendored_ggml_backup / LLAMA_FILE).exists()


def test_restore_after_sync_round_trips(builder):
    builder._sync_ggml_abi()
    builder._restore_vendored_ggml()
    assert state(builder) == VENDORED
    assert (builder._ggml_dir / "include" / "ggml.h").read_text() == "ggml_quantize_i8_convrot\n"


def test_repeated_cycles_are_stable(builder):
    for _ in range(3):
        builder._sync_ggml_abi()
        assert state(builder) == SYNCED
        builder._restore_vendored_ggml()
        assert state(builder) == VENDORED


def test_sync_replaces_a_stale_backup(builder):
    """A backup with no marker alongside it is stale and must not be trusted."""
    stale = builder._vendored_ggml_backup
    stale.mkdir()
    (stale / "stale.txt").write_text("from an older checkout\n")

    builder._sync_ggml_abi()
    assert state(builder) == SYNCED
    assert (stale / FORK_FILE).exists()
    assert not (stale / "stale.txt").exists()


def test_restore_fails_loudly_when_backup_is_missing(builder):
    """Better to stop than to silently build SD against llama.cpp's ggml."""
    import shutil

    builder._sync_ggml_abi()
    shutil.rmtree(builder._vendored_ggml_backup)

    with pytest.raises(SystemExit) as exc:
        builder._restore_vendored_ggml()
    assert exc.value.code == 1


def test_sync_is_skipped_when_llama_ggml_is_absent(builder):
    """Missing llama.cpp sources warn rather than destroying the vendored tree."""
    import shutil

    shutil.rmtree(builder.project.src / "llama.cpp")
    builder._sync_ggml_abi()
    assert state(builder) == VENDORED


# First sd.cpp master counter that calls fork-only ggml ops (INT8 ConvRot,
# leejet/stable-diffusion.cpp#1857). See the CEILING comment in manage.py.
SDCPP_FIRST_BROKEN_MASTER = 817


def test_sd_pin_is_compatible_with_shared_ggml():
    """Guard the pin ceiling documented next to SDCPP_VERSION.

    sd.cpp master-817-bcc7e29 (INT8 ConvRot) calls ggml ops that exist only in
    leejet's ggml fork, so it cannot be built with llama.cpp's ggml swapped in.
    Bumping past the ceiling breaks every dynamic GPU wheel build.
    """
    match = re.fullmatch(r"master-(\d+)-[0-9a-f]+", manage.SDCPP_VERSION)
    assert match, (
        f"SDCPP_VERSION {manage.SDCPP_VERSION!r} is not a master-<n>-<sha> pin; "
        "re-check it against the shared-ggml ceiling by hand."
    )
    assert int(match.group(1)) < SDCPP_FIRST_BROKEN_MASTER
