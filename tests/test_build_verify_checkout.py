"""Tests for `Builder.verify_checkout()` in `scripts/manage.py`."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANAGE_PY = PROJECT_ROOT / "scripts" / "manage.py"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


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


def _git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "tag.gpgSign=false", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def builder(tmp_path):
    """A whisper builder whose checkout has a lightweight and an annotated tag on HEAD~1,
    and HEAD one commit ahead."""
    src = tmp_path / "build"
    repo = src / "whisper.cpp"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "one")
    _git(repo, "tag", "light")
    _git(repo, "tag", "-a", "annotated", "-m", "annotated")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "two")

    b = manage.WhisperCppBuilder()
    b.project.src = src
    assert b.src_dir == repo
    return b


@pytest.mark.parametrize("tag", ["light", "annotated"])
def test_accepts_checkout_at_tag(builder, tag):
    _git(builder.src_dir, "checkout", "-q", tag)
    builder.version = tag
    builder.verify_checkout()


@pytest.mark.parametrize("tag", ["light", "annotated"])
def test_rejects_checkout_off_tag(builder, tag):
    builder.version = tag
    with pytest.raises(SystemExit):
        builder.verify_checkout()
