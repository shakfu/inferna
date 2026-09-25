"""LlamaModel.from_fileobj: loading a GGUF from an open file or fd, including
one embedded at an offset in a larger file.

Uses bge-small (37 MB) so the embedded copies stay cheap.
"""

import io
import os

import pytest

import inferna.llama.llama_cpp as cy
from conftest import ROOT

SMALL_MODEL = ROOT / "models" / "bge-small-en-v1.5-q8_0.gguf"

pytestmark = pytest.mark.skipif(not SMALL_MODEL.exists(), reason=f"{SMALL_MODEL.name} not downloaded")


@pytest.fixture(scope="module")
def reference():
    m = cy.LlamaModel(str(SMALL_MODEL), verbose=False)
    return m.n_params, m.desc


def _embed(tmp_path, prefix_len):
    """Write <prefix_len junk bytes><model><64 junk bytes>; return the path."""
    path = tmp_path / f"bundle_{prefix_len}.bin"
    with open(path, "wb") as out:
        out.write(b"\xab" * prefix_len)
        out.write(SMALL_MODEL.read_bytes())
        out.write(b"\xcd" * 64)
    return path


def test_fileobj_at_start(reference):
    with open(SMALL_MODEL, "rb") as f:
        m = cy.LlamaModel.from_fileobj(f)
        assert (m.n_params, m.desc) == reference
        assert m.path_model == str(SMALL_MODEL)


def test_raw_fd(reference):
    fd = os.open(SMALL_MODEL, os.O_RDONLY)
    try:
        m = cy.LlamaModel.from_fileobj(fd)
        assert (m.n_params, m.desc) == reference
        assert m.path_model is None
        assert os.lseek(fd, 0, os.SEEK_CUR) == 0
    finally:
        os.close(fd)


def test_embedded_at_current_position(tmp_path, reference):
    path = _embed(tmp_path, 64)
    with open(path, "rb") as f:
        f.seek(64)
        m = cy.LlamaModel.from_fileobj(f)
        assert (m.n_params, m.desc) == reference
        assert f.tell() == 64


def test_embedded_at_explicit_offset_restores_position(tmp_path, reference):
    path = _embed(tmp_path, 64)
    with open(path, "rb") as f:
        f.read(10)  # fills the read buffer past the logical position
        m = cy.LlamaModel.from_fileobj(f, offset=64)
        assert (m.n_params, m.desc) == reference
        assert f.tell() == 10
        assert f.read(1) == b"\xab"


def test_unaligned_offset_rejected_with_mmap(tmp_path):
    path = _embed(tmp_path, 3)
    params = cy.LlamaModelParams()
    params.load_mode = cy.LLAMA_LOAD_MODE_MMAP
    with open(path, "rb") as f, pytest.raises(ValueError, match="32-byte aligned"):
        cy.LlamaModel.from_fileobj(f, offset=3, params=params)


def test_unaligned_offset_loads_without_mmap(tmp_path, reference):
    path = _embed(tmp_path, 3)
    params = cy.LlamaModelParams()
    params.load_mode = cy.LLAMA_LOAD_MODE_NONE
    with open(path, "rb") as f:
        m = cy.LlamaModel.from_fileobj(f, offset=3, params=params)
        assert (m.n_params, m.desc) == reference


def test_bad_header_rejected_and_position_restored(tmp_path):
    path = _embed(tmp_path, 64)
    with open(path, "rb") as f:
        f.seek(5)
        with pytest.raises(ValueError, match="valid GGUF"):
            cy.LlamaModel.from_fileobj(f, offset=0)
        assert f.tell() == 5


@pytest.mark.parametrize("offset", [-1, SMALL_MODEL.stat().st_size if SMALL_MODEL.exists() else 0])
def test_offset_out_of_range(offset):
    with open(SMALL_MODEL, "rb") as f, pytest.raises(ValueError, match="outside"):
        cy.LlamaModel.from_fileobj(f, offset=offset)


def test_text_mode_rejected():
    with open(SMALL_MODEL, "r") as f, pytest.raises(TypeError, match="binary mode"):
        cy.LlamaModel.from_fileobj(f)


def test_object_without_fd_rejected():
    with pytest.raises(io.UnsupportedOperation):
        cy.LlamaModel.from_fileobj(io.BytesIO(SMALL_MODEL.read_bytes()[:64]))


def test_pipe_rejected():
    r, w = os.pipe()
    try:
        with pytest.raises(ValueError, match="not a regular file"):
            cy.LlamaModel.from_fileobj(r)
    finally:
        os.close(r)
        os.close(w)


@pytest.mark.parametrize("load_mode", ["LLAMA_LOAD_MODE_MMAP", "LLAMA_LOAD_MODE_NONE"])
def test_usable_after_file_closed(tmp_path, load_mode):
    """llama.cpp borrows the FILE* only for the load; the model must not read it later."""
    path = _embed(tmp_path, 64)
    params = cy.LlamaModelParams()
    params.load_mode = getattr(cy, load_mode)
    with open(path, "rb") as f:
        m = cy.LlamaModel.from_fileobj(f, offset=64, params=params)
    path.unlink()

    cparams = cy.LlamaContextParams()
    cparams.n_ctx = 64
    cparams.embeddings = True
    ctx = cy.LlamaContext(m, cparams)
    tokens = m.get_vocab().tokenize("hello world", add_special=True, parse_special=False)
    ctx.decode(cy.llama_batch_get_one(tokens, 0))
    emb = ctx.get_embeddings_seq(0)
    assert len(emb) == m.n_embd_out
    assert any(v != 0.0 for v in emb)
