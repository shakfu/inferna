"""Tests for the ``quarto_render`` tool in ``inferna.agents.tools``.

Mirrors the test surface in the cyllama sibling and the Go reference at
``~/projects/personal/margo/pkg/margo/agent/tools_quarto_test.go``:

* Argument-validation tests run unconditionally.
* Slug-derivation tests run unconditionally (pure-Python).
* The real ``quarto render`` invocations are gated on
  :func:`quarto_available` so the suite stays portable on hosts that
  don't have the binary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inferna.agents.tools import (
    _quarto_slug_from_content,
    _quarto_unique_path,
    default_quarto_output_dir,
    quarto_available,
    quarto_render,
)
from inferna.agents.tools.quarto import _quarto_write_target


needs_quarto = pytest.mark.skipif(not quarto_available(), reason="quarto not on PATH")


class TestQuartoSlug:
    @pytest.mark.parametrize(
        "content,expected",
        [
            ('---\ntitle: "How to Boil an Egg"\n---\n', "how-to-boil-an-egg"),
            ("---\ntitle: 'Mixed/Punct & Whitespace!'\n---\n", "mixed-punct-whitespace"),
            ("---\ntitle: bare value\n---\n", "bare-value"),
            ("no frontmatter at all", "document"),
            ('---\ntitle: "   "\n---', "document"),
        ],
    )
    def test_slug_cases(self, content: str, expected: str):
        assert _quarto_slug_from_content(content) == expected

    def test_long_title_is_capped(self):
        long_title = "a" * 200
        slug = _quarto_slug_from_content(f"---\ntitle: {long_title}\n---\n")
        assert len(slug) <= 64
        assert slug == "a" * 64


class TestQuartoUniquePath:
    def test_returns_base_when_free(self, tmp_path: Path):
        p = _quarto_unique_path(tmp_path, "doc", ".qmd")
        assert p == tmp_path / "doc.qmd"

    def test_increments_on_collision(self, tmp_path: Path):
        (tmp_path / "doc.qmd").write_text("x")
        p = _quarto_unique_path(tmp_path, "doc", ".qmd")
        assert p == tmp_path / "doc-2.qmd"

    def test_keeps_incrementing(self, tmp_path: Path):
        (tmp_path / "doc.qmd").write_text("x")
        (tmp_path / "doc-2.qmd").write_text("x")
        (tmp_path / "doc-3.qmd").write_text("x")
        p = _quarto_unique_path(tmp_path, "doc", ".qmd")
        assert p == tmp_path / "doc-4.qmd"


class TestQuartoRenderArgValidation:
    def test_empty_input_and_content_rejected(self):
        with pytest.raises(ValueError, match="input.*content"):
            quarto_render(input="", content="", to="html")

    def test_unsupported_format_rejected(self):
        with pytest.raises(ValueError, match="unsupported format"):
            quarto_render(input="x.qmd", content="", to="rot13")

    def test_missing_quarto_surfaces_install_hint(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("inferna.agents.tools.quarto._shutil.which", lambda _name: None)
        with pytest.raises(RuntimeError, match="quarto CLI not found"):
            quarto_render(input="x.qmd", content="", to="html")


class TestQuartoDefaultOutputDir:
    def test_env_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        target = tmp_path / "my-outputs"
        monkeypatch.setenv("INFERNA_QUARTO_OUTPUT_DIR", str(target))
        result = default_quarto_output_dir()
        assert result == target
        assert result.is_dir()

    def test_quarto_render_ships_curated_example(self):
        assert quarto_render.example_args is not None
        assert "content" in quarto_render.example_args
        assert quarto_render.example_args["to"] == "pptx"
        from inferna.agents.tools import _quarto_slug_from_content

        slug = _quarto_slug_from_content(quarto_render.example_args["content"])
        assert slug == "how-to-boil-an-egg"
        rendered = quarto_render.to_prompt_string()
        assert "How to Boil an Egg" in rendered
        assert '"to": "example"' not in rendered

    def test_default_under_documents(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("INFERNA_QUARTO_OUTPUT_DIR", raising=False)
        result = default_quarto_output_dir()
        assert result.name == "output"
        assert result.parent.name == "inferna"


@needs_quarto
class TestQuartoRenderLive:
    def test_render_existing_html(self, tmp_path: Path):
        src = tmp_path / "doc.qmd"
        src.write_text("---\ntitle: t\n---\n\nhello\n", encoding="utf-8")

        out = quarto_render(input=str(src), content="", to="html")
        rendered = tmp_path / "doc.html"
        assert rendered.exists(), f"expected {rendered} to exist; got:\n{out}"
        assert f"Output file: {rendered}" in out
        assert f"[doc.html](file://{rendered})" in out

    def test_create_and_render_writes_input(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        # Writes are confined to the output dir, so point it at tmp_path.
        monkeypatch.setenv("INFERNA_QUARTO_OUTPUT_DIR", str(tmp_path))
        dst = tmp_path / "presentation.qmd"
        src = '---\ntitle: "t"\nformat: html\n---\n\nbody\n'

        out = quarto_render(input=str(dst), content=src, to="html")
        assert dst.exists()
        rendered = tmp_path / "presentation.html"
        assert rendered.exists()
        assert f"Output file: {rendered}" in out

    def test_create_and_render_slug_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("INFERNA_QUARTO_OUTPUT_DIR", str(tmp_path))
        src = '---\ntitle: "How to Boil an Egg"\nformat: html\n---\n\nbody\n'

        out = quarto_render(input="", content=src, to="html")
        rendered = tmp_path / "how-to-boil-an-egg.html"
        assert rendered.exists()
        assert f"[how-to-boil-an-egg.html](file://{rendered})" in out

        out2 = quarto_render(input="", content=src, to="html")
        assert "how-to-boil-an-egg-2" in out2

    def test_input_not_found(self, tmp_path: Path):
        ghost = tmp_path / "does-not-exist.qmd"
        with pytest.raises(FileNotFoundError):
            quarto_render(input=str(ghost), content="", to="html")

    def test_output_dir_inside_the_root_renders_there(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("INFERNA_QUARTO_OUTPUT_DIR", str(tmp_path))
        src = '---\ntitle: "t"\nformat: html\n---\n\nbody\n'

        quarto_render(input="doc.qmd", content=src, to="html", output_dir="rendered")
        assert (tmp_path / "rendered" / "doc.html").exists()


class TestQuartoWriteConfinement:
    """``input`` and ``output_dir`` reach ``quarto_render`` straight from the
    model, and the write is followed by a render that executes the document's
    code cells. Text the agent merely reads must not choose where that lands.
    """

    @pytest.fixture
    def root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        out = tmp_path / "out"
        monkeypatch.setenv("INFERNA_QUARTO_OUTPUT_DIR", str(out))
        return out.resolve()

    SRC = '---\ntitle: "t"\nformat: html\n---\n\nbody\n'

    def test_absolute_path_outside_the_root_is_refused(self, root: Path, tmp_path: Path):
        escape = tmp_path / "elsewhere" / "evil.qmd"
        with pytest.raises(ValueError, match="refusing to write outside"):
            quarto_render(input=str(escape), content=self.SRC)
        assert not escape.exists()
        assert not escape.parent.exists()

    def test_dotdot_traversal_is_refused(self, root: Path):
        with pytest.raises(ValueError, match="refusing to write outside"):
            quarto_render(input="../../evil.qmd", content=self.SRC)

    def test_home_expansion_outside_the_root_is_refused(self, root: Path):
        with pytest.raises(ValueError, match="refusing to write outside"):
            quarto_render(input="~/evil.qmd", content=self.SRC)

    def test_symlink_pointing_out_of_the_root_is_refused(self, root: Path, tmp_path: Path):
        outside = tmp_path / "outside"
        outside.mkdir()
        root.mkdir(parents=True, exist_ok=True)
        (root / "link").symlink_to(outside, target_is_directory=True)
        with pytest.raises(ValueError, match="refusing to write outside"):
            quarto_render(input="link/evil.qmd", content=self.SRC)
        assert not (outside / "evil.qmd").exists()

    def test_relative_path_is_taken_relative_to_the_root(self, root: Path):
        target = _quarto_write_target("nested/report.qmd", self.SRC)
        assert target == root / "nested" / "report.qmd"

    def test_path_inside_the_root_is_allowed(self, root: Path):
        target = _quarto_write_target(str(root / "report.qmd"), self.SRC)
        assert target == root / "report.qmd"

    def test_empty_input_still_slugs_into_the_root(self, root: Path):
        target = _quarto_write_target("", '---\ntitle: "My Doc"\n---\n')
        assert target == root / "my-doc.qmd"

    def test_refusal_names_the_configured_root(self, root: Path, tmp_path: Path):
        with pytest.raises(ValueError, match="INFERNA_QUARTO_OUTPUT_DIR"):
            quarto_render(input=str(tmp_path / "evil.qmd"), content=self.SRC)

    def test_output_dir_absolute_outside_the_root_is_refused(self, root: Path, tmp_path: Path):
        with pytest.raises(ValueError, match="refusing to write outside"):
            quarto_render(content=self.SRC, output_dir=str(tmp_path / "elsewhere"))

    def test_output_dir_dotdot_traversal_is_refused(self, root: Path):
        with pytest.raises(ValueError, match="refusing to write outside"):
            quarto_render(content=self.SRC, output_dir="../../elsewhere")

    def test_output_dir_symlink_out_of_the_root_is_refused(self, root: Path, tmp_path: Path):
        outside = tmp_path / "outside"
        outside.mkdir()
        root.mkdir(parents=True, exist_ok=True)
        (root / "link").symlink_to(outside, target_is_directory=True)
        with pytest.raises(ValueError, match="refusing to write outside"):
            quarto_render(content=self.SRC, output_dir="link")

    def test_output_dir_on_existing_input_outside_the_root_is_refused(self, root: Path, tmp_path: Path):
        # Render-existing mode: output_dir is relative to the input's dir,
        # which here is outside the root.
        src = tmp_path / "doc.qmd"
        src.write_text(self.SRC, encoding="utf-8")
        with pytest.raises(ValueError, match="refusing to write outside"):
            quarto_render(input=str(src), output_dir="rendered")
