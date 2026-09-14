"""Documentation must not show ``eval`` on model-produced input.

Tool arguments come from the LLM, so a copied ``eval`` example runs whatever a
prompt-injected document asks for. Use ``inferna.agents.tools.calculator``.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL_CALL = re.compile(r"(?<![\w.])eval\(")
DOCS = sorted(
    [ROOT / "README.md", *(ROOT / "docs").rglob("*.md"), *(ROOT / "src" / "inferna").rglob("README.md")],
)


@pytest.mark.parametrize("path", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_eval_in_docs(path):
    hits = [
        f"{i}: {line.strip()}"
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if EVAL_CALL.search(line)
    ]
    assert not hits, f"eval( in {path.relative_to(ROOT)}:\n" + "\n".join(hits)
