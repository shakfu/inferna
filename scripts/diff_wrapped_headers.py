#!/usr/bin/env python3
"""Diff the upstream headers wrapped by inferna between two refs and
concatenate the per-header diffs into a single ``changes.diff`` document.

For each third-party project (llama.cpp, whisper.cpp, stable-diffusion.cpp)
the script runs ``git diff <from>..<to> -- <header>`` inside the project's
build checkout, then writes the combined output to ``changes.diff`` at the
inferna repo root.

Examples:
    # Compare llama.cpp b9025 -> b9190, leave others on HEAD vs latest tag
    scripts/diff_wrapped_headers.py --llama b9025 b9190

    # Pin every project explicitly, write to a custom path
    scripts/diff_wrapped_headers.py \\
        --llama b9025 b9190 \\
        --whisper v1.7.4 v1.7.5 \\
        --sd master-596-90e87bc master-612-d7ecbe1 \\
        --output docs/wrapper-deltas.diff
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Project:
    name: str  # short name used for CLI flag (e.g. "llama")
    checkout: Path  # path to the git checkout (relative to REPO_ROOT)
    headers: tuple[str, ...]  # header paths relative to the checkout root


# Headers actually #include'd by the inferna native bindings. Keep this list
# in sync with the includes in src/inferna/*/_*_native*.cpp.
PROJECTS: tuple[Project, ...] = (
    Project(
        name="llama",
        checkout=Path("build/llama.cpp"),
        headers=(
            "include/llama.h",
            "ggml/include/ggml-backend.h",
            "ggml/include/ggml-cpu.h",
            "ggml/include/gguf.h",
            "tools/mtmd/mtmd.h",
            "tools/mtmd/mtmd-helper.h",
        ),
    ),
    Project(
        name="whisper",
        checkout=Path("build/whisper.cpp"),
        headers=("include/whisper.h",),
    ),
    Project(
        name="sd",
        checkout=Path("build/stable-diffusion.cpp"),
        headers=("include/stable-diffusion.h",),
    ),
)


def run_git(checkout: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def previous_tag(checkout: Path) -> str | None:
    """Return the tag immediately before the current HEAD, if any."""
    try:
        out = run_git(checkout, "describe", "--tags", "--abbrev=0", "HEAD~1")
        return out.strip() or None
    except subprocess.CalledProcessError:
        return None


def current_ref(checkout: Path) -> str:
    """Return the most descriptive ref for HEAD (tag if available, else SHA)."""
    try:
        out = run_git(checkout, "describe", "--tags", "--always", "HEAD")
        return out.strip()
    except subprocess.CalledProcessError:
        return run_git(checkout, "rev-parse", "HEAD").strip()


def resolve_range(checkout: Path, raw: list[str] | None) -> tuple[str, str] | None:
    """Return (from, to) refs for a project.

    If ``raw`` is provided (from ``--llama``/``--whisper``/``--sd``), use it
    verbatim. Otherwise fall back to the tag immediately before HEAD plus
    HEAD itself. Returns None if no usable FROM can be determined.
    """
    if raw:
        return raw[0], raw[1]

    frm = previous_tag(checkout)
    if not frm:
        return None
    return frm, current_ref(checkout)


def diff_header(checkout: Path, frm: str, to: str, header: str) -> str:
    """Return the ``git diff`` for a single header, or an empty string."""
    if not (checkout / header).exists():
        return ""
    return run_git(
        checkout,
        "diff",
        f"{frm}..{to}",
        "--",
        header,
    )


def main() -> int:
    if not shutil.which("git"):
        sys.exit("error: git not found on PATH")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for proj in PROJECTS:
        parser.add_argument(
            f"--{proj.name}",
            nargs=2,
            metavar=("FROM", "TO"),
            help=(f"two refs (tags, branches, or SHAs) to diff in {proj.checkout}. Defaults to <previous tag>..HEAD."),
        )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=REPO_ROOT / "changes.diff",
        help="output file (default: changes.diff at repo root)",
    )
    args = parser.parse_args()

    sections: list[str] = []
    summary: list[str] = []

    for proj in PROJECTS:
        checkout = REPO_ROOT / proj.checkout
        if not (checkout / ".git").exists():
            summary.append(f"  {proj.name:<8} SKIP (no git checkout at {proj.checkout})")
            continue

        rng = resolve_range(checkout, getattr(args, proj.name))
        if rng is None:
            summary.append(f"  {proj.name:<8} SKIP (no prior tag; pass --{proj.name} FROM TO)")
            continue
        frm, to = rng

        proj_chunks: list[str] = []
        for header in proj.headers:
            diff = diff_header(checkout, frm, to, header)
            if diff:
                proj_chunks.append(f"### {header}\n\n{diff}")

        header_line = f"## {proj.name} ({proj.checkout}) {frm}..{to}\n"
        if proj_chunks:
            sections.append(header_line + "\n" + "\n".join(proj_chunks))
            summary.append(f"  {proj.name:<8} {frm}..{to}  ({len(proj_chunks)} header(s) changed)")
        else:
            sections.append(header_line + "\n(no changes in wrapped headers)\n")
            summary.append(f"  {proj.name:<8} {frm}..{to}  (no changes)")

    if not sections:
        sys.exit("error: nothing to diff (no checkouts found)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("# Wrapped-header diffs\n\n" + "\n\n".join(sections) + "\n")

    print(f"wrote {args.output}")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
