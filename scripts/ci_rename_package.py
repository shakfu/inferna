#!/usr/bin/env python3
"""Rename the project's package name in pyproject.toml for CI variants.

Workflows that build GPU variants (cuda12, cuda13, rocm, sycl, vulkan) ship
distinct PyPI distributions whose `name = "..."` differs from the canonical
`inferna`. Previously each workflow embedded its own `sed` invocation; that
relies on exact whitespace and silently no-ops if pyproject is reformatted.

Centralizing here:
- single regex/anchor across all workflows,
- explicit allowlist of accepted variant names,
- post-write verification (read back, confirm new name, fail loudly).

Usage:
    python scripts/ci_rename_package.py inferna-cuda12
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ALLOWED_VARIANTS = {
    "inferna",
    "inferna-cuda12",
    "inferna-cuda13",
    "inferna-rocm",
    "inferna-sycl",
    "inferna-vulkan",
    "inferna-opencl",
    "inferna-hip",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "new_name",
        help="Target package name (must be in the allowed-variants set)",
    )
    parser.add_argument(
        "--pyproject",
        default="pyproject.toml",
        help="Path to pyproject.toml (default: ./pyproject.toml)",
    )
    args = parser.parse_args()

    if args.new_name not in ALLOWED_VARIANTS:
        sys.stderr.write(
            f"ERROR: '{args.new_name}' is not in the allowed-variants set: "
            f"{sorted(ALLOWED_VARIANTS)}\n"
        )
        return 2

    path = Path(args.pyproject)
    if not path.is_file():
        sys.stderr.write(f"ERROR: pyproject not found: {path}\n")
        return 2

    text = path.read_text()
    # Match the canonical form; allow either single or double quotes.
    pattern = re.compile(r'^name\s*=\s*"inferna"\s*$', re.MULTILINE)
    match = pattern.search(text)
    if not match:
        sys.stderr.write(
            f'ERROR: could not find canonical `name = "inferna"` line in {path}. '
            'Was it already renamed, or has the file been reformatted?\n'
        )
        return 1

    new_text = pattern.sub(f'name = "{args.new_name}"', text, count=1)
    path.write_text(new_text)

    # Verify by re-reading.
    after = path.read_text()
    verify = re.compile(rf'^name\s*=\s*"{re.escape(args.new_name)}"\s*$', re.MULTILINE)
    if not verify.search(after):
        sys.stderr.write(
            f"ERROR: post-write verification failed — `name = \"{args.new_name}\"` "
            f"not found in {path} after substitution.\n"
        )
        return 1

    print(f'Renamed package: inferna -> {args.new_name} in {path}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
