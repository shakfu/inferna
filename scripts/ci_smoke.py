#!/usr/bin/env python3
# mypy: ignore-errors
"""Smoke-test imports for the inferna wheel under test.

Walked through after `pip install <wheel>` in CI; validates that the
package's public API surface and nanobind extensions actually load on
the runner. Optional extensions (whisper, sd) are tolerated as missing
but logged. Returns nonzero exit code if any required import fails.

Centralized so the same checks run identically across:
- build-cibw.yml smoke job
- build-cibw-abi3.yml smoke job
- build-gpu-wheels.yml smoke job
- build-gpu-wheels-abi3.yml smoke job
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        default="",
        help="Optional tag (e.g. backend name) appended to the success line.",
    )
    args = parser.parse_args()

    # Required: the package + the public API + the llama nanobind extension.
    # The whole point of this script is to verify the runtime imports
    # work, not to satisfy static analysis (file uses `# mypy:
    # ignore-errors` for that reason).
    import inferna

    print(f"inferna {inferna.__version__} imported on Python {sys.version.split()[0]}")

    from inferna import (  # noqa: F401
        LLM,
        complete,
        chat,
        AsyncLLM,
        batch_generate,
        BatchGenerator,
        estimate_gpu_layers,
        estimate_memory_usage,
        GenerationConfig,
        Response,
    )
    from inferna.llama import llama_cpp  # noqa: F401

    print("llama_cpp extension loaded")

    from inferna.agents import (  # noqa: F401
        ReActAgent,
        ConstrainedAgent,
        ContractAgent,
        tool,
    )

    print("agents loaded")

    from inferna.integrations.openai_compat import OpenAICompatibleClient  # noqa: F401

    print("integrations loaded")

    # Optional extensions — log but don't fail.
    try:
        from inferna.whisper import whisper_cpp  # noqa: F401

        print("whisper_cpp extension loaded")
    except ImportError as exc:
        print(f"whisper_cpp not available: {exc}")

    try:
        from inferna.sd import stable_diffusion  # noqa: F401

        print("stable_diffusion extension loaded")
    except ImportError as exc:
        print(f"stable_diffusion not available: {exc}")

    suffix = f" ({args.label})" if args.label else ""
    print(f"All imports passed{suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
