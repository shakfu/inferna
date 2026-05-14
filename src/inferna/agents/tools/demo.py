"""``DEMO_TOOLS`` -- the minimal reference set of pre-built tools that
ships with inferna.

These tools are auto-registered when ``inferna chat`` invokes any
``/agent*`` command, and are exported from ``inferna.agents`` so library
users can pass them to their own agents (or omit them entirely).

* :func:`calculator`    -- safe expression evaluation via an AST allowlist.
* :func:`quarto_render` -- subprocess + filesystem write with two usage modes.

Tuples are used so consumers can't mutate the shared list and
accidentally desync different agents in the same process.

``quarto_render`` is included unconditionally; runtime checks raise a
clear "install quarto" error if the binary is missing, so adding it to
the collection doesn't make non-quarto installs surface weird failures
at import time. Agent surfaces that want to hide unavailable tools
entirely should gate on :func:`inferna.agents.tools.quarto_available`
themselves.
"""

from typing import Tuple

from .calculator import calculator
from .core import Tool
from .quarto import quarto_render


DEMO_TOOLS: Tuple[Tool, ...] = (
    calculator,
    quarto_render,
)
