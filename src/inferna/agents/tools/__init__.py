"""Tool registry, definition system, and pre-built tools for inferna agents.

This package was previously a single ``inferna/agents/tools.py`` module
and was split into per-tool files for discoverability. The public
surface is preserved: every name that was importable from
``inferna.agents.tools`` before the split is still importable from it
now via this ``__init__``.

Subpackage layout:

* :mod:`inferna.agents.tools.core` -- framework (``Tool``,
  ``ToolRegistry``, ``@tool``, ``coerce_args``, constraint markers,
  schema generation, docstring extractors).
* :mod:`inferna.agents.tools.calculator` -- safe arithmetic.
* :mod:`inferna.agents.tools.quarto` -- subprocess wrapper around the
  ``quarto`` CLI.
* :mod:`inferna.agents.tools.demo` -- the ``DEMO_TOOLS`` tuple.

Custom tools should be written in their own modules and decorated with
``@tool``; this package is not a registry of all available tools, only
the in-tree reference set.
"""

# ---------------------------------------------------------------------------
# Framework -- everything that used to live at the top of tools.py.
# ---------------------------------------------------------------------------
from .core import (
    # Public errors
    ToolArgumentError,
    ToolTimeoutError,
    # Constraint markers
    Ge,
    Gt,
    Le,
    Lt,
    MaxLen,
    MinLen,
    MultipleOf,
    Pattern,
    # Core types
    Tool,
    ToolRegistry,
    # Public functions
    coerce_args,
    tool,
    # Module logger (kept public for callers that wire their own handler)
    logger,
    # Private helpers re-exported for backwards compat with tests / advanced
    # callers. These were importable from inferna.agents.tools pre-split, so
    # we keep them re-exported here. They are NOT part of the documented
    # public API; treat the leading underscore as the stability signal.
    _apply_constraint_marker,
    _coerce_type,
    _coerce_value,
    _enforce_constraints,
    _extract_epytext_style,
    _extract_google_style,
    _extract_numpy_style,
    _extract_param_description,
    _extract_sphinx_style,
    _generate_schema_from_function,
    _python_type_to_json_schema,
    _python_type_to_json_type,
    _safe_get_type_hints,
)

# ---------------------------------------------------------------------------
# Pre-built tools. Each is a thin module that imports ``tool`` from .core.
# Constants and helpers are re-exported with their leading-underscore names
# preserved so the pre-split import sites keep working.
# ---------------------------------------------------------------------------
from .calculator import (
    _CALC_BINOPS,
    _CALC_MAX_EXPONENT,
    _CALC_MAX_EXPR_LEN,
    _CALC_UNARYOPS,
    _calc_eval,
    calculator,
)
from .quarto import (
    _QUARTO_FORMATS,
    _QUARTO_OUTPUT_CREATED_RE,
    _QUARTO_RENDER_TIMEOUT_SECONDS,
    _QUARTO_SLUG_RE,
    _QUARTO_TITLE_RE,
    _quarto_slug_from_content,
    _quarto_unique_path,
    default_quarto_output_dir,
    quarto_available,
    quarto_render,
)
from .demo import DEMO_TOOLS


__all__ = [
    # Framework
    "Tool",
    "ToolRegistry",
    "ToolArgumentError",
    "ToolTimeoutError",
    "Ge",
    "Gt",
    "Le",
    "Lt",
    "MaxLen",
    "MinLen",
    "MultipleOf",
    "Pattern",
    "coerce_args",
    "tool",
    "logger",
    # Pre-built tools
    "calculator",
    "quarto_render",
    # Quarto helpers (documented public)
    "default_quarto_output_dir",
    "quarto_available",
    # Demo collection
    "DEMO_TOOLS",
]
