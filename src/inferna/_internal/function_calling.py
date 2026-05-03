"""Compile OpenAI-shaped ``tools`` + ``tool_choice`` into a grammar.

Wire format matches OpenAI's chat-completion tool-calling API:

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        },
        ...
    ]

    tool_choice = "required"                            # must call a tool
    tool_choice = {"type": "function",
                   "function": {"name": "get_weather"}} # must call this one
    tool_choice = "auto"                                # model decides
    tool_choice = "none"                                # forbid tool calls

The compiled artefact reuses :class:`CompiledResponseFormat` from
``structured.py`` so it threads through the existing ``response_format=``
plumbing on ``LLM`` without a second sampler-builder code path.

Output shape produced by the model:
  * ``required`` / specific-function: ``{"name": "...", "arguments": {...}}``
    -- always exactly one tool call, structurally guaranteed.
  * ``auto``: an envelope ``{"tool_call": {...}}`` *or*
    ``{"content": "..."}``. The model picks; the validator routes the
    parsed result onto ``Response.tool_calls`` or ``Response.text``
    depending on which arm fired.

The validator returns a ``CompiledToolResult`` so :class:`LLM` can
attach :attr:`Response.tool_calls` (and override the rendered text in
auto-mode when the model chose the content path).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Union

from .structured import CompiledResponseFormat


# ----------------------------------------------------------------------
# Result types (mirrored on inferna.api by the LLM caller)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class FunctionCall:
    name: str
    arguments: str  # JSON-serialised, matching OpenAI's wire format


@dataclass(frozen=True)
class ToolCall:
    id: str
    type: str  # currently always "function"
    function: FunctionCall


@dataclass(frozen=True)
class CompiledToolResult:
    """Validator return type when ``tools=`` was supplied.

    Exactly one of ``tool_calls`` / ``content`` is populated:
      * ``required`` / specific-function -> ``tool_calls`` (length 1).
      * ``auto`` -> whichever arm the model chose.
    """

    tool_calls: List[ToolCall]
    content: Optional[str]


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def compile_tools(
    tools: List[Dict[str, Any]],
    tool_choice: Union[str, Dict[str, Any]] = "required",
) -> CompiledResponseFormat:
    """Compile OpenAI-shaped ``tools`` + ``tool_choice`` into a grammar.

    Returns a :class:`CompiledResponseFormat` whose validator yields a
    :class:`CompiledToolResult`. Caller (typically ``LLM._generate``)
    inspects the result and attaches it to ``Response.tool_calls``.

    Raises:
        ValueError: malformed ``tools`` / ``tool_choice``, or
            ``tool_choice="none"`` (which the caller should handle by
            simply not passing ``tools=``).
        TypeError: wrong argument types.
    """
    if not isinstance(tools, list) or not tools:
        raise ValueError("tools must be a non-empty list")

    # Index tools by name (and validate shape) up front so the schema
    # builders can assume well-formed input.
    by_name: Dict[str, Dict[str, Any]] = {}
    for i, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise TypeError(f"tools[{i}] must be a dict, got {type(tool).__name__}")
        if tool.get("type") != "function":
            raise ValueError(f"tools[{i}].type must be 'function', got {tool.get('type')!r}")
        fn = tool.get("function")
        if not isinstance(fn, dict):
            raise TypeError(f"tools[{i}].function must be a dict")
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"tools[{i}].function.name must be a non-empty string")
        if name in by_name:
            raise ValueError(f"duplicate tool name: {name!r}")
        # parameters is optional; default to "no arguments" when absent.
        params = fn.get("parameters") or {"type": "object", "properties": {}}
        if not isinstance(params, dict):
            raise TypeError(f"tools[{i}].function.parameters must be a dict")
        by_name[name] = params

    # Resolve tool_choice into either a list of allowed tool names
    # (forced) or the special "auto" mode.
    forced_names, mode = _resolve_choice(tool_choice, by_name)
    if mode == "none":
        raise ValueError(
            "tool_choice='none' should be handled by omitting tools= entirely; "
            "compile_tools cannot return a no-op grammar"
        )

    if mode == "required":
        schema = _build_required_schema(forced_names, by_name)
        return CompiledResponseFormat(
            grammar=_schema_to_grammar(schema),
            grammar_root="root",
            validator=_make_required_validator(set(forced_names)),
        )

    # mode == "auto"
    schema = _build_auto_schema(by_name)
    return CompiledResponseFormat(
        grammar=_schema_to_grammar(schema),
        grammar_root="root",
        validator=_make_auto_validator(set(by_name.keys())),
    )


# ----------------------------------------------------------------------
# tool_choice resolution
# ----------------------------------------------------------------------


def _resolve_choice(
    tool_choice: Union[str, Dict[str, Any]],
    by_name: Dict[str, Dict[str, Any]],
) -> tuple[List[str], str]:
    """Return (allowed_names, mode) where mode in {required, auto, none}."""
    if isinstance(tool_choice, str):
        if tool_choice == "required":
            return list(by_name.keys()), "required"
        if tool_choice == "auto":
            return list(by_name.keys()), "auto"
        if tool_choice == "none":
            return [], "none"
        raise ValueError(f"tool_choice string must be 'required' | 'auto' | 'none', got {tool_choice!r}")

    if isinstance(tool_choice, dict):
        if tool_choice.get("type") != "function":
            raise ValueError(f"tool_choice dict must have type='function', got {tool_choice.get('type')!r}")
        fn = tool_choice.get("function")
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            raise ValueError("tool_choice dict must have function.name as a string")
        name = fn["name"]
        if name not in by_name:
            raise ValueError(f"tool_choice.function.name={name!r} is not in tools")
        return [name], "required"

    raise TypeError(f"tool_choice must be a string or dict, got {type(tool_choice).__name__}")


# ----------------------------------------------------------------------
# Schema builders
# ----------------------------------------------------------------------


def _build_required_schema(
    allowed_names: List[str],
    by_name: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Schema: oneOf({name=tool_i, arguments=tool_i.parameters}).

    Constrains each branch to its own tool's parameters schema, so
    ``arguments`` cannot be valid for the wrong tool.
    """
    branches = []
    for name in allowed_names:
        branches.append(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "const": name},
                    "arguments": by_name[name],
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
        )
    if len(branches) == 1:
        return branches[0]
    return {"oneOf": branches}


def _build_auto_schema(by_name: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Schema: {tool_call: {...}} | {content: str}.

    The envelope shape lets the model pick between calling a tool and
    answering directly. Two top-level branches keep the grammar small
    and the validator's routing trivial.
    """
    tool_call_branch = _build_required_schema(list(by_name.keys()), by_name)
    return {
        "oneOf": [
            {
                "type": "object",
                "properties": {"tool_call": tool_call_branch},
                "required": ["tool_call"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
                "additionalProperties": False,
            },
        ]
    }


def _schema_to_grammar(schema: Dict[str, Any]) -> str:
    # Local import keeps the json_schema_to_grammar module out of the
    # import-time path for callers who never use function calling.
    from ..utils.json_schema_to_grammar import json_schema_to_grammar

    return json_schema_to_grammar(schema)


# ----------------------------------------------------------------------
# Validators (text -> CompiledToolResult)
# ----------------------------------------------------------------------


def _new_call_id() -> str:
    # Match OpenAI's "call_<24 hex chars>" shape closely enough for
    # downstream parity tooling.
    return "call_" + uuid.uuid4().hex[:24]


def _make_required_validator(allowed: Set[str]) -> Callable[[str], CompiledToolResult]:
    def validate(text: str) -> CompiledToolResult:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Tool call output is not valid JSON: {exc}") from exc
        return CompiledToolResult(
            tool_calls=[_object_to_tool_call(obj, allowed)],
            content=None,
        )

    return validate


def _make_auto_validator(allowed: Set[str]) -> Callable[[str], CompiledToolResult]:
    def validate(text: str) -> CompiledToolResult:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Tool/content envelope is not valid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError("auto-mode envelope must be a JSON object")
        if "tool_call" in obj:
            return CompiledToolResult(
                tool_calls=[_object_to_tool_call(obj["tool_call"], allowed)],
                content=None,
            )
        if "content" in obj:
            content = obj["content"]
            if not isinstance(content, str):
                raise ValueError("auto-mode 'content' must be a string")
            return CompiledToolResult(tool_calls=[], content=content)
        raise ValueError("auto-mode envelope must contain 'tool_call' or 'content'")

    return validate


def _object_to_tool_call(obj: Any, allowed: Set[str]) -> ToolCall:
    if not isinstance(obj, dict):
        raise ValueError("tool call must be a JSON object")
    name = obj.get("name")
    args = obj.get("arguments")
    if not isinstance(name, str):
        raise ValueError("tool call 'name' must be a string")
    if name not in allowed:
        raise ValueError(f"tool call name {name!r} is not in the allowed set {sorted(allowed)}")
    # OpenAI's wire format keeps arguments as a JSON-serialised string;
    # we mirror that so callers piping into OpenAI clients get identical
    # output. ``json.dumps`` round-trips a dict cleanly.
    args_str = json.dumps(args, separators=(",", ":"))
    return ToolCall(id=_new_call_id(), type="function", function=FunctionCall(name=name, arguments=args_str))
