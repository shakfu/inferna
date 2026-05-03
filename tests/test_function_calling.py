"""Tests for OpenAI-compatible function calling.

Two layers:
  1. ``compile_tools`` (pure-Python): grammar generation, validator
     correctness, error paths.
  2. Live integration via ``LLM(..., tools=..., tool_choice=...)``: end-to-end
     contract that the model produces a parseable tool call and that
     ``Response.tool_calls`` is populated.
"""

from __future__ import annotations

import gc
import json

import pytest

from inferna import LLM, FunctionCall, ToolCall
from inferna._internal.function_calling import (
    CompiledToolResult,
    compile_tools,
)


# ----------------------------------------------------------------------
# compile_tools — pure-Python (no model required)
# ----------------------------------------------------------------------


@pytest.fixture
def two_tools() -> list[dict]:
    return [
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
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two integers.",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            },
        },
    ]


class TestCompileGrammar:
    def test_required_grammar_compiles(self, two_tools) -> None:
        c = compile_tools(two_tools, tool_choice="required")
        assert c.grammar_root == "root"
        # Both tool names appear in the grammar (as quoted JSON literals
        # inside name rules; substring check is sufficient).
        assert "get_weather" in c.grammar
        assert "add" in c.grammar

    def test_specific_function_grammar_constrains_to_one(self, two_tools) -> None:
        c = compile_tools(
            two_tools,
            tool_choice={"type": "function", "function": {"name": "add"}},
        )
        # Only 'add' should appear; 'get_weather' must not.
        assert "add" in c.grammar
        assert "get_weather" not in c.grammar

    def test_auto_grammar_includes_both_envelope_branches(self, two_tools) -> None:
        c = compile_tools(two_tools, tool_choice="auto")
        # Envelope keys appear in the grammar.
        assert "tool_call" in c.grammar
        assert "content" in c.grammar


class TestCompileValidators:
    def test_required_validator_parses_tool_call(self, two_tools) -> None:
        c = compile_tools(two_tools, tool_choice="required")
        result = c.validator('{"name": "add", "arguments": {"a": 1, "b": 2}}')
        assert isinstance(result, CompiledToolResult)
        assert result.content is None
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert isinstance(tc, ToolCall)
        assert tc.type == "function"
        assert tc.function.name == "add"
        assert json.loads(tc.function.arguments) == {"a": 1, "b": 2}
        assert tc.id.startswith("call_")

    def test_required_rejects_unknown_tool_name(self, two_tools) -> None:
        # The grammar would prevent this output in practice; the
        # validator backstops bugs in the grammar / forced-output paths.
        c = compile_tools(two_tools, tool_choice="required")
        with pytest.raises(ValueError, match="not in the allowed set"):
            c.validator('{"name": "delete_db", "arguments": {}}')

    def test_required_rejects_invalid_json(self, two_tools) -> None:
        c = compile_tools(two_tools, tool_choice="required")
        with pytest.raises(ValueError, match="not valid JSON"):
            c.validator("not json")

    def test_auto_routes_tool_call_branch(self, two_tools) -> None:
        c = compile_tools(two_tools, tool_choice="auto")
        result = c.validator('{"tool_call": {"name": "add", "arguments": {"a": 1, "b": 2}}}')
        assert result.content is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "add"

    def test_auto_routes_content_branch(self, two_tools) -> None:
        c = compile_tools(two_tools, tool_choice="auto")
        result = c.validator('{"content": "Hello!"}')
        assert result.content == "Hello!"
        assert result.tool_calls == []

    def test_auto_rejects_unknown_envelope_key(self, two_tools) -> None:
        c = compile_tools(two_tools, tool_choice="auto")
        with pytest.raises(ValueError, match="must contain"):
            c.validator('{"other": "x"}')


class TestCompileErrors:
    def test_empty_tools_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty list"):
            compile_tools([], tool_choice="required")

    def test_missing_function_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            compile_tools([{"type": "function", "function": {}}], tool_choice="required")

    def test_duplicate_tool_names(self) -> None:
        tool = {
            "type": "function",
            "function": {"name": "x", "parameters": {"type": "object", "properties": {}}},
        }
        with pytest.raises(ValueError, match="duplicate tool name"):
            compile_tools([tool, tool], tool_choice="required")

    def test_invalid_tool_type(self) -> None:
        with pytest.raises(ValueError, match="must be 'function'"):
            compile_tools(
                [{"type": "retriever", "function": {"name": "x"}}],
                tool_choice="required",
            )

    def test_tool_choice_string_unknown(self, two_tools) -> None:
        with pytest.raises(ValueError, match="must be 'required'"):
            compile_tools(two_tools, tool_choice="bogus")

    def test_tool_choice_none_raises(self, two_tools) -> None:
        # Caller should omit tools= entirely; compile_tools refuses.
        with pytest.raises(ValueError, match="should be handled by omitting"):
            compile_tools(two_tools, tool_choice="none")

    def test_tool_choice_unknown_function_name(self, two_tools) -> None:
        with pytest.raises(ValueError, match="not in tools"):
            compile_tools(
                two_tools,
                tool_choice={"type": "function", "function": {"name": "missing"}},
            )

    def test_tool_choice_wrong_type(self, two_tools) -> None:
        with pytest.raises(TypeError, match="must be a string or dict"):
            compile_tools(two_tools, tool_choice=42)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Live integration (model required)
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def llm(model_path: str):
    instance = LLM(model_path, verbose=False)
    yield instance
    instance.close()
    del instance
    gc.collect()


@pytest.fixture
def live_tools() -> list[dict]:
    return [
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
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two integers.",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            },
        },
    ]


class TestFunctionCallingLive:
    def test_required_yields_tool_call(self, llm: LLM, live_tools) -> None:
        r = llm("What is the weather in Paris?", tools=live_tools, tool_choice="required")
        assert r.tool_calls is not None
        assert len(r.tool_calls) == 1
        tc = r.tool_calls[0]
        assert isinstance(tc, ToolCall)
        assert isinstance(tc.function, FunctionCall)
        assert tc.function.name in {"get_weather", "add"}
        # Arguments are JSON-serialised string per OpenAI wire format.
        parsed_args = json.loads(tc.function.arguments)
        assert isinstance(parsed_args, dict)

    def test_specific_function_constrains_name(self, llm: LLM, live_tools) -> None:
        r = llm(
            "Compute something.",
            tools=live_tools,
            tool_choice={"type": "function", "function": {"name": "add"}},
        )
        assert r.tool_calls is not None
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].function.name == "add"
        # arguments must conform to add's schema (a, b are integers).
        args = json.loads(r.tool_calls[0].function.arguments)
        assert isinstance(args.get("a"), int)
        assert isinstance(args.get("b"), int)

    def test_auto_mode_returns_tool_calls_field(self, llm: LLM, live_tools) -> None:
        # Don't assert which arm fires (model decides). Just assert the
        # field is populated rather than None.
        r = llm("What's the weather in Tokyo?", tools=live_tools, tool_choice="auto")
        assert r.tool_calls is not None  # could be [] if model chose content

    def test_no_tools_leaves_tool_calls_none(self, llm: LLM) -> None:
        r = llm("Say hi.")
        assert r.tool_calls is None


class TestApiSurface:
    def test_tools_with_response_format_raises(self, llm: LLM, live_tools) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            llm("hi", tools=live_tools, response_format={"type": "json_object"})

    def test_tools_with_stream_raises(self, llm: LLM, live_tools) -> None:
        with pytest.raises(NotImplementedError, match="not supported yet"):
            llm("hi", tools=live_tools, stream=True)
