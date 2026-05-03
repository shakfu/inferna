"""Tests for ``response_format`` (structured outputs).

Covers the three accepted shapes:
  * ``{"type": "json_object"}`` -- arbitrary JSON.
  * ``{"type": "json_schema", "schema": <dict>}`` -- strict dict schema.
  * pydantic ``BaseModel`` subclass -- shorthand for the above.

Plus the error paths (malformed input) and the contract that
``response_format=None`` leaves ``Response.parsed`` at ``None``.
"""

from __future__ import annotations

import gc
import json

import pytest
from pydantic import BaseModel

from inferna import LLM
from inferna._internal.structured import (
    CompiledResponseFormat,
    compile_response_format,
)


# ----------------------------------------------------------------------
# Pure-Python compile_response_format (no model required)
# ----------------------------------------------------------------------


class TestCompileBasic:
    def test_none_returns_none(self) -> None:
        assert compile_response_format(None) is None

    def test_json_object_returns_compiled(self) -> None:
        c = compile_response_format({"type": "json_object"})
        assert isinstance(c, CompiledResponseFormat)
        assert "root" in c.grammar
        assert c.grammar_root == "root"

    def test_json_object_validator_accepts_json(self) -> None:
        c = compile_response_format({"type": "json_object"})
        assert c is not None
        assert c.validator('{"x": 1}') == {"x": 1}
        assert c.validator("[1, 2, 3]") == [1, 2, 3]

    def test_json_object_validator_rejects_garbage(self) -> None:
        c = compile_response_format({"type": "json_object"})
        assert c is not None
        with pytest.raises(ValueError, match="not valid JSON"):
            c.validator("not json")

    def test_json_schema_dict(self) -> None:
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        c = compile_response_format({"type": "json_schema", "schema": schema})
        assert c is not None
        assert c.validator('{"x": 1}') == {"x": 1}

    def test_json_schema_caching_is_keyed_by_value(self) -> None:
        # Two semantically equal schemas (different dict identity) should
        # land in the same cache slot.
        s1 = {"type": "object", "properties": {"a": {"type": "string"}}}
        s2 = dict(s1)
        c1 = compile_response_format({"type": "json_schema", "schema": s1})
        c2 = compile_response_format({"type": "json_schema", "schema": s2})
        assert c1 is c2  # identity, courtesy of lru_cache


class TestCompilePydantic:
    def test_pydantic_shorthand(self) -> None:
        class M(BaseModel):
            x: int

        c = compile_response_format(M)
        assert c is not None
        assert "root" in c.grammar
        out = c.validator('{"x": 7}')
        assert isinstance(out, M)
        assert out.x == 7

    def test_pydantic_in_json_schema_wrapper(self) -> None:
        class M(BaseModel):
            name: str

        c = compile_response_format({"type": "json_schema", "schema": M})
        assert c is not None
        out = c.validator('{"name": "alice"}')
        assert isinstance(out, M)

    def test_pydantic_validator_rejects_invalid(self) -> None:
        class M(BaseModel):
            x: int

        c = compile_response_format(M)
        assert c is not None
        with pytest.raises(ValueError, match="failed M validation"):
            c.validator('{"x": "not-an-int"}')


class TestCompileErrors:
    def test_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="must be 'json_object' or 'json_schema'"):
            compile_response_format({"type": "bogus"})

    def test_json_schema_missing_schema_field(self) -> None:
        with pytest.raises(ValueError, match="requires a 'schema' field"):
            compile_response_format({"type": "json_schema"})

    def test_non_dict_non_pydantic(self) -> None:
        with pytest.raises(TypeError, match="must be None, a dict"):
            compile_response_format("not-a-spec")  # type: ignore[arg-type]

    def test_schema_field_wrong_type(self) -> None:
        with pytest.raises(TypeError, match=r"schema'\] must be a dict"):
            compile_response_format({"type": "json_schema", "schema": 42})


# ----------------------------------------------------------------------
# Live integration with a model
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def llm(model_path: str):
    instance = LLM(model_path, verbose=False)
    yield instance
    instance.close()
    del instance
    gc.collect()


class TestResponseFormatLive:
    def test_json_object_mode(self, llm: LLM) -> None:
        r = llm(
            "Return a JSON object with fields name and age. JSON only:",
            response_format={"type": "json_object"},
        )
        assert r.parsed is not None
        # Round-trip: text must parse to the same shape.
        assert json.loads(r.text) == r.parsed

    def test_json_schema_mode(self, llm: LLM) -> None:
        schema = {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "population": {"type": "integer"},
            },
            "required": ["city", "population"],
        }
        r = llm(
            "Tell me about Tokyo. JSON only:",
            response_format={"type": "json_schema", "schema": schema},
        )
        assert isinstance(r.parsed, dict)
        assert "city" in r.parsed and "population" in r.parsed
        assert isinstance(r.parsed["city"], str)
        assert isinstance(r.parsed["population"], int)

    def test_pydantic_shorthand_mode(self, llm: LLM) -> None:
        class Person(BaseModel):
            name: str
            age: int

        r = llm("Make up a person. JSON only:", response_format=Person)
        assert isinstance(r.parsed, Person)
        assert isinstance(r.parsed.name, str)
        assert isinstance(r.parsed.age, int)

    def test_no_response_format_leaves_parsed_none(self, llm: LLM) -> None:
        r = llm("Say hi.")
        assert r.parsed is None

    def test_response_format_bypasses_cache(self, llm: LLM) -> None:
        # Two calls with the same prompt + response_format should both
        # produce parsed output; we don't assert text equality (sampling)
        # but we do assert that neither call returns the literal cached
        # Response from a prior plain-text call.
        plain = llm("hi")
        with_fmt = llm("hi", response_format={"type": "json_object"})
        assert plain.parsed is None
        assert with_fmt.parsed is not None
