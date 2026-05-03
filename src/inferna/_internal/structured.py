"""Compile a ``response_format`` argument into a GBNF grammar + validator.

Supported input shapes:

* ``{"type": "json_object"}`` -- any well-formed JSON value.
* ``{"type": "json_schema", "schema": <dict | pydantic.BaseModel subclass>}``
  -- constrain output to a specific JSON schema. Pydantic v2 models are
  introspected via ``model_json_schema()`` and validated via
  ``model_validate_json``; plain dict schemas are validated via
  ``json.loads`` round-trip only.
* A pydantic ``BaseModel`` subclass directly (shorthand for the above).

The compiled artefact is cached. Schemas are typically reused across
many calls (one schema per agent, one schema per route handler), and
the GBNF compile pass over a non-trivial schema is on the order of
milliseconds -- enough to be worth amortising in tight per-token loops.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Optional

from ..utils.json_schema_to_grammar import json_schema_to_grammar


@dataclass(frozen=True)
class CompiledResponseFormat:
    """Result of compiling a ``response_format`` argument.

    Attributes:
        grammar: GBNF grammar string ready to feed into
            ``LlamaSampler.add_grammar``.
        grammar_root: Root rule name. ``json_schema_to_grammar`` always
            produces ``root``, so this is currently a constant; kept as
            a field to make future evolution explicit.
        validator: Callable that takes the generated text and returns
            the parsed Python object. Raises ``ValueError`` on
            validation failure.
    """

    grammar: str
    grammar_root: str
    validator: Callable[[str], Any]


# Standard "any JSON value" grammar (matches the llama.cpp reference at
# llama.cpp/grammars/json.gbnf). Used for response_format={"type":
# "json_object"} where the caller wants well-formed JSON without
# committing to a schema.
_ANY_JSON_GBNF = r"""
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws

ws ::= | " " | "\n" [ \t]{0,20}
""".strip()


def compile_response_format(rf: Any) -> Optional[CompiledResponseFormat]:
    """Compile a ``response_format`` argument.

    Returns ``None`` when ``rf`` is ``None``. Raises ``ValueError`` /
    ``TypeError`` for malformed input so callers fail fast at the API
    boundary rather than mid-generation.
    """
    if rf is None:
        return None

    # Pydantic shorthand: ``response_format=MyModel``.
    if _is_pydantic_model(rf):
        return _compile_pydantic(rf)

    if not isinstance(rf, dict):
        raise TypeError(
            f"response_format must be None, a dict, or a pydantic BaseModel subclass; got {type(rf).__name__}"
        )

    rf_type = rf.get("type")
    if rf_type == "json_object":
        return _compile_any_json()
    if rf_type == "json_schema":
        schema = rf.get("schema")
        if schema is None:
            raise ValueError("response_format with type='json_schema' requires a 'schema' field")
        if _is_pydantic_model(schema):
            return _compile_pydantic(schema)
        if isinstance(schema, dict):
            return _compile_dict_schema(schema)
        raise TypeError(
            f"response_format['schema'] must be a dict or pydantic BaseModel subclass; got {type(schema).__name__}"
        )

    raise ValueError(f"response_format['type'] must be 'json_object' or 'json_schema'; got {rf_type!r}")


def _is_pydantic_model(obj: Any) -> bool:
    """True if ``obj`` is a pydantic v2 ``BaseModel`` subclass.

    Done structurally to avoid importing pydantic at module import time
    (it's an optional dependency).
    """
    return isinstance(obj, type) and hasattr(obj, "model_json_schema") and hasattr(obj, "model_validate_json")


_ANY_JSON_COMPILED: Optional[CompiledResponseFormat] = None


def _compile_any_json() -> CompiledResponseFormat:
    global _ANY_JSON_COMPILED
    if _ANY_JSON_COMPILED is None:
        _ANY_JSON_COMPILED = CompiledResponseFormat(
            grammar=_ANY_JSON_GBNF,
            grammar_root="root",
            validator=_dict_validator,
        )
    return _ANY_JSON_COMPILED


def _compile_dict_schema(schema: dict[str, Any]) -> CompiledResponseFormat:
    # Sort keys so semantically equal schemas hash to the same cache slot.
    key = json.dumps(schema, sort_keys=True, default=str)
    return _compile_dict_schema_cached(key)


@lru_cache(maxsize=128)
def _compile_dict_schema_cached(schema_json: str) -> CompiledResponseFormat:
    """Compile a JSON-serialised schema string. Cached on the string."""
    schema = json.loads(schema_json)
    grammar = json_schema_to_grammar(schema)
    return CompiledResponseFormat(
        grammar=grammar,
        grammar_root="root",
        validator=_dict_validator,
    )


def _dict_validator(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Generated text is not valid JSON: {exc}") from exc


def _compile_pydantic(model_cls: Any) -> CompiledResponseFormat:
    # Pydantic classes are hashable by identity; cache on id().
    return _compile_pydantic_cached(id(model_cls), model_cls)


# Hold a separate small cache: keyed on id() to avoid re-running
# model_json_schema() on every call. Misses are rare in practice.
_pydantic_cache: dict[int, CompiledResponseFormat] = {}


def _compile_pydantic_cached(model_id: int, model_cls: Any) -> CompiledResponseFormat:
    cached = _pydantic_cache.get(model_id)
    if cached is not None:
        return cached
    schema = model_cls.model_json_schema()
    grammar = json_schema_to_grammar(schema)
    compiled = CompiledResponseFormat(
        grammar=grammar,
        grammar_root="root",
        validator=_make_pydantic_validator(model_cls),
    )
    _pydantic_cache[model_id] = compiled
    return compiled


def _make_pydantic_validator(model_cls: Any) -> Callable[[str], Any]:
    def validate(text: str) -> Any:
        try:
            return model_cls.model_validate_json(text)
        except Exception as exc:  # pydantic.ValidationError + json errors
            raise ValueError(f"Generated text failed {model_cls.__name__} validation: {exc}") from exc

    return validate
