"""Single-source chat-template renderer.

The two-tier (Jinja → C-API) fallback ladder previously lived inline in
three places: ``LLM._apply_template`` / ``LLM._apply_jinja_template``
(``api.py``), the standalone ``apply_chat_template`` /
``_apply_jinja_template_standalone`` pair (also ``api.py``), and
``Chat._apply_template`` (``llama/chat.py``). All three are now thin
wrappers over :func:`apply_template` here.

The pipeline:

  1. **Vendored jinja2** -- evaluates the model's embedded chat template
     in an ``ImmutableSandboxedEnvironment`` (matches HuggingFace's
     transformers semantics; handles every GGUF whose template is
     Jinja-shaped, including Gemma 4 / Qwen3). Tried only when the
     caller didn't pass an explicit ``template=`` argument, since named
     templates are a feature of the legacy path.

  2. **Legacy C-API** -- ``llama_chat_apply_template`` over the
     substring-heuristic table built into llama.cpp.

  3. **Plain ``role: content``** -- fallback when the model has no
     embedded template at all.

The Jinja path's exception handling matches the (post-fix) ladder used
by the LLM instance method: ``_JinjaTemplateError`` (intentional
template raises like Gemma's ``raise_exception``) silently degrades to
the C-API path; ``(TypeError, KeyError, AttributeError)`` (recoverable
render-context issues) degrades but logs a warning so silent
miscompilations are observable; everything else propagates.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

# Re-export the Jinja TemplateError under a stable internal name so
# callers can ``except _JinjaTemplateError`` without depending on the
# vendored path. Same trick used in api.py / chat.py prior to the
# extraction.
from .._vendor.jinja2.exceptions import TemplateError as _JinjaTemplateError

if TYPE_CHECKING:
    from ..llama.llama_cpp import LlamaModel

logger = logging.getLogger(__name__)


def apply_template(
    model: "LlamaModel",
    messages: List[Dict[str, str]],
    template: Optional[str] = None,
    add_generation_prompt: bool = True,
) -> str:
    """Render ``messages`` against ``model``'s chat template.

    Args:
        model: Loaded ``LlamaModel`` (used for both the embedded
            template string and the BOS/EOS tokens that the Jinja env
            exposes via ``{{ bos_token }}`` / ``{{ eos_token }}``).
        messages: ``[{"role": ..., "content": ...}, ...]``.
        template: Optional override -- either a Jinja template string,
            a GGUF metadata key under which a named template is stored,
            or one of llama.cpp's built-in aliases (``chatml``,
            ``llama3``, ``gemma``, ...). When provided, the Jinja tier
            is skipped (named-template lookup is a C-API feature).
        add_generation_prompt: Append the assistant prefix that primes
            the model for a reply. Standard for inference; off for
            log-likelihood scoring.

    Returns:
        Formatted prompt string.
    """
    # Tier 1: vendored Jinja, only when no explicit template.
    if template is None:
        try:
            return _apply_jinja(model, messages, add_generation_prompt)
        except _JinjaTemplateError:
            # Intentional template raise (e.g. Gemma's
            # raise_exception). Fall through to the C-API tier.
            pass
        except (TypeError, KeyError, AttributeError) as exc:
            # Recoverable render-context issues: missing variable,
            # wrong type in messages, attribute lookup on None. Log so
            # silent miscompilations are observable rather than buried.
            # Bugs in our own code (NameError, ImportError, ...) keep
            # propagating.
            logger.warning(
                "Jinja chat-template rendering failed (%s); falling back to C-API path",
                type(exc).__name__,
            )

    # Tier 2: C-API substring-heuristic.
    if template:
        tmpl = model.get_default_chat_template_by_name(template) or template
    else:
        tmpl = model.get_default_chat_template()

    if tmpl:
        from ..llama.llama_cpp import LlamaChatMessage

        chat_messages = []
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise TypeError(f"Message at index {i} must be a dict, got {type(msg).__name__}")
            role = msg.get("role")
            if not role or not isinstance(role, str):
                raise ValueError(f"Message at index {i} missing or invalid 'role': {msg!r}")
            content = msg.get("content")
            if content is None:
                raise ValueError(f"Message at index {i} missing 'content' key")
            chat_messages.append(LlamaChatMessage(role=role, content=str(content)))
        return cast(str, model.chat_apply_template(tmpl, chat_messages, add_generation_prompt))

    # Tier 3: bare role/content layout.
    return format_messages_simple(messages)


def get_template(model: "LlamaModel", template_name: Optional[str] = None) -> str:
    """Look up the chat template string from a model.

    Returns the model's default template when ``template_name`` is None,
    otherwise the named template (or empty string if not present).
    """
    if template_name:
        return cast(str, model.get_default_chat_template_by_name(template_name))
    return cast(str, model.get_default_chat_template())


def format_messages_simple(messages: List[Dict[str, str]]) -> str:
    """Plain ``Role: content`` fallback when no chat template exists."""
    parts: List[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"{role.capitalize()}: {content}")
    return "\n\n".join(parts) + "\n\nAssistant:"


def _apply_jinja(
    model: "LlamaModel",
    messages: List[Dict[str, str]],
    add_generation_prompt: bool,
) -> str:
    """Render the model's embedded chat template via vendored jinja2.

    Mirrors ``transformers.PreTrainedTokenizerBase.apply_chat_template``:
    ``messages`` / ``bos_token`` / ``eos_token`` / ``add_generation_prompt``
    in the render context, plus the ``raise_exception`` / ``strftime_now``
    globals and the ``tojson`` filter. Any GGUF whose chat template was
    generated from a HuggingFace tokenizer config is compatible by
    construction.

    Raises:
        _JinjaTemplateError: explicit ``raise_exception`` from the
            template, or no embedded template at all.
    """
    from .._vendor.jinja2 import ext as _jinja2_ext
    from .._vendor.jinja2.sandbox import ImmutableSandboxedEnvironment

    template_str = model.get_default_chat_template()
    if not template_str:
        raise _JinjaTemplateError("Model has no embedded chat template")

    vocab = model.get_vocab()
    bos_id = vocab.token_bos()
    eos_id = vocab.token_eos()
    bos_token = vocab.token_to_piece(bos_id, special=True) if bos_id >= 0 else ""
    eos_token = vocab.token_to_piece(eos_id, special=True) if eos_id >= 0 else ""

    def raise_exception(message: str) -> str:
        raise _JinjaTemplateError(message)

    def tojson_filter(
        value: Any,
        ensure_ascii: bool = False,
        indent: Optional[int] = None,
        separators: Optional[Tuple[str, str]] = None,
        sort_keys: bool = False,
    ) -> str:
        return json.dumps(
            value,
            ensure_ascii=ensure_ascii,
            indent=indent,
            separators=separators,
            sort_keys=sort_keys,
        )

    def strftime_now(fmt: str) -> str:
        return datetime.now().strftime(fmt)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[_jinja2_ext.loopcontrols],
    )
    env.filters["tojson"] = tojson_filter
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now

    # Validate message shape inside the new code path so callers see
    # consistent error types regardless of which tier fired.
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise TypeError(f"Message at index {i} must be a dict, got {type(msg).__name__}")
        if not msg.get("role") or not isinstance(msg.get("role"), str):
            raise ValueError(f"Message at index {i} missing or invalid 'role': {msg!r}")
        if msg.get("content") is None:
            raise ValueError(f"Message at index {i} missing 'content' key")

    compiled = env.from_string(template_str)
    return cast(
        str,
        compiled.render(
            messages=messages,
            bos_token=bos_token,
            eos_token=eos_token,
            add_generation_prompt=add_generation_prompt,
        ),
    )
