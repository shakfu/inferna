"""
High-Level API for inferna

This module provides the primary user-facing API for inferna, including
both synchronous and asynchronous interfaces.

Example:
    >>> from inferna import complete, LLM
    >>>
    >>> # Simple completion
    >>> response = complete("What is 2+2?", model_path="models/llama.gguf")
    >>> print(response)
    >>>
    >>> # Streaming completion
    >>> for chunk in complete("Tell me a story", model_path="models/llama.gguf", stream=True):
    >>>     print(chunk, end="", flush=True)
    >>>
    >>> # Using the LLM class
    >>> llm = LLM("models/llama.gguf")
    >>> response = llm("What is Python?")

Async Example:
    >>> import asyncio
    >>> from inferna import complete_async, AsyncLLM
    >>>
    >>> async def main():
    >>>     # Simple async completion
    >>>     response = await complete_async("What is 2+2?", model_path="model.gguf")
    >>>     print(response)
    >>>
    >>>     # Using AsyncLLM class
    >>>     async with AsyncLLM("model.gguf") as llm:
    >>>         response = await llm("What is Python?")
    >>>         print(response)
    >>>
    >>>         # Async streaming
    >>>         async for chunk in llm.stream("Tell me a story"):
    >>>             print(chunk, end="", flush=True)
    >>>
    >>> asyncio.run(main())
"""

import asyncio
import codecs
import heapq
import math
import signal
import threading
from typing import (
    TYPE_CHECKING,
    AsyncIterator,
    Iterator,
    Optional,
    Dict,
    Any,
    List,
    Callable,
    Union,
    Tuple,
    cast,
)

if TYPE_CHECKING:
    from .agents.mcp import McpResource, McpTool
from dataclasses import dataclass, field
import logging
import time

logger = logging.getLogger(__name__)

from .defaults import (
    LLAMA_DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_MIN_P,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_N_GPU_LAYERS,
    DEFAULT_N_BATCH,
    DEFAULT_MAIN_GPU,
    DEFAULT_SPLIT_MODE,
)

from ._internal.chat_template import (
    apply_template as _apply_chat_template,
)
from ._internal.chat_template import (
    get_template as _get_chat_template,
)
from ._internal.function_calling import (
    CompiledToolResult,
    ToolCall,
    compile_tools,
)
from ._internal.mcp_facade import MCPFacade
from ._internal.response_cache import ResponseCache, ResponseCacheInfo, make_cache_key
from ._internal.structured import CompiledResponseFormat, compile_response_format
from .llama.llama_cpp import (
    LlamaAdapterLora,
    LlamaBatch,
    LlamaChatMessage,  # noqa: F401  -- re-exported for backwards compat
    LlamaModel,
    LlamaContext,
    LlamaModelParams,
    LlamaContextParams,
    LlamaSampler,
    LlamaSamplerChainParams,
    llama_batch_get_one,
    ggml_backend_load_all,
    disable_logging,
)


@dataclass
class GenerationConfig:
    """
    Configuration for text generation.

    Attributes:
        max_tokens: Maximum number of tokens to generate (see defaults.py)
        temperature: Sampling temperature, 0.0 = greedy (see defaults.py)
        top_k: Top-k sampling parameter (see defaults.py)
        top_p: Top-p (nucleus) sampling parameter (see defaults.py)
        min_p: Minimum probability threshold (see defaults.py)
        repeat_penalty: Penalty for repeating tokens (see defaults.py)
        n_gpu_layers: Number of layers to offload to GPU (see defaults.py)
        main_gpu: Primary GPU device index for inference (see defaults.py)
        split_mode: How to split model across GPUs (see defaults.py)
            0 = NONE: Use single GPU only (main_gpu)
            1 = LAYER: Split layers and KV cache across GPUs
            2 = ROW: Split with tensor parallelism (if supported)
        tensor_split: Proportion of work per GPU (default: None = auto)
            List of floats, one per GPU. Values are normalized by llama.cpp.
            Example: [1, 2] assigns 1/3 to GPU 0 and 2/3 to GPU 1.
        n_ctx: Context window size, None = auto (default: None)
        n_batch: Batch size for processing (see defaults.py)
        seed: Random seed for reproducibility (see defaults.py)
        stop_sequences: List of strings that stop generation (default: [])
        add_bos: Add beginning-of-sequence token (default: True)
        parse_special: Parse special tokens in prompt (default: True)

    Raises:
        ValueError: If any parameter is outside its valid range.

    Example:
        >>> # Use GPU 1 as primary device
        >>> config = GenerationConfig(main_gpu=1)
        >>>
        >>> # Multi-GPU with layer splitting
        >>> config = GenerationConfig(split_mode=1, n_gpu_layers=99)
        >>>
        >>> # Multi-GPU with tensor parallelism (row splitting)
        >>> config = GenerationConfig(split_mode=2, n_gpu_layers=99)
        >>>
        >>> # Custom tensor split: 30% GPU 0, 70% GPU 1
        >>> config = GenerationConfig(tensor_split=[0.3, 0.7])
    """

    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_k: int = DEFAULT_TOP_K
    top_p: float = DEFAULT_TOP_P
    min_p: float = DEFAULT_MIN_P
    repeat_penalty: float = DEFAULT_REPEAT_PENALTY
    n_gpu_layers: int = DEFAULT_N_GPU_LAYERS
    main_gpu: int = DEFAULT_MAIN_GPU
    split_mode: int = DEFAULT_SPLIT_MODE
    tensor_split: Optional[List[float]] = None
    n_ctx: Optional[int] = None
    n_batch: int = DEFAULT_N_BATCH
    seed: int = LLAMA_DEFAULT_SEED
    stop_sequences: List[str] = field(default_factory=list)
    add_bos: bool = True
    parse_special: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a dictionary, copying mutable values."""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
            "repeat_penalty": self.repeat_penalty,
            "n_gpu_layers": self.n_gpu_layers,
            "main_gpu": self.main_gpu,
            "split_mode": self.split_mode,
            "tensor_split": self.tensor_split.copy() if self.tensor_split else None,
            "n_ctx": self.n_ctx,
            "n_batch": self.n_batch,
            "seed": self.seed,
            "stop_sequences": self.stop_sequences.copy(),
            "add_bos": self.add_bos,
            "parse_special": self.parse_special,
        }

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        errors = []

        if self.max_tokens < 0:
            errors.append(f"max_tokens must be >= 0, got {self.max_tokens}")

        if self.temperature < 0.0:
            errors.append(f"temperature must be >= 0.0, got {self.temperature}")

        if self.top_k < 0:
            errors.append(f"top_k must be >= 0, got {self.top_k}")

        if not 0.0 <= self.top_p <= 1.0:
            errors.append(f"top_p must be between 0.0 and 1.0, got {self.top_p}")

        if not 0.0 <= self.min_p <= 1.0:
            errors.append(f"min_p must be between 0.0 and 1.0, got {self.min_p}")

        if self.repeat_penalty < 0.0:
            errors.append(f"repeat_penalty must be >= 0.0, got {self.repeat_penalty}")

        if self.n_gpu_layers < -1:
            errors.append(f"n_gpu_layers must be >= -1 (-1 = offload all), got {self.n_gpu_layers}")

        if self.main_gpu < 0:
            errors.append(f"main_gpu must be >= 0, got {self.main_gpu}")

        if self.split_mode not in (0, 1, 2):
            errors.append(f"split_mode must be 0, 1, or 2, got {self.split_mode}")

        if self.tensor_split is not None:
            if not isinstance(self.tensor_split, list):
                errors.append(f"tensor_split must be a list or None, got {type(self.tensor_split)}")
            elif any(not isinstance(v, (int, float)) or v < 0 for v in self.tensor_split):
                errors.append("tensor_split values must be non-negative numbers")

        if self.n_ctx is not None and self.n_ctx < 1:
            errors.append(f"n_ctx must be >= 1 or None, got {self.n_ctx}")

        if self.n_batch < 1:
            errors.append(f"n_batch must be >= 1, got {self.n_batch}")

        if self.seed < -1:
            errors.append(f"seed must be >= -1, got {self.seed}")

        if errors:
            raise ValueError("Invalid GenerationConfig: " + "; ".join(errors))


@dataclass
class GenerationStats:
    """Statistics from a generation run."""

    prompt_tokens: int
    generated_tokens: int
    total_time: float
    tokens_per_second: float
    prompt_time: float = 0.0
    generation_time: float = 0.0


@dataclass(frozen=True)
class TopLogprob:
    """One entry in a per-token ``top_logprobs`` list.

    Mirrors OpenAI's chat-completion logprobs schema (sans ``bytes``,
    which is encoder-specific and not currently surfaced).
    """

    token: str
    token_id: int
    logprob: float


@dataclass(frozen=True)
class TokenLogprob:
    """Logprob record for a single generated token.

    ``logprob`` is the log-softmax of the raw model logits at the
    sampling step -- i.e. ``log p(token | context)`` under the
    untransformed model distribution. Sampler-side transforms
    (temperature, top-k, top-p, grammar masking, penalties) reshape the
    *sampling* distribution but not the reported logprob, matching
    OpenAI's contract: a token sampled under heavy temperature scaling
    can land on a low-logprob outcome.
    """

    token: str
    token_id: int
    logprob: float
    # Optional top-K alternatives at this step, in descending logprob
    # order. Empty when the caller didn't request ``top_logprobs``.
    top_logprobs: List[TopLogprob] = field(default_factory=list)


@dataclass
class Response:
    """
    Response from text generation.

    This class wraps generated text with optional metadata and provides
    convenient conversion methods. It implements __str__ for backward
    compatibility, so it can be used anywhere a string is expected.

    Attributes:
        text: The generated text content
        stats: Optional generation statistics (tokens, timing, etc.)
        finish_reason: Why generation stopped ("stop", "length", "error")
        model: Model identifier/path used for generation

    Example:
        >>> response = complete("Hello", model_path="model.gguf")
        >>> print(response)  # Works like a string
        >>> print(response.text)  # Explicit text access
        >>> print(response.to_json())  # JSON output
        >>> data = response.to_dict()  # Dictionary for serialization
    """

    text: str
    stats: Optional[GenerationStats] = None
    finish_reason: str = "stop"
    model: str = ""
    # Populated when the call passed ``response_format=``: the validator
    # output (``json.loads(text)`` for dict schemas / ``json_object``,
    # a populated pydantic model instance for ``BaseModel`` schemas).
    # ``None`` for plain (unconstrained) generation.
    parsed: Optional[Any] = None
    # Populated when the call passed ``tools=``: the tool calls the
    # model produced (length 1 when ``tool_choice`` forced a call;
    # length 0 in ``auto`` mode if the model chose to answer in text).
    # ``None`` when ``tools=`` was not supplied.
    tool_calls: Optional[List[ToolCall]] = None
    # Populated when the call passed ``logprobs=True``: one entry per
    # generated token, in order. Each entry carries the sampled
    # token's raw-logit logprob plus (when ``top_logprobs > 0``) the
    # top-K alternative tokens at that step. ``None`` when logprobs
    # were not requested. Streaming generations leave this ``None``
    # because the caller assembles chunks externally.
    logprobs: Optional[List[TokenLogprob]] = None

    def __str__(self) -> str:
        """Return the text content. Enables backward-compatible string usage."""
        return self.text

    def __repr__(self) -> str:
        """Return a detailed representation."""
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Response(text={text_preview!r}, finish_reason={self.finish_reason!r})"

    def __eq__(self, other: object) -> bool:
        """Compare with strings or other Response objects."""
        if isinstance(other, str):
            return self.text == other
        if isinstance(other, Response):
            return self.text == other.text
        return NotImplemented

    def __hash__(self) -> int:
        """Hash based on text content."""
        return hash(self.text)

    def __len__(self) -> int:
        """Return length of text content."""
        return len(self.text)

    def __iter__(self) -> Iterator[str]:
        """Iterate over characters in text."""
        return iter(self.text)

    def __contains__(self, item: str) -> bool:
        """Check if substring is in text."""
        return item in self.text

    def __add__(self, other: object) -> str:
        """Concatenate with strings."""
        if isinstance(other, str):
            return self.text + other
        if isinstance(other, Response):
            return self.text + other.text
        return NotImplemented

    def __radd__(self, other: object) -> str:
        """Support string + Response."""
        if isinstance(other, str):
            return other + self.text
        return NotImplemented

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert response to dictionary.

        Returns:
            Dictionary containing all response data.

        Example:
            >>> response = complete("Hello", model_path="model.gguf")
            >>> data = response.to_dict()
            >>> print(data["text"])
        """
        result: Dict[str, Any] = {
            "text": self.text,
            "finish_reason": self.finish_reason,
            "model": self.model,
        }
        if self.stats is not None:
            result["stats"] = {
                "prompt_tokens": self.stats.prompt_tokens,
                "generated_tokens": self.stats.generated_tokens,
                "total_time": self.stats.total_time,
                "tokens_per_second": self.stats.tokens_per_second,
                "prompt_time": self.stats.prompt_time,
                "generation_time": self.stats.generation_time,
            }
        return result

    def to_json(self, indent: Optional[int] = None) -> str:
        """
        Convert response to JSON string.

        Args:
            indent: JSON indentation level (None for compact)

        Returns:
            JSON string representation.

        Example:
            >>> response = complete("Hello", model_path="model.gguf")
            >>> print(response.to_json(indent=2))
        """
        import json

        return json.dumps(self.to_dict(), indent=indent)


# ResponseCacheInfo and the cache implementation now live in
# ``inferna._internal.response_cache``. They are re-imported above so
# the public name ``ResponseCacheInfo`` remains importable from
# ``inferna.api`` for backwards compatibility.


class _SigintHandle:
    """Restorer returned by ``LLM.install_sigint_handler()``.

    Acts as a context manager (``__exit__`` restores the prior handler) and
    as an imperative handle (call ``.restore()`` directly). Idempotent --
    a second restore is a no-op.

    Standard last-installed-first-restored caveat applies: if multiple
    handlers are stacked, restore them in reverse order.
    """

    __slots__ = ("_previous", "_restored")

    def __init__(self, previous: object) -> None:
        self._previous = previous
        self._restored = False

    def restore(self) -> None:
        if self._restored:
            return
        try:
            signal.signal(signal.SIGINT, self._previous)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            # Off-main-thread or unrestorable handler; best-effort.
            pass
        self._restored = True

    def __enter__(self) -> "_SigintHandle":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.restore()


class LLM:
    """
    High-level LLM interface with model caching and convenient API.

    This class manages model lifecycle and provides simple methods for
    text generation with streaming support. It supports context reuse
    for improved performance when the context size doesn't change.

    Resource Management:
        The LLM class manages GPU memory and contexts. For proper cleanup:
        - Use as a context manager: `with LLM(...) as llm:`
        - Call `llm.close()` explicitly when done
        - Or let Python's garbage collector handle it via `__del__`

    Example:
        >>> # Simple usage with direct parameters
        >>> with LLM("models/llama.gguf", temperature=0.9, max_tokens=100) as llm:
        >>>     response = llm("What is Python?")
        >>>     print(response)
        >>>
        >>> # Streaming output
        >>> with LLM("models/llama.gguf") as llm:
        >>>     for chunk in llm("Tell me a joke", stream=True):
        >>>         print(chunk, end="")
        >>>
        >>> # With explicit GenerationConfig (for reuse or complex configs)
        >>> config = GenerationConfig(temperature=0.9, max_tokens=100)
        >>> with LLM("models/llama.gguf", config=config) as llm:
        >>>     response = llm("Hello!")
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[GenerationConfig] = None,
        verbose: bool = False,
        cache_size: int = 0,
        cache_ttl: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize generator with a model.

        Args:
            model_path: Path to GGUF model file
            config: Generation configuration (uses defaults if None)
            verbose: Print detailed information during generation
            cache_size: Maximum number of responses to cache (0 = disabled)
            cache_ttl: Cache time-to-live in seconds (None = no expiration)
            **kwargs: Generation parameters (temperature, max_tokens, etc.)
                      These override values in config if both are provided.

        Example:
            >>> # Direct parameters (recommended for simple cases)
            >>> llm = LLM("model.gguf", temperature=0.9, max_tokens=100)
            >>>
            >>> # Explicit config
            >>> config = GenerationConfig(temperature=0.9)
            >>> llm = LLM("model.gguf", config=config)
            >>>
            >>> # Config with overrides
            >>> llm = LLM("model.gguf", config=config, temperature=0.5)
            >>>
            >>> # With response caching
            >>> llm = LLM("model.gguf", cache_size=100, cache_ttl=3600)
        """
        self.model_path = model_path
        self.verbose = verbose
        self._closed = False
        self._last_stream_stats: Optional[GenerationStats] = None

        # Concurrent-use guard for the underlying llama.cpp context.
        # llama_context is not thread-safe and we release the GIL during
        # native generation calls, so two Python threads sharing one LLM
        # can race inside C++ code and corrupt KV cache, sampler, or
        # batch state. We protect every native-touching public method
        # with a non-blocking acquire of this lock: a second concurrent
        # caller fails fast with a clear RuntimeError instead of silently
        # corrupting state.
        #
        # This is intentionally a contention check, NOT a thread-id
        # check: legitimate sequential ownership transfer between threads
        # (e.g. asyncio.to_thread, ThreadPoolExecutor.submit) must keep
        # working, since there is no concurrent access in those patterns.
        # close()/__del__ are intentionally NOT guarded because the
        # garbage collector may run them on any thread.
        self._busy_lock = threading.Lock()

        # Cancellation signal for in-flight generations. ``cancel()`` sets
        # this; ``_generate_stream`` polls it between tokens and breaks out.
        # The same signal is mirrored to the underlying LlamaContext as a
        # C-level bint so ggml's abort callback can short-circuit a
        # long-running ``llama_decode`` (e.g. during prompt prefill on a
        # large context) without waiting for the next token boundary.
        self._cancel_event = threading.Event()

        # Initialize response cache
        self._cache: Optional[ResponseCache] = None
        if cache_size > 0:
            self._cache = ResponseCache(cache_size, cache_ttl)

        from .utils.validation import validate_gguf_file

        # Surface clear, typed errors (FileNotFoundError, IsADirectoryError,
        # PermissionError, ValueError) for the common bad-input cases before
        # llama.cpp gets a chance to fail with a NULL pointer or segfault
        # inside its GGUF parser. validate_gguf_file checks magic, version,
        # and tensor/kv counts.
        validate_gguf_file(model_path, kind="GGUF model")

        # Build config: start with provided config or defaults, then apply kwargs
        if config is None:
            if kwargs:
                self.config = GenerationConfig(**kwargs)
            else:
                self.config = GenerationConfig()
        else:
            if kwargs:
                # Create a copy of config with kwargs overrides
                config_dict = config.to_dict()
                config_dict.update(kwargs)
                self.config = GenerationConfig(**config_dict)
            else:
                self.config = config

        # Disable llama.cpp logging unless verbose mode is enabled
        if not verbose:
            disable_logging()

        # Load backends
        ggml_backend_load_all()

        # Initialize model
        model_params = LlamaModelParams()
        model_params.n_gpu_layers = self.config.n_gpu_layers
        model_params.main_gpu = self.config.main_gpu
        model_params.split_mode = self.config.split_mode
        if self.config.tensor_split is not None:
            model_params.tensor_split = self.config.tensor_split

        if self.verbose:
            print(f"Loading model: {model_path}")
            gpu_info = (
                f"GPU config: n_gpu_layers={self.config.n_gpu_layers}, "
                f"main_gpu={self.config.main_gpu}, split_mode={self.config.split_mode}"
            )
            if self.config.tensor_split:
                gpu_info += f", tensor_split={self.config.tensor_split}"
            print(gpu_info)

        self.model = LlamaModel(model_path, model_params)
        self.vocab = self.model.get_vocab()

        if self.verbose:
            print(f"Model loaded: {self.model.n_params} parameters")
            print(f"Vocabulary size: {self.vocab.n_vocab}")

        # Context will be created on-demand and cached when possible
        self._ctx: Optional[LlamaContext] = None
        self._ctx_size: int = 0  # Track current context size for reuse decisions
        self._sampler: Optional[LlamaSampler] = None

        # Token ids currently resident in seq 0 of the live context's KV
        # cache, in order. Maintained across calls so a follow-up prompt
        # that shares a prefix with the previous (prompt + generated)
        # turn can skip the redundant prefill: ``_generate_stream``
        # computes the longest common prefix, drops the divergent tail
        # via ``ctx.memory_seq_rm(0, overlap, -1)``, and prefills only
        # the suffix. Cleared on ``reset_context``, on context
        # recreation (size grew), on close, and whenever LoRA state
        # changes (a different adapter set produces different KV).
        self._kv_seq_tokens: List[int] = []

        # MCP client is created lazily on the first add_mcp_server() call so
        # callers who never use MCP pay no import or connection cost.
        self._mcp = MCPFacade()

        # LoRA adapters currently applied to the generation context. The list
        # is the source of truth across context recreations: ``_ensure_context``
        # re-applies it after constructing a fresh ``LlamaContext`` so a LoRA
        # loaded with ``load_lora`` survives ``reset_context()`` and the
        # automatic resize that fires when ``required_ctx > self._ctx_size``.
        # Each entry is ``(adapter_handle, scale)``; the model owns the
        # adapter's underlying memory (see LlamaModel.lora_adapter_init).
        self._loras: List[tuple[LlamaAdapterLora, float]] = []

        # Embedding context (separate from the generation context because
        # llama.cpp's embeddings flag is set at context creation and a
        # generation context's KV state would interfere with embedding output).
        # Lazily constructed on the first ``embed()`` call and reused for
        # subsequent calls.
        self._embed_ctx: Optional[LlamaContext] = None
        self._embed_n_ctx: int = 0

    def __enter__(self) -> "LLM":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Context manager exit - ensures cleanup."""
        self.close()

    def __del__(self) -> None:
        """Destructor - cleanup resources if not already done."""
        if not getattr(self, "_closed", True):
            try:
                self.close()
            except Exception:
                pass

    def close(self) -> None:
        """
        Explicitly release resources (context, sampler).

        This method frees the context and sampler to release GPU memory.
        The model remains loaded for potential reuse. Call this when you're
        done with generation or want to free memory.

        After calling close(), the LLM instance can still be used - new
        contexts will be created as needed.
        """
        if getattr(self, "_closed", True):
            return

        if getattr(self, "verbose", False):
            print("Closing LLM resources")

        # Release context and sampler (use getattr for safety in __del__)
        if getattr(self, "_ctx", None) is not None:
            self._ctx = None
            self._ctx_size = 0
        # Drop the prompt-cache shadow with the context that owned the KV.
        if getattr(self, "_kv_seq_tokens", None):
            self._kv_seq_tokens = []

        if getattr(self, "_sampler", None) is not None:
            self._sampler = None

        # Drop the embedding context (constructed lazily by embed()). The
        # underlying llama_context owns no shared state with the generation
        # context, so dropping it just frees the embedding-mode KV cache.
        if getattr(self, "_embed_ctx", None) is not None:
            self._embed_ctx = None
            self._embed_n_ctx = 0

        # Drop adapter handles. The model owns each adapter's memory, so
        # dropping our references just clears the apply-list; the next
        # generation context to come up will start with no adapters
        # attached, matching the freshly-loaded-LLM contract.
        if getattr(self, "_loras", None):
            self._loras = []

        # Disconnect any MCP servers the caller attached. The facade
        # handles the best-effort teardown (errors are logged, not
        # raised) so a flaky remote server can't block local cleanup.
        mcp = getattr(self, "_mcp", None)
        if mcp is not None:
            mcp.close()

        self._closed = True

    def cancel(self) -> None:
        """
        Request cancellation of an in-flight generation.

        Safe to call from any thread. Has effect at two layers:

        - Between tokens: ``_generate_stream`` polls a ``threading.Event``
          each iteration and exits cleanly when set. This stops generation
          within ~1 token of latency (typically a few milliseconds).
        - Mid-decode: the underlying llama_context has a nogil
          ggml_abort_callback installed that reads a C-level flag. Setting
          it causes ``llama_decode`` to abort the current batch -- this
          matters when prefilling a long prompt, where one ``decode`` call
          can take seconds before the next token-boundary check fires.

        The flag is automatically cleared at the start of the next
        generation, so it is safe to call ``cancel()`` even when no
        generation is currently running -- the request will not "carry
        over" to the next call.

        No-op if no context exists yet (cancellation only matters during
        generation, and generation always creates a context first).
        """
        self._cancel_event.set()
        ctx = getattr(self, "_ctx", None)
        if ctx is not None:
            ctx.cancel = True

    @property
    def cancel_requested(self) -> bool:
        """Whether ``cancel()`` has been called and not yet cleared."""
        return self._cancel_event.is_set()

    def install_sigint_handler(self) -> _SigintHandle:
        """
        Install a SIGINT (Ctrl-C) handler that calls ``self.cancel()``.

        Useful for CLI scripts: Ctrl-C will interrupt generation cleanly
        — including during a long ``llama_decode`` (prompt prefill) where
        the default ``KeyboardInterrupt`` mechanism would otherwise be
        delayed until the C call returns. The generator exits via the
        normal cancellation path and returns whatever was produced so
        far; no exception propagates from the handler itself.

        Must be called from the main Python thread (``signal.signal``
        restriction). The previous SIGINT handler is saved and can be
        restored by calling ``.restore()`` on the returned object, or by
        using it as a context manager.

        Note: this library normally avoids touching signal handlers --
        installing one is opt-in for callers who explicitly want this
        behavior. Multiple installations stack; restore them in reverse
        order (last-installed-first-restored).

        Example (context manager)::

            with llm.install_sigint_handler():
                for chunk in llm("Long prompt", stream=True):
                    print(chunk, end="", flush=True)

        Example (imperative)::

            handle = llm.install_sigint_handler()
            try:
                response = llm("Prompt")
            finally:
                handle.restore()

        Returns:
            A ``_SigintHandle`` that restores the previous handler on
            ``__exit__`` or ``.restore()``.
        """
        previous = signal.signal(signal.SIGINT, lambda *_: self.cancel())
        return _SigintHandle(previous)

    def _try_acquire_busy(self) -> None:
        """Acquire the busy-lock or raise on contention.

        Non-blocking: if another thread is currently inside a guarded
        method, we raise immediately rather than serialize behind it.
        Serializing would hide the bug; raising lets the caller see that
        their concurrent-use pattern is unsafe and fix it.
        """
        if not self._busy_lock.acquire(blocking=False):
            raise RuntimeError(
                "LLM is currently being used by another thread. llama.cpp "
                "contexts are not thread-safe — create one LLM per thread "
                "instead of sharing a single instance across threads."
            )

    def _stream_with_busy_release(self, gen: Iterator[str]) -> Iterator[str]:
        """Wrap a streaming generator so the busy lock is released when
        the stream is exhausted, closed, or garbage collected.

        Generator ``finally`` blocks run on every termination path
        (StopIteration, exception, .close(), gc), so the lock is
        guaranteed to be released even if the caller drops the iterator
        without consuming it.
        """
        try:
            yield from gen
        finally:
            try:
                self._busy_lock.release()
            except RuntimeError:
                # Lock already released — defensive, should not happen.
                pass

    def reset_context(self) -> None:
        """
        Force recreation of context on next generation.

        This clears the KV cache and ensures a fresh context is created.
        Useful when you want to start a completely new conversation without
        any prior context.
        """
        self._try_acquire_busy()
        try:
            if self._ctx is not None:
                if self.verbose:
                    print("Resetting context")
                self._ctx = None
                self._ctx_size = 0
            # Drop the prompt-cache shadow: the next generation should
            # see no prefix overlap and prefill from scratch.
            self._kv_seq_tokens = []
        finally:
            self._busy_lock.release()

    # ------------------------------------------------------------------
    # LoRA adapter management
    # ------------------------------------------------------------------

    def load_lora(self, path: str, scale: float = 1.0) -> LlamaAdapterLora:
        """Load a LoRA adapter and apply it to the generation context.

        The adapter is loaded against ``self.model`` (so it must have been
        trained on a compatible base) and then applied to the live context
        with the given ``scale``. If the context has not been created yet,
        the adapter is recorded and applied automatically when the first
        generation triggers ``_ensure_context``.

        Args:
            path: Filesystem path to a GGUF LoRA adapter.
            scale: Adapter strength. ``1.0`` applies the adapter at full
                weight, ``0.0`` disables it (useful for keeping it loaded
                but inactive), negative values subtract.

        Returns:
            The ``LlamaAdapterLora`` handle. Pass it back to
            :meth:`unload_lora` to remove this specific adapter, or call
            :meth:`clear_loras` to remove all of them.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the adapter fails to load (e.g. wrong base
                model, malformed file).
        """
        adapter = self.model.lora_adapter_init(path)
        self._loras.append((adapter, scale))
        if self._ctx is not None:
            self._apply_loras_to(self._ctx)
        if self.verbose:
            print(f"Loaded LoRA: {path} (scale={scale})")
        return adapter

    def unload_lora(self, adapter: LlamaAdapterLora) -> None:
        """Remove a previously loaded LoRA adapter.

        The adapter is dropped from the apply-list and the change is
        pushed to the live context (if any). The adapter handle remains
        valid -- the model still owns it -- so callers can re-apply it
        later via ``load_lora`` (which will re-load from disk; for now
        the handle itself is not re-attachable without a fresh load).

        Args:
            adapter: A handle previously returned from :meth:`load_lora`.

        Raises:
            ValueError: If ``adapter`` is not currently applied.
        """
        before = len(self._loras)
        self._loras = [(a, s) for (a, s) in self._loras if a is not adapter]
        if len(self._loras) == before:
            raise ValueError("unload_lora: adapter is not currently applied")
        if self._ctx is not None:
            self._apply_loras_to(self._ctx)

    def clear_loras(self) -> None:
        """Remove all LoRA adapters from the generation context."""
        if not self._loras and self._ctx is None:
            return
        self._loras = []
        if self._ctx is not None:
            self._apply_loras_to(self._ctx)

    def list_loras(self) -> List[tuple[LlamaAdapterLora, float]]:
        """Return a copy of the (adapter, scale) list currently applied."""
        return list(self._loras)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(
        self,
        text: Union[str, List[str]],
        pooling: str = "mean",
        normalize: bool = True,
    ) -> Union[List[float], List[List[float]]]:
        """Compute an embedding for one or more strings.

        Constructs a sibling embedding context (separate from the
        generation context, since llama.cpp's embeddings flag is set at
        context creation and a generation context's KV state would
        interfere with embedding output). The embedding context is
        cached on the instance and reused across calls.

        Pooling is performed in Python over the per-token hidden states
        returned by ``llama_get_embeddings``. This matches the strategy
        in :class:`inferna.rag.Embedder` and avoids relying on
        llama.cpp's internal pooling, which has historically been
        unreliable for some generative-model architectures used in
        embedding mode.

        Args:
            text: A single string or a list of strings.
            pooling: One of ``"mean"``, ``"cls"``, ``"last"``. ``"mean"``
                averages over all tokens (the standard choice for
                BERT-style embedding models like BGE/Snowflake);
                ``"cls"`` uses the first token's hidden state;
                ``"last"`` uses the last token's hidden state (often
                preferred for decoder-only models).
            normalize: If ``True`` (default), L2-normalize each output
                vector. Cosine similarity collapses to a dot product
                under L2-normal vectors, which is what most vector
                stores expect.

        Returns:
            A single ``list[float]`` if ``text`` is a string, or a
            ``list[list[float]]`` if ``text`` is a list. The dimension
            matches ``self.model.n_embd``.

        Raises:
            ValueError: If ``pooling`` is not one of the supported
                strategies.
        """
        if isinstance(text, str):
            single = True
            texts: List[str] = [text]
        else:
            single = False
            texts = list(text)
        if not texts:
            return []

        if pooling not in ("mean", "cls", "last"):
            raise ValueError(f"Invalid pooling type: {pooling!r}. Must be one of: 'mean', 'cls', 'last'")

        # Tokenise upfront so we can size the embedding context to the
        # longest input. ``add_special=True`` matches Embedder; embedding
        # models (BGE etc.) expect the [CLS] / [BOS] token.
        tokenised = [self.vocab.tokenize(t, add_special=True, parse_special=False) for t in texts]
        max_tokens = max((len(toks) for toks in tokenised), default=1)
        # Cap at the model's training context so we don't ask for more
        # than the model can encode.
        n_ctx_train = self.model.n_ctx_train
        if max_tokens > n_ctx_train:
            for i, toks in enumerate(tokenised):
                if len(toks) > n_ctx_train:
                    tokenised[i] = toks[:n_ctx_train]
            max_tokens = n_ctx_train

        ctx = self._ensure_embed_context(max_tokens)
        n_embd = self.model.n_embd

        results: List[List[float]] = []
        for tokens in tokenised:
            if not tokens:
                # Empty input → empty embedding. Skip the decode rather
                # than feeding a zero-token batch (which llama.cpp rejects).
                results.append([0.0] * n_embd)
                continue
            n_tokens = len(tokens)
            ctx.kv_cache_clear()

            # Mark every token for output so per-token hidden states are
            # available for pooling. ``embd=0`` because we're feeding token
            # ids, not pre-computed embeddings.
            batch = LlamaBatch(n_tokens=n_tokens, embd=0, n_seq_max=1)
            for pos, tok in enumerate(tokens):
                batch.add(tok, pos, [0], True)
            ctx.decode(batch)

            raw = ctx.get_embeddings()
            # ``get_embeddings`` returns a flat list sized
            # ``n_tokens * n_embd``; defensively recompute in case the
            # tokeniser stripped tokens.
            actual_tokens = len(raw) // n_embd if n_embd else 0
            if actual_tokens == 0:
                results.append([0.0] * n_embd)
                continue

            if pooling == "mean":
                pooled = [0.0] * n_embd
                for t in range(actual_tokens):
                    base = t * n_embd
                    for i in range(n_embd):
                        pooled[i] += raw[base + i]
                inv = 1.0 / actual_tokens
                pooled = [v * inv for v in pooled]
            elif pooling == "cls":
                pooled = list(raw[:n_embd])
            else:  # "last"
                base = (actual_tokens - 1) * n_embd
                pooled = list(raw[base : base + n_embd])

            if normalize:
                norm_sq = sum(v * v for v in pooled)
                if norm_sq > 0.0:
                    inv = norm_sq**-0.5
                    pooled = [v * inv for v in pooled]
            results.append(pooled)

        return results[0] if single else results

    def _ensure_embed_context(self, required_tokens: int) -> LlamaContext:
        """Lazily construct (or resize) the embedding context."""
        if self._embed_ctx is not None and self._embed_n_ctx >= required_tokens:
            return self._embed_ctx

        if self.verbose and self._embed_ctx is not None:
            print(f"Resizing embed context: {self._embed_n_ctx} -> {required_tokens}")
        elif self.verbose:
            print(f"Creating embed context: {required_tokens} tokens")

        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = max(required_tokens, 32)
        ctx_params.n_batch = max(required_tokens, 32)
        # Pooling is done in Python; ask llama.cpp for raw per-token states.
        ctx_params.pooling_type = 0  # NONE
        ctx_params.no_perf = not self.verbose

        ctx = LlamaContext(self.model, ctx_params)
        ctx.set_embeddings_mode(True)
        self._embed_ctx = ctx
        self._embed_n_ctx = ctx_params.n_ctx
        return ctx

    @property
    def cache_enabled(self) -> bool:
        """Return True if response caching is enabled."""
        return self._cache is not None

    def cache_info(self) -> Optional[ResponseCacheInfo]:
        """
        Return cache statistics.

        Returns:
            ResponseCacheInfo with hits, misses, maxsize, currsize, ttl.
            Returns None if caching is disabled.
        """
        if self._cache is None:
            return None
        return self._cache.info()

    def cache_clear(self) -> None:
        """
        Clear all cached responses and reset cache statistics.

        Does nothing if caching is disabled.
        """
        if self._cache is not None:
            self._cache.clear()

    def _make_cache_key(self, prompt: str, config: GenerationConfig) -> Optional[str]:
        """Thin wrapper around :func:`make_cache_key`.

        Kept as a method so subclasses / tests can monkeypatch it; the
        actual key derivation lives in ``_internal/response_cache.py``.
        """
        return make_cache_key(prompt, config, random_seed_sentinel=LLAMA_DEFAULT_SEED)

    def _ensure_context(self, prompt_length: int, config: GenerationConfig) -> LlamaContext:
        """
        Create or recreate context if needed.

        Context is reused when possible to avoid allocation overhead.
        A new context is created when:
        - No context exists
        - The required size exceeds current context size
        - The instance was closed (will reopen)

        KV cache is **not** cleared here. The prompt-cache layer in
        ``_generate_stream`` manages the seq-0 KV state explicitly:
        when the new prompt shares a prefix with the previous turn it
        rolls the cache back to the divergence point via
        ``memory_seq_rm`` and prefills only the suffix; otherwise it
        wipes seq 0 and prefills from scratch.

        Returns the live context so callers don't have to re-narrow
        ``self._ctx`` from ``Optional`` after this method runs.
        """
        # Reopen if closed
        if self._closed:
            self._closed = False

        # Calculate required context size
        if config.n_ctx is None:
            required_ctx = prompt_length + config.max_tokens
        else:
            required_ctx = config.n_ctx

        # Check if we can reuse existing context
        if self._ctx is not None and self._ctx_size >= required_ctx:
            if self.verbose:
                print(f"Reusing context (size {self._ctx_size}, need {required_ctx})")
            return self._ctx

        # Need to create new context (either none exists or too small).
        # The fresh context starts with empty KV, so the prompt-cache
        # shadow must be cleared too -- otherwise the next generation
        # would think a prefix is cached when it isn't.
        if self.verbose:
            if self._ctx is not None:
                print(f"Recreating context: {self._ctx_size} -> {required_ctx} tokens")
            else:
                print(f"Creating context: {required_ctx} tokens")
        self._kv_seq_tokens = []

        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = required_ctx
        ctx_params.n_batch = config.n_batch
        ctx_params.no_perf = not self.verbose

        # Note: Seed is set in sampler, not context
        ctx = LlamaContext(self.model, ctx_params)
        # Wire mid-decode cancellation: a nogil ggml_abort_callback that
        # reads a C bint owned by the context. ``cancel()`` flips that
        # bint and the next ggml op poll aborts the in-progress batch.
        ctx.install_cancel_callback()
        self._ctx = ctx
        self._ctx_size = required_ctx
        # Re-apply any LoRA adapters previously attached via load_lora().
        # set_adapters_lora is a no-op when the list is empty, so this is
        # safe to call unconditionally on every context (re)creation.
        self._apply_loras_to(ctx)
        return ctx

    def _apply_loras_to(self, ctx: LlamaContext) -> None:
        """Apply ``self._loras`` to a context.

        Centralised so context creation, ``load_lora``, ``unload_lora``,
        and ``clear_loras`` all route the same call into llama.cpp. Passing
        an empty list clears any previously applied set on the context.
        Also invalidates the prompt-cache shadow (``self._kv_seq_tokens``)
        because changing the active adapter set produces a different
        attention model -- the previously cached KV no longer reflects
        the model the next generation will see.
        """
        adapters = [a for a, _ in self._loras]
        scales = [s for _, s in self._loras]
        ctx.set_adapters_lora(adapters, scales)
        self._kv_seq_tokens = []

    def _ensure_sampler(
        self,
        config: GenerationConfig,
        grammar: Optional[str] = None,
        grammar_root: str = "root",
    ) -> LlamaSampler:
        """Create or recreate sampler if needed.

        Returns the live sampler so callers don't have to re-narrow
        ``self._sampler`` from ``Optional`` after this method runs.

        Args:
            config: GenerationConfig driving the sampler chain.
            grammar: Optional GBNF grammar string. When supplied, a
                grammar sampler is added FIRST in the chain so it
                filters the logit set before any other sampler runs;
                this is the order ``llama_sampler_chain`` expects for
                grammar to be effective.
            grammar_root: Root rule name in ``grammar``.
        """
        # Always create fresh sampler to respect new config
        sampler_params = LlamaSamplerChainParams()
        sampler_params.no_perf = not self.verbose

        sampler = LlamaSampler(sampler_params)

        # Grammar must come first: it filters the logit distribution
        # down to grammar-valid tokens, then temperature/top-k/top-p
        # operate over that filtered set.
        if grammar is not None:
            sampler.add_grammar(self.vocab, grammar, grammar_root)

        # Add sampling methods based on config
        if config.temperature == 0.0:
            # Greedy sampling
            sampler.add_greedy()
        else:
            # Probabilistic sampling
            sampler.add_min_p(config.min_p, 1)
            sampler.add_top_k(config.top_k)
            sampler.add_top_p(config.top_p, 1)
            sampler.add_temp(config.temperature)

            # Distribution sampler
            if config.seed != LLAMA_DEFAULT_SEED:
                sampler.add_dist(config.seed)
            else:
                sampler.add_dist(LLAMA_DEFAULT_SEED)

        self._sampler = sampler
        return sampler

    def __call__(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
        response_format: Optional[Any] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Union[str, Dict[str, Any]] = "auto",
        logprobs: bool = False,
        top_logprobs: int = 0,
    ) -> Union[Response, Iterator[str]]:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            config: Generation configuration (uses instance config if None)
            stream: If True, return iterator of text chunks
            on_token: Optional callback called for each generated token
            response_format: Optional structured-output specifier. Accepts
                ``{"type": "json_object"}`` (any well-formed JSON),
                ``{"type": "json_schema", "schema": <dict | BaseModel>}``,
                or a pydantic ``BaseModel`` subclass directly. When
                supplied, generation is grammar-constrained and the
                resulting text is parsed onto ``Response.parsed``.
            tools: Optional OpenAI-shaped tool list (``[{"type": "function",
                "function": {"name": ..., "parameters": ...}}, ...]``).
                When supplied, generation is grammar-constrained to a
                tool-call envelope and ``Response.tool_calls`` is
                populated. Mutually exclusive with ``response_format``.
            tool_choice: ``"required"`` (must call a tool, default when
                ``tools`` is set unless ``"auto"`` is requested), ``"auto"``
                (model picks tool vs. plain content), ``"none"`` (forbid;
                callers should just omit ``tools``), or
                ``{"type": "function", "function": {"name": "x"}}``
                (constrain to one specific tool).
            logprobs: If ``True``, attach a ``Response.logprobs`` list
                with one ``TokenLogprob`` per generated token. The
                logprob is the log-softmax of the raw model logits at
                each step (independent of temperature / top-k / top-p
                / grammar transforms applied by the sampler).
            top_logprobs: When ``> 0``, also attach the top-K
                alternative tokens at each step. Implies
                ``logprobs=True``. Capped at the model vocab size.

        Returns:
            Response object (if stream=False) or iterator of text chunks (if stream=True).
            The Response object can be used as a string due to __str__ implementation.
        """
        if tools is not None and response_format is not None:
            raise ValueError("tools= and response_format= are mutually exclusive")
        if tools is not None and stream:
            # Tool-call streaming would need to assemble JSON chunks
            # before validation; the OpenAI streaming protocol handles
            # this by emitting incremental ``arguments`` deltas. Keep
            # the surface narrow for now.
            raise NotImplementedError("stream=True with tools= is not supported yet")
        if top_logprobs < 0:
            raise ValueError(f"top_logprobs must be >= 0, got {top_logprobs}")
        if top_logprobs > 0:
            logprobs = True
        if logprobs and stream:
            # Streaming logprobs would land per-chunk in the iterator;
            # callers that want them today should use the non-streaming
            # path. We can wire this through _generate_stream later if
            # there's demand.
            raise NotImplementedError("stream=True with logprobs=True is not supported yet")

        self._try_acquire_busy()
        if stream:
            # The wrapper releases the lock when the generator is
            # exhausted, closed, or garbage collected. Returning the
            # wrapper unconditionally (no try/finally here) hands lock
            # ownership to the generator.
            return self._stream_with_busy_release(
                self._generate_stream(prompt, config, on_token, response_format=response_format)
            )
        try:
            return self._generate(
                prompt,
                config,
                on_token,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
            )
        finally:
            self._busy_lock.release()

    def _generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        on_token: Optional[Callable[[str], None]] = None,
        response_format: Optional[Any] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Union[str, Dict[str, Any]] = "auto",
        logprobs: bool = False,
        top_logprobs: int = 0,
    ) -> Response:
        """Non-streaming generation returning Response object."""
        config = config or self.config

        # Compile the structured-output / tool-call spec up front so a
        # malformed schema raises before we touch the GPU. ``tools=`` and
        # ``response_format=`` share the same compiled-grammar surface
        # so only one pipeline lights up at a time (mutual exclusion is
        # enforced by ``__call__``).
        compiled: Optional[CompiledResponseFormat]
        if tools is not None:
            compiled = compile_tools(tools, tool_choice)
            using_tools = True
        else:
            compiled = compile_response_format(response_format)
            using_tools = False

        # Check cache first (only when no on_token callback and no
        # response_format / tools / logprobs -- structured outputs bypass
        # caching since the cache key would have to include the schema
        # and pydantic validators don't reliably round-trip through
        # pickle; logprobs bypass caching because the cached Response
        # would have stale per-token records that don't reflect the
        # caller's current top_logprobs setting).
        cache_key: Optional[str] = None
        cache = self._cache
        if cache is not None and on_token is None and compiled is None and not logprobs:
            cache_key = self._make_cache_key(prompt, config)
            if cache_key is not None:
                cached = cache.get(cache_key)
                if cached is not None:
                    if self.verbose:
                        print("Cache hit")
                    return cast(Response, cached)

        start_time = time.time()

        # Tokenize for stats
        prompt_tokens = self.vocab.tokenize(prompt, add_special=config.add_bos, parse_special=config.parse_special)
        n_prompt = len(prompt_tokens)

        # Per-token logprob sink. ``_generate_stream`` appends one
        # ``TokenLogprob`` per generated token when this list is non-None.
        logprobs_sink: Optional[List[TokenLogprob]] = [] if logprobs else None

        # Generate text
        chunks = list(
            self._generate_stream(
                prompt,
                config,
                on_token,
                _compiled_format=compiled,
                _logprobs_sink=logprobs_sink,
                _top_logprobs=top_logprobs,
            )
        )
        text = "".join(chunks)

        end_time = time.time()
        total_time = end_time - start_time

        # Calculate stats
        response_tokens = self.vocab.tokenize(text, add_special=False, parse_special=False)
        n_generated = len(response_tokens)

        stats = GenerationStats(
            prompt_tokens=n_prompt,
            generated_tokens=n_generated,
            total_time=total_time,
            tokens_per_second=n_generated / total_time if total_time > 0 else 0.0,
        )

        # Route the validator's output. For ``tools=`` the result is a
        # CompiledToolResult which fans out across both Response.text
        # (auto-mode 'content' branch) and Response.tool_calls; for
        # ``response_format=`` it's a plain dict / pydantic instance
        # that lands on Response.parsed.
        parsed: Optional[Any] = None
        tool_calls: Optional[List[ToolCall]] = None
        final_text = text
        if compiled is not None:
            try:
                validated = compiled.validator(text)
            except ValueError:
                # Grammar guarantees structural validity, but the model
                # can still hit max_tokens mid-structure. Surface that
                # via Response.parsed=None / tool_calls=None rather
                # than raising; caller decides whether to retry.
                validated = None
            if using_tools:
                if isinstance(validated, CompiledToolResult):
                    tool_calls = validated.tool_calls
                    if validated.content is not None:
                        final_text = validated.content
                else:
                    tool_calls = []
            else:
                parsed = validated

        response = Response(
            text=final_text,
            stats=stats,
            finish_reason="stop",
            model=self.model_path,
            parsed=parsed,
            tool_calls=tool_calls,
            logprobs=logprobs_sink,
        )

        # Store in cache if enabled
        if cache_key is not None and cache is not None:
            cache.put(cache_key, response)

        return response

    def _generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        on_token: Optional[Callable[[str], None]] = None,
        response_format: Optional[Any] = None,
        _compiled_format: Optional[CompiledResponseFormat] = None,
        _logprobs_sink: Optional[List[TokenLogprob]] = None,
        _top_logprobs: int = 0,
    ) -> Iterator[str]:
        """
        Internal streaming generation implementation.

        Yields:
            Text chunks as they are generated
        """
        # Use provided config or fall back to instance config
        config = config or self.config

        # Reset cancellation state at the start of each generation. A stale
        # ``cancel()`` from before this call does not carry over.
        self._cancel_event.clear()

        # Resolve structured-output grammar. ``_generate`` pre-compiles
        # and threads it via ``_compiled_format`` to avoid double work;
        # streaming callers (``__call__(stream=True)``) come in through
        # ``response_format`` and we compile here.
        compiled = _compiled_format
        if compiled is None and response_format is not None:
            compiled = compile_response_format(response_format)
        grammar = compiled.grammar if compiled is not None else None
        grammar_root = compiled.grammar_root if compiled is not None else "root"

        start_time = time.time()

        # Tokenize prompt
        prompt_tokens = self.vocab.tokenize(prompt, add_special=config.add_bos, parse_special=config.parse_special)
        n_prompt = len(prompt_tokens)

        if self.verbose:
            print(f"Prompt tokens: {n_prompt}")

        # Ensure context and sampler are ready
        # Always recreate sampler to ensure fresh state
        ctx = self._ensure_context(n_prompt, config)
        sampler = self._ensure_sampler(config, grammar=grammar, grammar_root=grammar_root)
        # Mirror the Python-side cancel flag onto the C-level flag for the
        # active context. Cleared here; ``cancel()`` sets both layers.
        ctx.cancel = False

        # ----- Prompt-cache: prefix-share with the previous turn -------
        #
        # Compute the longest token-id prefix the new prompt shares with
        # whatever currently lives in seq 0's KV. Drop the divergent
        # tail and prefill only the suffix. This makes multi-turn chat
        # essentially free on the prompt side once the conversation
        # exceeds a few turns: a 4k-token system prompt only gets
        # decoded once.
        #
        # Always reserve at least the last prompt token for fresh
        # decode -- ``sample(ctx, -1)`` reads logits from the most
        # recent decode, so we need at least one fresh token to give
        # the sampler valid logits to consume.
        cached = self._kv_seq_tokens
        max_overlap = min(len(cached), n_prompt)
        overlap = 0
        for k in range(max_overlap):
            if cached[k] != prompt_tokens[k]:
                break
            overlap = k + 1
        # Reserve the last prompt token so the sampler has fresh logits.
        # Guarded against an empty prompt (n_prompt == 0): the
        # overlap-vs-n_prompt comparison would otherwise yield -1 and
        # corrupt the prefill loop's start position.
        if n_prompt > 0 and overlap >= n_prompt:
            overlap = n_prompt - 1

        if overlap > 0:
            # Trim seq 0 back to position ``overlap``. memory_seq_rm with
            # p1=-1 means "drop everything from p0 to the end".
            ctx.memory_seq_rm(0, overlap, -1)
            if self.verbose:
                print(f"Prompt cache hit: reusing {overlap}/{n_prompt} prompt tokens")
        else:
            # No prefix overlap: clear seq 0 and fall through to full
            # prefill. memory_seq_rm with p0=0 / p1=-1 drops everything
            # in this sequence; other sequences (e.g. embedding) live
            # in different contexts so they're untouched.
            ctx.memory_seq_rm(0, 0, -1)

        # Process the prompt suffix in batches to avoid exceeding n_batch.
        # If the caller has already issued ``cancel()`` (or does so during
        # a long prefill), the abort callback aborts the in-progress
        # decode and we bail before generation begins.
        n_batch = config.n_batch
        for i in range(overlap, n_prompt, n_batch):
            if self._cancel_event.is_set():
                # Cancellation mid-prefill leaves the KV in a partial
                # state. Drop the shadow so the next call rebuilds from
                # scratch rather than trusting torn state.
                self._kv_seq_tokens = []
                return
            batch_tokens = prompt_tokens[i : i + n_batch]
            batch = llama_batch_get_one(batch_tokens, i)  # absolute position
            ctx.decode(batch)

        # Generate tokens
        n_pos = n_prompt
        n_generated = 0
        # Track sampled token ids so the prompt-cache shadow can be
        # updated at the end of the loop. The KV at completion holds
        # ``prompt_tokens + sampled_ids`` in seq 0; recording the same
        # list lets the next call's prefix match find this turn's full
        # output (system prompt + user + assistant) as a candidate
        # prefix.
        sampled_ids: List[int] = []

        # Stop sequence handling: buffer recent output to detect sequences spanning tokens
        # We only need to buffer enough to detect the longest stop sequence
        stop_buffer = ""
        max_stop_len = max(len(s) for s in config.stop_sequences) if config.stop_sequences else 0

        # Byte-level BPE tokenizers routinely emit pieces that contain
        # partial UTF-8 codepoints (e.g. one lead byte, then continuation
        # bytes in subsequent tokens). An incremental decoder buffers the
        # tail until a complete codepoint lands and emits "" otherwise --
        # this preserves bytes that would be corrupted to U+FFFD by a
        # per-piece replace decode, and keeps stop-sequence matching
        # operating on the true decoded text.
        utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        for _ in range(config.max_tokens):
            # Cooperative cancellation check between tokens. The mid-decode
            # ggml abort callback handles cancellation during long
            # ``llama_decode`` calls; this handles the steady-state
            # token-by-token loop with sub-millisecond latency.
            if self._cancel_event.is_set():
                break

            # Sample next token
            new_token_id = sampler.sample(ctx, -1)

            # Capture per-token logprob before EOG / piece decoding so
            # we record one entry per generated token in the same order
            # the caller observes them. Skipped for EOG (matches
            # OpenAI's contract: the stop token does not appear in the
            # logprobs array).
            if _logprobs_sink is not None and not self.vocab.is_eog(new_token_id):
                _logprobs_sink.append(self._build_token_logprob(ctx, new_token_id, _top_logprobs))

            # Check for end of generation
            if self.vocab.is_eog(new_token_id):
                break

            # Decode token to text via the incremental UTF-8 decoder.
            piece_bytes = self.vocab.token_to_piece_bytes(new_token_id, special=True)
            piece = utf8_decoder.decode(piece_bytes)
            if not piece:
                # Codepoint not yet complete -- still need to advance the
                # context so the next token has the right KV state.
                batch = llama_batch_get_one([new_token_id], n_pos)
                ctx.decode(batch)
                sampled_ids.append(new_token_id)
                n_pos += 1
                n_generated += 1
                continue

            # Handle stop sequences
            if config.stop_sequences:
                # Add piece to buffer and check for stop sequences
                stop_buffer += piece

                # Find earliest stop sequence in buffer
                stop_pos, stop_len = self._find_stop_sequence(stop_buffer, config.stop_sequences)

                if stop_pos is not None:
                    # Stop sequence found - yield text before it and stop
                    text_before_stop = stop_buffer[:stop_pos]
                    if text_before_stop:
                        if on_token:
                            on_token(text_before_stop)
                        yield text_before_stop
                    # Clear buffer to prevent flush at end
                    stop_buffer = ""
                    break

                # No stop found yet - yield text that can't be part of a stop sequence
                # Keep (max_stop_len - 1) characters to detect sequences spanning tokens
                # Example: if max_stop_len=2 and buffer="abc", safe to yield "ab", keep "c"
                chars_to_keep = max_stop_len - 1
                safe_len = len(stop_buffer) - chars_to_keep
                if safe_len > 0:
                    safe_text = stop_buffer[:safe_len]
                    stop_buffer = stop_buffer[safe_len:]
                    if on_token:
                        on_token(safe_text)
                    yield safe_text
            else:
                # No stop sequences - yield immediately
                if on_token:
                    on_token(piece)
                yield piece

            # Prepare next batch
            batch = llama_batch_get_one([new_token_id], n_pos)
            ctx.decode(batch)

            sampled_ids.append(new_token_id)
            n_pos += 1
            n_generated += 1

        # Update the prompt-cache shadow to reflect the actual KV state
        # at this point: prompt_tokens followed by every token we
        # decoded. This includes EOG-truncated runs and stop-sequence
        # truncations because both of those break out of the loop
        # *before* the next decode -- meaning the KV reflects the last
        # appended ``sampled_ids`` entry, not the current iteration's
        # un-decoded one.
        self._kv_seq_tokens = list(prompt_tokens) + sampled_ids

        # Flush any bytes still buffered in the incremental decoder. With
        # errors="replace" this turns dangling continuation bytes into
        # U+FFFD rather than dropping them silently.
        tail = utf8_decoder.decode(b"", final=True)
        if tail:
            if config.stop_sequences:
                stop_buffer += tail
            else:
                if on_token:
                    on_token(tail)
                yield tail

        # Flush remaining buffer (no stop sequence found)
        if config.stop_sequences and stop_buffer:
            if on_token:
                on_token(stop_buffer)
            yield stop_buffer

        # Store streaming stats so callers can retrieve them after exhaustion
        total_time = time.time() - start_time
        self._last_stream_stats = GenerationStats(
            prompt_tokens=n_prompt,
            generated_tokens=n_generated,
            total_time=total_time,
            tokens_per_second=n_generated / total_time if total_time > 0 else 0.0,
        )

        if self.verbose:
            print(f"\nGenerated {n_generated} tokens")
            sampler.print_perf_data()
            ctx.print_perf_data()

    def _build_token_logprob(
        self,
        ctx: LlamaContext,
        token_id: int,
        top_k: int,
    ) -> TokenLogprob:
        """Compute the logprob of ``token_id`` (and optionally top-K) at the last decoded position.

        Uses ``ctx.get_logits_ith(-1)`` to fetch the raw logit vector
        produced by the most recent ``decode`` call. The log-softmax is
        computed in numerically-stable form (subtract max before exp).
        Pure Python because numpy is a dev-only dep; per-token cost is
        ~30ms on a 128k vocab, which is a real but tolerable overhead
        on the ``logprobs=True`` opt-in path. ``top_k`` is the
        commonly-small alternative count and uses ``heapq.nlargest`` so
        per-token top-K cost is O(n log k), not O(n log n).
        """
        logits = ctx.get_logits_ith(-1)
        max_logit = max(logits)
        # ``sum(map(math.exp, ...))`` is ~3x faster than the equivalent
        # for-accumulator loop because the C-implemented map+sum pair
        # avoids per-iteration Python bytecode dispatch.
        exp_sum = sum(map(math.exp, (v - max_logit for v in logits)))
        log_z = max_logit + math.log(exp_sum)
        sampled_logprob = float(logits[token_id]) - log_z

        sampled_piece = self._safe_piece(token_id)

        top_logprobs: List[TopLogprob] = []
        if top_k > 0:
            # Pull top-K by raw logit (equivalent to top-K by logprob
            # since log-softmax is monotonic). ``heapq.nlargest`` is
            # O(n log k); for the typical k=5..20 case this is much
            # cheaper than a full sort over a 128k vocab.
            k = min(top_k, len(logits))
            indexed = heapq.nlargest(k, enumerate(logits), key=lambda kv: kv[1])
            for tid, raw in indexed:
                top_logprobs.append(
                    TopLogprob(
                        token=self._safe_piece(tid),
                        token_id=int(tid),
                        logprob=float(raw) - log_z,
                    )
                )

        return TokenLogprob(
            token=sampled_piece,
            token_id=int(token_id),
            logprob=sampled_logprob,
            top_logprobs=top_logprobs,
        )

    def _safe_piece(self, token_id: int) -> str:
        """Decode ``token_id`` to a string, replacing invalid UTF-8 with U+FFFD.

        Used by the logprob capture path where we want a printable
        representation of every candidate token, even byte-level BPE
        fragments that can't decode cleanly on their own.
        """
        return cast(str, self.vocab.token_to_piece(token_id, special=True))

    def _find_stop_sequence(self, text: str, stop_sequences: List[str]) -> Tuple[Optional[int], int]:
        """
        Find the earliest stop sequence in text.

        Args:
            text: Text to search
            stop_sequences: List of stop sequences to look for

        Returns:
            Tuple of (position, length) where position is the start index of the
            earliest stop sequence found, or (None, 0) if none found.
        """
        earliest_pos = None
        earliest_len = 0

        for stop in stop_sequences:
            pos = text.find(stop)
            if pos != -1:
                # Found a stop sequence - keep track of earliest one
                if earliest_pos is None or pos < earliest_pos:
                    earliest_pos = pos
                    earliest_len = len(stop)

        return earliest_pos, earliest_len

    def generate_with_stats(self, prompt: str, config: Optional[GenerationConfig] = None) -> Response:
        """
        Generate text and return Response with detailed statistics.

        This method is now equivalent to __call__ since Response always
        includes stats. Kept for backward compatibility.

        Args:
            prompt: Input text prompt
            config: Generation configuration

        Returns:
            Response object with text and statistics
        """
        self._try_acquire_busy()
        try:
            return self._generate(prompt, config)
        finally:
            self._busy_lock.release()

    def chat(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
        stream: bool = False,
        template: Optional[str] = None,
    ) -> Union[Response, Iterator[str]]:
        """
        Generate a response from chat messages using the model's chat template.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            config: Generation configuration (uses instance config if None)
            stream: If True, return iterator of text chunks
            template: Custom chat template name (e.g., "llama3", "chatml").
                      If None, uses the model's default template.

        Returns:
            Response object (if stream=False) or iterator of text chunks (if stream=True)

        Example:
            >>> messages = [
            >>>     {"role": "system", "content": "You are helpful."},
            >>>     {"role": "user", "content": "Hello!"}
            >>> ]
            >>> response = llm.chat(messages)
            >>> print(response)  # Works like a string
            >>> print(response.stats.tokens_per_second)  # Access stats
        """
        # No busy-lock acquire here: chat() is a thin wrapper that
        # delegates to __call__, which holds the lock. Acquiring twice
        # would deadlock since the lock is non-reentrant.
        prompt = self._apply_template(messages, template)
        return self(prompt, config=config, stream=stream)

    def _apply_template(
        self,
        messages: List[Dict[str, str]],
        template: Optional[str] = None,
        add_generation_prompt: bool = True,
    ) -> str:
        """Render ``messages`` against the loaded model's chat template.

        Thin wrapper over :func:`inferna._internal.chat_template.apply_template`,
        which owns the Jinja → C-API → simple-format fallback ladder.
        """
        return _apply_chat_template(self.model, messages, template, add_generation_prompt)

    def get_chat_template(self, template_name: Optional[str] = None) -> str:
        """Return the chat template string for ``template_name`` (or the model's default)."""
        return _get_chat_template(self.model, template_name)

    # ------------------------------------------------------------------
    # MCP client surface
    #
    # Lifts the transports in ``inferna.agents.mcp`` onto the high-level
    # LLM API so non-agent callers can attach MCP servers without going
    # through the agent framework. The McpClient is created lazily on the
    # first add_mcp_server() call and torn down in close().
    # ------------------------------------------------------------------

    # MCP delegations (logic lives in inferna._internal.mcp_facade.MCPFacade).

    def add_mcp_server(
        self,
        name: str,
        *,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        transport: Optional[Any] = None,
        request_timeout: Optional[float] = None,
        shutdown_timeout: Optional[float] = None,
    ) -> None:
        """Attach an MCP server and connect immediately.

        Transport is inferred when ``transport`` is omitted: ``command``
        => stdio, ``url`` => http. Tools are namespaced as
        ``"<name>/<tool>"`` so identically named tools from different
        servers stay distinguishable.
        """
        self._mcp.add_server(
            name,
            command=command,
            args=args,
            env=env,
            cwd=cwd,
            url=url,
            headers=headers,
            transport=transport,
            request_timeout=request_timeout,
            shutdown_timeout=shutdown_timeout,
        )

    def remove_mcp_server(self, name: str) -> None:
        """Disconnect and forget an MCP server by name."""
        self._mcp.remove_server(name)

    def list_mcp_tools(self) -> List["McpTool"]:
        """Return all discovered MCP tools as ``McpTool`` instances."""
        return self._mcp.list_tools()

    def list_mcp_resources(self) -> List["McpResource"]:
        """Return all discovered MCP resources as ``McpResource`` instances."""
        return self._mcp.list_resources()

    def call_mcp_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Invoke an MCP tool by full ``"server/tool"`` name.

        Useful when the caller wants explicit control over the tool loop
        rather than handing dispatch to ``chat_with_tools``.
        """
        return self._mcp.call_tool(name, arguments)

    def read_mcp_resource(self, uri: str) -> str:
        """Read the contents of an MCP resource by URI."""
        return self._mcp.read_resource(uri)

    def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        *,
        tools: Optional[List[Any]] = None,
        use_mcp: bool = True,
        max_iterations: int = 8,
        verbose: bool = False,
        system_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> str:
        """Run a tool-calling loop over chat messages.

        Built on top of ``ReActAgent``: the agent prompt is text-based, so
        this works on any GGUF without requiring a model trained for
        OpenAI-style structured tool calls. MCP tools attached via
        ``add_mcp_server()`` are merged with caller-supplied ``tools``
        unless ``use_mcp=False``.

        The last user message is treated as the agent task; any system
        message becomes the agent's system prompt unless ``system_prompt``
        is given explicitly. Multi-turn conversation history beyond a
        single user turn is not yet plumbed through -- the ReAct loop
        operates one task at a time.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            tools: Additional inferna ``Tool`` instances to expose alongside
                MCP tools.
            use_mcp: When True (default), include all MCP tools from
                attached servers.
            max_iterations: Maximum thought/action cycles.
            verbose: Print agent reasoning to stdout.
            system_prompt: Override the system prompt. If None and the
                messages contain a leading system message, that content is
                used.
            generation_config: Override generation config for the loop.

        Returns:
            The agent's final answer string.
        """
        from .agents.react import ReActAgent

        # Resolve task and system prompt from the messages list.
        task: Optional[str] = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                task = str(msg.get("content", ""))
                break
        if task is None:
            raise ValueError("chat_with_tools requires at least one user message")

        if system_prompt is None:
            for msg in messages:
                if msg.get("role") == "system":
                    system_prompt = str(msg.get("content", ""))
                    break

        merged_tools: List[Any] = list(tools) if tools else []
        if use_mcp:
            merged_tools.extend(self._mcp.get_tools_for_agent())

        agent = ReActAgent(
            llm=self,
            tools=merged_tools,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            verbose=verbose,
            generation_config=generation_config,
        )
        result = agent.run(task)
        # AgentResult.answer is always a str; the hasattr guard is
        # defensive against future changes to the agent return type.
        answer = getattr(result, "answer", None)
        return str(answer) if answer is not None else str(result)


# Convenience functions


def complete(
    prompt: str,
    model_path: str,
    config: Optional[GenerationConfig] = None,
    stream: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> Union[Response, Iterator[str]]:
    """
    Convenience function for one-off text completion.

    For repeated completions, use the LLM class for better performance.

    Args:
        prompt: Input text prompt
        model_path: Path to GGUF model file
        config: Generation configuration
        stream: If True, return iterator of text chunks
        verbose: Enable detailed logging from llama.cpp
        **kwargs: Additional config parameters (override config values)

    Returns:
        Response object (if stream=False) or iterator of text chunks (if stream=True).
        The Response can be used as a string: print(response), str(response), etc.

    Example:
        >>> response = complete("Hello", model_path="models/llama.gguf")
        >>> print(response)  # Works like a string
        >>> print(response.text)  # Explicit text access
        >>> print(response.stats.tokens_per_second)  # Access stats
        >>> print(response.to_json())  # JSON output
        >>>
        >>> # With custom parameters
        >>> response = complete(
        >>>     "Tell me a joke",
        >>>     model_path="models/llama.gguf",
        >>>     temperature=0.9,
        >>>     max_tokens=100
        >>> )
    """
    # Merge config with kwargs
    if config is None:
        config = GenerationConfig(**kwargs)
    else:
        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    with LLM(model_path, config=config, verbose=verbose) as llm:
        return llm(prompt, stream=stream)


def chat(
    messages: List[Dict[str, str]],
    model_path: str,
    config: Optional[GenerationConfig] = None,
    stream: bool = False,
    verbose: bool = False,
    template: Optional[str] = None,
    **kwargs: Any,
) -> Union[Response, Iterator[str]]:
    """
    Convenience function for chat-style generation.

    Uses the model's built-in chat template if available, otherwise falls back
    to a simple format. You can also specify a custom template.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model_path: Path to GGUF model file
        config: Generation configuration
        stream: If True, return iterator of text chunks
        verbose: Enable detailed logging from llama.cpp
        template: Custom chat template name (e.g., "llama3", "chatml", "mistral").
                  If None, uses the model's default template.
                  See llama.cpp wiki for supported templates.
        **kwargs: Additional config parameters

    Returns:
        Response object (if stream=False) or iterator of text chunks (if stream=True)

    Example:
        >>> messages = [
        >>>     {"role": "system", "content": "You are a helpful assistant."},
        >>>     {"role": "user", "content": "What is Python?"}
        >>> ]
        >>> response = chat(messages, model_path="models/llama.gguf")
        >>> print(response)  # Works like a string
        >>> print(response.stats)  # Access statistics
        >>>
        >>> # With explicit template
        >>> response = chat(messages, model_path="models/llama.gguf", template="chatml")
    """
    prompt = apply_chat_template(messages, model_path, template, verbose=verbose)
    return complete(prompt, model_path, config, stream, verbose=verbose, **kwargs)


def apply_chat_template(
    messages: List[Dict[str, str]],
    model_path: str,
    template: Optional[str] = None,
    add_generation_prompt: bool = True,
    verbose: bool = False,
) -> str:
    """
    Apply a chat template to format messages into a prompt string.

    Uses the model's built-in chat template from its GGUF metadata. If no template
    is found, falls back to a simple User/Assistant format.

    Supported templates (built into llama.cpp):
        - llama2, llama3
        - chatml (used by many models including Qwen, Yi, etc.)
        - mistral, mistral-v1, mistral-v3, mistral-v3-tekken, mistral-v7
        - phi3, phi4
        - falcon3
        - deepseek, deepseek2, deepseek3
        - command-r
        - vicuna, vicuna-orca
        - zephyr
        - gemma, gemma2
        - orion
        - openchat
        - monarch
        - exaone3
        - granite
        - gigachat
        - megrez

    See: https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
                  Supported roles: 'system', 'user', 'assistant'
        model_path: Path to GGUF model file
        template: Optional template name to use instead of model's default.
                  If None, uses the model's built-in template.
        add_generation_prompt: If True, adds the assistant prompt prefix
        verbose: Enable detailed logging

    Returns:
        Formatted prompt string ready for generation

    Example:
        >>> messages = [
        >>>     {"role": "system", "content": "You are helpful."},
        >>>     {"role": "user", "content": "Hello!"}
        >>> ]
        >>> prompt = apply_chat_template(messages, "model.gguf")
        >>> print(prompt)
        <|im_start|>system
        You are helpful.<|im_end|>
        <|im_start|>user
        Hello!<|im_end|>
        <|im_start|>assistant
    """
    if not verbose:
        disable_logging()

    ggml_backend_load_all()

    # Load model to get template (n_gpu_layers=0: metadata only, no GPU alloc)
    model_params = LlamaModelParams()
    model_params.n_gpu_layers = 0
    model = LlamaModel(model_path, model_params)

    return _apply_chat_template(model, messages, template, add_generation_prompt)


def get_chat_template(model_path: str, template_name: Optional[str] = None) -> str:
    """
    Get the chat template string from a model.

    Args:
        model_path: Path to GGUF model file
        template_name: Optional specific template name to retrieve

    Returns:
        Template string, or empty string if not found

    Example:
        >>> template = get_chat_template("models/llama.gguf")
        >>> print(template)  # Shows the Jinja-style template
    """
    disable_logging()
    ggml_backend_load_all()

    model_params = LlamaModelParams()
    model_params.n_gpu_layers = 0
    model = LlamaModel(model_path, model_params)

    return _get_chat_template(model, template_name)


def simple(
    model_path: str,
    prompt: str,
    ngl: int = DEFAULT_N_GPU_LAYERS,
    n_predict: int = 32,
    n_ctx: Optional[int] = None,
    verbose: bool = False,
) -> bool:
    """
    Simple, educational example showing raw llama.cpp usage.

    This function demonstrates how to use llama.cpp primitives directly
    without the abstractions provided by LLM or complete().

    Args:
        model_path: Path to GGUF model file
        prompt: Input text prompt
        ngl: Number of GPU layers (default: 99)
        n_predict: Number of tokens to generate (default: 32)
        n_ctx: Context size (default: auto-calculated)
        verbose: Enable llama.cpp logging (default: False)

    Returns:
        True if successful
    """
    from .llama import llama_cpp as cy

    # load dynamic backends

    if not verbose:
        cy.disable_logging()

    cy.ggml_backend_load_all()

    # initialize the model

    model_params = cy.LlamaModelParams()
    model_params.n_gpu_layers = ngl

    model = cy.LlamaModel(model_path, model_params)
    vocab = model.get_vocab()

    # tokenize the prompt
    print(f"vocab.n_vocab = {vocab.n_vocab}")

    # find the number of tokens in the prompt
    prompt_tokens = vocab.tokenize(prompt, add_special=True, parse_special=True)
    n_prompt = len(prompt_tokens)
    print(f"n_prompt: {n_prompt}")

    # initialize the context

    ctx_params = cy.LlamaContextParams()
    # n_ctx is the context size
    if n_ctx is not None:
        ctx_params.n_ctx = n_ctx
    else:
        ctx_params.n_ctx = n_prompt + n_predict - 1
    # n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = n_prompt
    # enable performance counters
    ctx_params.no_perf = False

    ctx = cy.LlamaContext(model, ctx_params)

    # initialize the sampler

    sparams = cy.LlamaSamplerChainParams()
    sparams.no_perf = False

    smplr = cy.LlamaSampler(sparams)
    smplr.add_greedy()

    # print the prompt token-by-token
    print()
    prompt = ""
    for i in prompt_tokens:
        try:
            prompt += vocab.token_to_piece(i, lstrip=0, special=False)
        except UnicodeDecodeError:
            continue
    print(prompt)

    # prepare a batch for the prompt
    batch = cy.llama_batch_get_one(prompt_tokens)

    # main loop
    t_main_start: int = cy.ggml_time_us()
    n_decode = 0

    n_pos = n_prompt
    response = ""
    for i in range(n_predict):
        ctx.decode(batch)

        # sample the next token
        new_token_id = smplr.sample(ctx, -1)

        # is it an end of generation?
        if vocab.is_eog(new_token_id):
            break

        piece: str = vocab.token_to_piece(new_token_id, special=True)
        response += piece

        # prepare the next batch with the sampled token
        batch = cy.llama_batch_get_one([new_token_id], n_pos)
        n_pos += 1

        n_decode += 1

    print()
    print(f"response: {response}")
    print()

    t_main_end: int = cy.ggml_time_us()

    print(
        "decoded %d tokens in %.2f s, speed: %.2f t/s"
        % (n_decode, (t_main_end - t_main_start) / 1000000.0, n_decode / ((t_main_end - t_main_start) / 1000000.0))
    )
    print()

    smplr.print_perf_data()
    ctx.print_perf_data()

    return True


# =============================================================================
# Async API
# =============================================================================


class AsyncLLM:
    """
    Async wrapper around the LLM class for non-blocking text generation.

    This class provides an async interface to the synchronous LLM operations.
    Inference runs in a thread pool to avoid blocking the event loop, making
    it suitable for use in async web frameworks like FastAPI, aiohttp, etc.

    Note: The underlying model is still synchronous - this wrapper just moves
    the blocking operations off the main event loop. For true parallelism with
    multiple requests, use multiple AsyncLLM instances or batch processing.

    Resource Management:
        Use as an async context manager for proper cleanup:
        - `async with AsyncLLM(...) as llm:`

    Example:
        >>> async def main():
        >>>     # Simple usage with direct parameters
        >>>     async with AsyncLLM("model.gguf", temperature=0.9) as llm:
        >>>         response = await llm("What is Python?")
        >>>         print(response)
        >>>
        >>>         # Async streaming
        >>>         async for chunk in llm.stream("Tell me a joke"):
        >>>             print(chunk, end="", flush=True)
        >>>
        >>> asyncio.run(main())
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[GenerationConfig] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize async generator with a model.

        Args:
            model_path: Path to GGUF model file
            config: Generation configuration (uses defaults if None)
            verbose: Print detailed information during generation
            **kwargs: Generation parameters (temperature, max_tokens, etc.)
                      These override values in config if both are provided.

        Example:
            >>> # Direct parameters
            >>> llm = AsyncLLM("model.gguf", temperature=0.9, max_tokens=100)
            >>>
            >>> # With config
            >>> config = GenerationConfig(temperature=0.9)
            >>> llm = AsyncLLM("model.gguf", config=config)
        """
        self._llm = LLM(model_path, config=config, verbose=verbose, **kwargs)
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "AsyncLLM":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Async context manager exit - ensures cleanup."""
        await self.close()

    async def close(self) -> None:
        """
        Explicitly release resources.

        Runs cleanup in a thread to avoid blocking if cleanup is slow.
        """
        await asyncio.to_thread(self._llm.close)

    async def reset_context(self) -> None:
        """Force recreation of context on next generation."""
        await asyncio.to_thread(self._llm.reset_context)

    @property
    def config(self) -> GenerationConfig:
        """Get the current generation config."""
        return self._llm.config

    @property
    def model_path(self) -> str:
        """Get the model path."""
        return self._llm.model_path

    async def __call__(self, prompt: str, config: Optional[GenerationConfig] = None, **kwargs: Any) -> Response:
        """
        Generate text from a prompt asynchronously.

        Args:
            prompt: Input text prompt
            config: Generation configuration (uses instance config if None)
            **kwargs: Override config parameters for this call

        Returns:
            Response object with text and statistics

        Example:
            >>> response = await llm("What is the meaning of life?")
            >>> print(response)  # Works like a string
            >>> print(response.stats.tokens_per_second)  # Access stats
            >>> response = await llm("Explain quantum physics", max_tokens=200)
        """
        # Build config with overrides if kwargs provided
        effective_config: Optional[GenerationConfig]
        if kwargs:
            effective_config = self._build_config(config, kwargs)
        else:
            effective_config = config

        async with self._lock:
            return cast(
                Response,
                await asyncio.to_thread(self._llm._generate, prompt, effective_config),
            )

    async def generate(self, prompt: str, config: Optional[GenerationConfig] = None, **kwargs: Any) -> Response:
        """
        Generate text from a prompt asynchronously.

        Alias for __call__ for explicit method name preference.

        Args:
            prompt: Input text prompt
            config: Generation configuration
            **kwargs: Override config parameters

        Returns:
            Response object with text and statistics
        """
        return await self(prompt, config, **kwargs)

    async def stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream generated text chunks asynchronously.

        Yields text chunks as they are generated. Each chunk is yielded
        as soon as it's available from the underlying model.

        Args:
            prompt: Input text prompt
            config: Generation configuration
            timeout: Maximum seconds to wait for each chunk (None = no limit)
            **kwargs: Override config parameters

        Yields:
            Text chunks as they are generated

        Example:
            >>> async for chunk in llm.stream("Tell me a story"):
            >>>     print(chunk, end="", flush=True)
        """
        # Build config with overrides if kwargs provided
        effective_config: Optional[GenerationConfig]
        if kwargs:
            effective_config = self._build_config(config, kwargs)
        else:
            effective_config = config

        # Use a queue to bridge sync generator to async iterator
        queue: asyncio.Queue[Union[str, None, Exception]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        async def producer() -> None:
            """Run sync generator in thread and put items in queue."""
            try:

                def generate_sync() -> None:
                    for chunk in self._llm._generate_stream(prompt, effective_config):
                        # Schedule putting item in queue from the thread
                        asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
                    # Signal completion
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)

                await asyncio.to_thread(generate_sync)
            except Exception as e:
                await queue.put(e)

        # Start producer task
        async with self._lock:
            producer_task = asyncio.create_task(producer())

            try:
                while True:
                    if timeout is not None:
                        item = await asyncio.wait_for(queue.get(), timeout=timeout)
                    else:
                        item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                # Ensure producer completes; if it hasn't started yet or is
                # still running, cancel it and suppress CancelledError.
                if not producer_task.done():
                    producer_task.cancel()
                    try:
                        await producer_task
                    except asyncio.CancelledError:
                        pass
                else:
                    # Retrieve (and discard) the result so any exception
                    # doesn't become an unhandled task exception.
                    try:
                        producer_task.result()
                    except Exception:
                        pass

    async def generate_with_stats(self, prompt: str, config: Optional[GenerationConfig] = None) -> Response:
        """
        Generate text and return Response with detailed statistics.

        This method is now equivalent to __call__ since Response always
        includes stats. Kept for backward compatibility.

        Args:
            prompt: Input text prompt
            config: Generation configuration

        Returns:
            Response object with text and statistics
        """
        async with self._lock:
            return cast(
                Response,
                await asyncio.to_thread(self._llm.generate_with_stats, prompt, config),
            )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
        template: Optional[str] = None,
    ) -> Response:
        """
        Generate a response from chat messages using the model's chat template.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            config: Generation configuration (uses instance config if None)
            template: Custom chat template name (e.g., "llama3", "chatml").
                      If None, uses the model's default template.

        Returns:
            Response object with text and statistics

        Example:
            >>> messages = [
            >>>     {"role": "system", "content": "You are helpful."},
            >>>     {"role": "user", "content": "Hello!"}
            >>> ]
            >>> response = await llm.chat(messages)
            >>> print(response)  # Works like a string
            >>> print(response.stats)  # Access statistics
        """
        async with self._lock:
            return cast(
                Response,
                await asyncio.to_thread(
                    self._llm.chat,
                    messages,
                    config,
                    False,  # stream=False
                    template,
                ),
            )

    def get_chat_template(self, template_name: Optional[str] = None) -> str:
        """
        Get the chat template string from the loaded model.

        Args:
            template_name: Optional specific template name to retrieve

        Returns:
            Template string, or empty string if not found
        """
        return self._llm.get_chat_template(template_name)

    def _build_config(self, base_config: Optional[GenerationConfig], overrides: Dict[str, Any]) -> GenerationConfig:
        """Build a config with overrides applied."""
        config = base_config or self._llm.config
        config_dict = config.to_dict()
        config_dict.update(overrides)
        return GenerationConfig(**config_dict)


async def complete_async(
    prompt: str, model_path: str, config: Optional[GenerationConfig] = None, verbose: bool = False, **kwargs: Any
) -> Response:
    """
    Async convenience function for one-off text completion.

    For repeated completions, use the AsyncLLM class for better performance
    (avoids reloading the model each time).

    Args:
        prompt: Input text prompt
        model_path: Path to GGUF model file
        config: Generation configuration
        verbose: Enable detailed logging
        **kwargs: Additional config parameters (temperature, max_tokens, etc.)

    Returns:
        Response object with text and statistics

    Example:
        >>> response = await complete_async(
        >>>     "What is Python?",
        >>>     model_path="model.gguf",
        >>>     temperature=0.7
        >>> )
        >>> print(response)  # Works like a string
        >>> print(response.stats)  # Access statistics
    """
    return cast(
        Response,
        await asyncio.to_thread(
            complete,
            prompt,
            model_path,
            config,
            False,  # stream=False
            verbose,
            **kwargs,
        ),
    )


async def chat_async(
    messages: List[Dict[str, str]],
    model_path: str,
    config: Optional[GenerationConfig] = None,
    verbose: bool = False,
    **kwargs: Any,
) -> Response:
    """
    Async convenience function for chat-style generation.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model_path: Path to GGUF model file
        config: Generation configuration
        verbose: Enable detailed logging
        **kwargs: Additional config parameters

    Returns:
        Response object with text and statistics

    Example:
        >>> messages = [
        >>>     {"role": "system", "content": "You are a helpful assistant."},
        >>>     {"role": "user", "content": "What is Python?"}
        >>> ]
        >>> response = await chat_async(messages, model_path="model.gguf")
        >>> print(response)  # Works like a string
    """
    return cast(
        Response,
        await asyncio.to_thread(
            chat,
            messages,
            model_path,
            config,
            False,  # stream=False
            verbose,
            **kwargs,
        ),
    )


async def stream_complete_async(
    prompt: str, model_path: str, config: Optional[GenerationConfig] = None, verbose: bool = False, **kwargs: Any
) -> AsyncIterator[str]:
    """
    Async streaming completion for one-off use.

    For repeated completions, use AsyncLLM.stream() for better performance.

    Args:
        prompt: Input text prompt
        model_path: Path to GGUF model file
        config: Generation configuration
        verbose: Enable detailed logging
        **kwargs: Additional config parameters

    Yields:
        Text chunks as they are generated

    Example:
        >>> async for chunk in stream_complete_async("Tell me a story", "model.gguf"):
        >>>     print(chunk, end="", flush=True)
    """
    async with AsyncLLM(model_path, config=config, verbose=verbose, **kwargs) as llm:
        async for chunk in llm.stream(prompt):
            yield chunk
