from .defaults import (  # noqa: F401
    LLAMA_DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_MIN_P,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_PENALTY_LAST_N,
    DEFAULT_PENALTY_FREQ,
    DEFAULT_PENALTY_PRESENT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_N_GPU_LAYERS,
    DEFAULT_N_BATCH,
    DEFAULT_MAIN_GPU,
    DEFAULT_SPLIT_MODE,
)

# High-level API
from .api import (
    LLM,
    complete,
    chat,
    simple,
    GenerationConfig,
    GenerationStats,
    Response,
    ResponseCacheInfo,
    # Async API
    AsyncLLM,
    complete_async,
    chat_async,
)

# Tool-calling types live in an internal module; re-export at the top
# level so callers don't reach into _internal.
from ._internal.function_calling import FunctionCall as FunctionCall
from ._internal.function_calling import ToolCall as ToolCall

# Batching
from .batching import batch_generate, BatchGenerator, BatchRequest, BatchResponse

# Memory utilities
from .memory import estimate_gpu_layers, estimate_memory_usage

__version__ = "0.1.2"
