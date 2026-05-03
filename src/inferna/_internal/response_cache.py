"""LRU+TTL response cache for :class:`inferna.LLM`.

Extracted from ``api.py`` so the cache can be reused (or swapped) without
dragging the rest of the LLM surface along. The implementation is the
same one that lived inline as ``_ResponseLRUCache``; the public surface
on ``LLM`` (``cache_enabled`` / ``cache_info()`` / ``cache_clear()``) is
unchanged.

``make_cache_key`` is the canonical key derivation: it includes only
output-affecting fields of :class:`GenerationConfig` (prompt,
temperature, top_k, top_p, min_p, repeat_penalty, max_tokens, sorted
stop_sequences, seed, add_bos, parse_special) and excludes
infrastructure (n_gpu_layers, main_gpu, split_mode, tensor_split, n_ctx,
n_batch). It returns ``None`` when ``seed == LLAMA_DEFAULT_SEED`` so
non-deterministic runs bypass the cache automatically.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any, NamedTuple, Optional, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from ..api import GenerationConfig


class ResponseCacheInfo(NamedTuple):
    """Cache statistics for LLM response caching."""

    hits: int
    misses: int
    maxsize: int
    currsize: int
    ttl: Optional[float]  # TTL in seconds, None if no expiration


class ResponseCache:
    """LRU cache for Response objects with optional TTL support.

    Stores ``(value, timestamp)`` tuples in an ``OrderedDict`` and checks
    expiration on ``get``; expired entries count as misses and are
    removed in-line.
    """

    def __init__(self, maxsize: int, ttl: Optional[float] = None) -> None:
        self._maxsize = maxsize
        self._ttl = ttl
        self._cache: "OrderedDict[str, Tuple[Any, float]]" = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Return the cached value, or ``None`` for miss / expired."""
        if key not in self._cache:
            self._misses += 1
            return None

        value, timestamp = self._cache[key]

        if self._ttl is not None and time.time() - timestamp > self._ttl:
            del self._cache[key]
            self._misses += 1
            return None

        self._cache.move_to_end(key)
        self._hits += 1
        return value

    def put(self, key: str, value: Any) -> None:
        """Store ``value``. Evicts the LRU entry when full."""
        if key in self._cache:
            self._cache[key] = (value, time.time())
            self._cache.move_to_end(key)
            return

        if len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)
        self._cache[key] = (value, time.time())

    def clear(self) -> None:
        """Drop all entries and reset hit/miss counters."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def info(self) -> ResponseCacheInfo:
        return ResponseCacheInfo(
            hits=self._hits,
            misses=self._misses,
            maxsize=self._maxsize,
            currsize=len(self._cache),
            ttl=self._ttl,
        )


def make_cache_key(
    prompt: str,
    config: "GenerationConfig",
    *,
    random_seed_sentinel: int,
) -> Optional[str]:
    """Derive a stable cache key from ``prompt`` + output-affecting config.

    Returns ``None`` when ``config.seed == random_seed_sentinel`` (i.e.
    the caller asked for a fresh sample on every call) so the cache is
    bypassed automatically. Callers pass ``LLAMA_DEFAULT_SEED`` from
    ``llama.llama_cpp``; the sentinel is wired through as a parameter
    rather than imported here so this module stays import-cycle-free.
    """
    if config.seed == random_seed_sentinel:
        return None

    key_parts = [
        prompt,
        str(config.temperature),
        str(config.top_k),
        str(config.top_p),
        str(config.min_p),
        str(config.repeat_penalty),
        str(config.max_tokens),
        str(sorted(config.stop_sequences)),
        str(config.seed),
        str(config.add_bos),
        str(config.parse_special),
    ]
    key_str = "\0".join(key_parts)
    return hashlib.sha256(key_str.encode()).hexdigest()
