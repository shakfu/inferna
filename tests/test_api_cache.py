"""
Tests for LLM response caching functionality.

Tests cover:
- Cache disabled by default
- Cache enabled with cache_size > 0
- Cache hits for identical prompts
- Cache misses for different prompts/configs
- LRU eviction when full
- Cache clear resets stats
- TTL expiration
- Random seed (seed=-1) bypasses cache
- Cached responses retain stats and model path
"""

import dataclasses
import time

import pytest

from inferna import LLM, GenerationConfig, ResponseCacheInfo
from inferna._internal.response_cache import CACHE_KEY_EXCLUDED_FIELDS, make_cache_key


class TestCacheDisabled:
    """Tests for cache disabled (default) behavior."""

    def test_cache_disabled_by_default(self, model_path):
        """Cache should be disabled when cache_size=0 (default)."""
        with LLM(model_path, max_tokens=16) as llm:
            assert llm.cache_enabled is False
            assert llm.cache_info() is None

    def test_cache_clear_noop_when_disabled(self, model_path):
        """cache_clear() should not raise when cache is disabled."""
        with LLM(model_path, max_tokens=16) as llm:
            llm.cache_clear()  # Should not raise
            assert llm.cache_info() is None


class TestCacheEnabled:
    """Tests for cache enabled behavior."""

    def test_cache_enabled_with_size(self, model_path):
        """Cache should be enabled when cache_size > 0."""
        with LLM(model_path, max_tokens=16, cache_size=10) as llm:
            assert llm.cache_enabled is True
            info = llm.cache_info()
            assert info is not None
            assert info.maxsize == 10
            assert info.currsize == 0
            assert info.hits == 0
            assert info.misses == 0

    def test_cache_enabled_with_ttl(self, model_path):
        """Cache should store TTL value."""
        with LLM(model_path, max_tokens=16, cache_size=10, cache_ttl=60.0) as llm:
            info = llm.cache_info()
            assert info.ttl == 60.0

    def test_cache_info_type(self, model_path):
        """cache_info() should return ResponseCacheInfo NamedTuple."""
        with LLM(model_path, max_tokens=16, cache_size=10) as llm:
            info = llm.cache_info()
            assert isinstance(info, ResponseCacheInfo)
            assert hasattr(info, "hits")
            assert hasattr(info, "misses")
            assert hasattr(info, "maxsize")
            assert hasattr(info, "currsize")
            assert hasattr(info, "ttl")


class TestCacheHitsMisses:
    """Tests for cache hit/miss behavior."""

    def test_cache_miss_first_call(self, model_path):
        """First call should be a cache miss."""
        with LLM(model_path, max_tokens=16, cache_size=10, seed=42) as llm:
            llm("Hello")
            info = llm.cache_info()
            assert info.misses == 1
            assert info.hits == 0
            assert info.currsize == 1

    def test_cache_hit_same_prompt(self, model_path):
        """Same prompt should hit cache."""
        with LLM(model_path, max_tokens=16, cache_size=10, seed=42) as llm:
            response1 = llm("Hello")
            response2 = llm("Hello")

            info = llm.cache_info()
            assert info.hits == 1
            assert info.misses == 1
            assert info.currsize == 1

            # Responses should be identical
            assert response1.text == response2.text

    def test_cache_miss_different_prompt(self, model_path):
        """Different prompts should be cache misses."""
        with LLM(model_path, max_tokens=16, cache_size=10, seed=42) as llm:
            llm("Hello")
            llm("Goodbye")

            info = llm.cache_info()
            assert info.misses == 2
            assert info.hits == 0
            assert info.currsize == 2

    def test_cache_miss_different_temperature(self, model_path):
        """Same prompt with different temperature should miss cache."""
        with LLM(model_path, max_tokens=16, cache_size=10, seed=42) as llm:
            llm("Hello")
            llm("Hello", config=GenerationConfig(max_tokens=16, temperature=0.5, seed=42))

            info = llm.cache_info()
            assert info.misses == 2
            assert info.currsize == 2

    def test_cache_miss_different_max_tokens(self, model_path):
        """Same prompt with different max_tokens should miss cache."""
        with LLM(model_path, max_tokens=16, cache_size=10, seed=42) as llm:
            llm("Hello")
            llm("Hello", config=GenerationConfig(max_tokens=32, seed=42))

            info = llm.cache_info()
            assert info.misses == 2
            assert info.currsize == 2


class TestCacheLRUEviction:
    """Tests for LRU cache eviction behavior."""

    def test_lru_eviction(self, model_path):
        """Oldest entry should be evicted when cache is full."""
        with LLM(model_path, max_tokens=8, cache_size=2, seed=42) as llm:
            llm("A")  # cache: [A], miss 1
            llm("B")  # cache: [A, B], miss 2
            llm("C")  # cache: [B, C], miss 3 - A evicted

            info = llm.cache_info()
            assert info.currsize == 2
            assert info.misses == 3

            # B should hit (still in cache from step 2)
            llm("B")
            info = llm.cache_info()
            assert info.hits == 1
            # cache: [C, B] - B moved to end

            # C should hit (still in cache)
            llm("C")
            info = llm.cache_info()
            assert info.hits == 2

            # A should miss (was evicted at step 3)
            llm("A")
            info = llm.cache_info()
            assert info.misses == 4

    def test_lru_access_updates_order(self, model_path):
        """Accessing an entry should move it to end (most recent)."""
        with LLM(model_path, max_tokens=8, cache_size=2, seed=42) as llm:
            llm("A")  # cache: [A]
            llm("B")  # cache: [A, B]
            llm("A")  # cache: [B, A] - A moved to end (hit)
            llm("C")  # cache: [A, C] - B evicted (oldest)

            info = llm.cache_info()
            assert info.currsize == 2

            # B should miss (was evicted)
            llm("B")
            info = llm.cache_info()
            assert info.misses == 4  # A, B, C, B again


class TestCacheClear:
    """Tests for cache_clear() behavior."""

    def test_cache_clear_resets_stats(self, model_path):
        """cache_clear() should reset all statistics."""
        with LLM(model_path, max_tokens=16, cache_size=10, seed=42) as llm:
            llm("Hello")
            llm("Hello")  # hit

            info = llm.cache_info()
            assert info.hits == 1
            assert info.misses == 1
            assert info.currsize == 1

            llm.cache_clear()

            info = llm.cache_info()
            assert info.hits == 0
            assert info.misses == 0
            assert info.currsize == 0

    def test_cache_clear_forces_regeneration(self, model_path):
        """After clear, same prompt should regenerate."""
        with LLM(model_path, max_tokens=16, cache_size=10, seed=42) as llm:
            llm("Hello")
            llm.cache_clear()
            llm("Hello")

            info = llm.cache_info()
            assert info.misses == 1  # New miss after clear
            assert info.currsize == 1


class TestCacheTTL:
    """Tests for TTL expiration behavior."""

    def test_ttl_expiration(self, model_path):
        """Expired entries should be treated as misses."""
        with LLM(model_path, max_tokens=16, cache_size=10, cache_ttl=0.1, seed=42) as llm:
            llm("Hello")

            info = llm.cache_info()
            assert info.currsize == 1
            assert info.misses == 1

            # Wait for TTL to expire
            time.sleep(0.15)

            # Same prompt should miss (expired)
            llm("Hello")

            info = llm.cache_info()
            assert info.misses == 2  # Both misses

    def test_ttl_not_expired(self, model_path):
        """Non-expired entries should hit cache."""
        with LLM(model_path, max_tokens=16, cache_size=10, cache_ttl=10.0, seed=42) as llm:
            llm("Hello")
            llm("Hello")  # Should hit

            info = llm.cache_info()
            assert info.hits == 1

    def test_no_ttl_never_expires(self, model_path):
        """Entries without TTL should never expire."""
        with LLM(model_path, max_tokens=16, cache_size=10, seed=42) as llm:
            llm("Hello")
            llm("Hello")  # Should hit

            info = llm.cache_info()
            assert info.ttl is None
            assert info.hits == 1


class TestRandomSeedBypass:
    """Tests for random seed cache bypass."""

    def test_random_seed_bypasses_cache(self, model_path):
        """seed=-1 (random) should bypass cache entirely."""
        with LLM(model_path, max_tokens=16, cache_size=10) as llm:
            # Default seed is -1
            llm("Hello")
            llm("Hello")

            info = llm.cache_info()
            # Neither call should have used cache
            assert info.hits == 0
            assert info.misses == 0
            assert info.currsize == 0

    def test_fixed_seed_uses_cache(self, model_path):
        """Fixed seed should use cache."""
        with LLM(model_path, max_tokens=16, cache_size=10, seed=42) as llm:
            llm("Hello")
            llm("Hello")

            info = llm.cache_info()
            assert info.hits == 1
            assert info.currsize == 1


class TestCachedResponseMetadata:
    """Tests for cached response metadata preservation."""

    def test_cached_response_has_stats(self, model_path):
        """Cached responses should retain stats."""
        with LLM(model_path, max_tokens=16, cache_size=10, seed=42) as llm:
            response1 = llm("Hello")
            response2 = llm("Hello")  # From cache

            assert response1.stats is not None
            assert response2.stats is not None
            # Stats should be identical (same object from cache)
            assert response2.stats.prompt_tokens == response1.stats.prompt_tokens
            assert response2.stats.generated_tokens == response1.stats.generated_tokens

    def test_cached_response_has_model_path(self, model_path):
        """Cached responses should retain model path."""
        with LLM(model_path, max_tokens=16, cache_size=10, seed=42) as llm:
            response1 = llm("Hello")
            response2 = llm("Hello")  # From cache

            assert response1.model == model_path
            assert response2.model == model_path

    def test_cached_response_has_finish_reason(self, model_path):
        """Cached responses should retain finish_reason."""
        with LLM(model_path, max_tokens=16, cache_size=10, seed=42) as llm:
            response1 = llm("Hello")
            response2 = llm("Hello")  # From cache

            assert response1.finish_reason == "stop"
            assert response2.finish_reason == "stop"


class TestStreamingNotCached:
    """Tests to verify streaming is not cached."""

    def test_streaming_not_cached(self, model_path):
        """Streaming responses should not be cached."""
        with LLM(model_path, max_tokens=16, cache_size=10, seed=42) as llm:
            # Streaming call
            chunks = list(llm("Hello", stream=True))
            assert len(chunks) > 0

            info = llm.cache_info()
            # Streaming should not populate cache
            assert info.currsize == 0
            assert info.misses == 0
            assert info.hits == 0


class TestExportedFromPackage:
    """Tests for proper export from package."""

    def test_response_cache_info_exported(self):
        """ResponseCacheInfo should be exported from inferna."""
        from inferna import ResponseCacheInfo

        assert ResponseCacheInfo is not None
        # Verify it's a NamedTuple
        assert hasattr(ResponseCacheInfo, "_fields")
        assert "hits" in ResponseCacheInfo._fields


class TestCacheKeyFields:
    """Every GenerationConfig field is either in the key or excluded with a
    reason. A new field joins the key by default; this table makes the test
    fail until it is given a distinguishing value."""

    SENTINEL = -1
    # A valid value for each keyed field that differs from BASE.
    ALTERNATES = {
        "max_tokens": 17,
        "temperature": 0.3,
        "top_k": 7,
        "top_p": 0.5,
        "min_p": 0.2,
        "repeat_penalty": 1.3,
        "frequency_penalty": 0.4,
        "presence_penalty": 0.6,
        "penalty_last_n": 32,
        "mirostat": 2,
        "mirostat_tau": 3.0,
        "mirostat_eta": 0.2,
        "typical_p": 0.7,
        "typical_min_keep": 3,
        "xtc_probability": 0.5,
        "xtc_threshold": 0.2,
        "dynatemp_range": 0.5,
        "dynatemp_exponent": 2.0,
        "logit_bias": {15: -5.0},
        "n_ctx": 1024,
        "seed": 43,
        "stop_sequences": ["END"],
        "add_bos": False,
        "parse_special": False,
    }
    BASE = GenerationConfig(seed=42)

    def _key(self, config: GenerationConfig) -> str:
        key = make_cache_key("hello", config, random_seed_sentinel=self.SENTINEL)
        assert key is not None
        return key

    def test_every_field_is_keyed_or_excluded(self):
        names = {f.name for f in dataclasses.fields(GenerationConfig)}
        assert set(CACHE_KEY_EXCLUDED_FIELDS) <= names
        assert names - set(CACHE_KEY_EXCLUDED_FIELDS) == set(self.ALTERNATES)
        assert all(CACHE_KEY_EXCLUDED_FIELDS.values()), "each exclusion needs a reason"

    @pytest.mark.parametrize("name", sorted(ALTERNATES))
    def test_keyed_field_changes_the_key(self, name):
        changed = dataclasses.replace(self.BASE, **{name: self.ALTERNATES[name]})
        assert self._key(changed) != self._key(self.BASE)

    @pytest.mark.parametrize(
        "name,value",
        [("n_gpu_layers", 0), ("main_gpu", 1), ("split_mode", 2), ("tensor_split", [0.5, 0.5]), ("n_batch", 64)],
    )
    def test_excluded_field_does_not_change_the_key(self, name, value):
        assert name in CACHE_KEY_EXCLUDED_FIELDS
        changed = dataclasses.replace(self.BASE, **{name: value})
        assert self._key(changed) == self._key(self.BASE)

    def test_order_insensitive_values_hash_alike(self):
        a = dataclasses.replace(self.BASE, stop_sequences=["a", "b"], logit_bias={1: 1.0, 2: -1.0})
        b = dataclasses.replace(self.BASE, stop_sequences=["b", "a"], logit_bias={2: -1.0, 1: 1.0})
        assert self._key(a) == self._key(b)

    def test_random_seed_sentinel_returns_none(self):
        config = dataclasses.replace(self.BASE, seed=self.SENTINEL)
        assert make_cache_key("hello", config, random_seed_sentinel=self.SENTINEL) is None
