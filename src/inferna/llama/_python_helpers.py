"""Pure-Python helpers extracted from llama_cpp.pyx during the nanobind migration.

Three groups:
  1. Memory pools (TokenMemoryPool, BatchMemoryPool + global instances).
  2. Model download + HF/Docker resolution (download_model, get_hf_file, ...).
  3. N-gram cache (NgramCache).
"""

from __future__ import annotations

import glob as _glob
import json
import os
import re
import struct
import time
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Imported lazily inside callers to avoid an import cycle through llama_cpp.py.
# from . import _llama_native as _N
#
# nanobind-bound `LlamaBatch` has no static type stub, so mypy sees it
# as `Any`. Public batch-pool helpers below are typed `Any` for that
# reason — honest about the boundary, and downstream callers still get
# correct runtime behavior. `# type: ignore[no-any-return]` would be
# noisier without adding information.


# =============================================================================
# Memory pools
# =============================================================================


class TokenMemoryPool:
    """Memory pool for efficient token list reuse."""

    def __init__(self, max_pool_size: int = 10, max_token_size: int = 1024) -> None:
        self._pools: dict[int, list[list[int]]] = {}
        self._usage_count: dict[int, int] = {}
        self._max_pool_size = max_pool_size
        self._max_token_size = max_token_size
        for size in (8, 16, 32, 64, 128, 256, 512):
            if size <= max_token_size:
                self._pools[size] = []

    def get_token_list(self, size: int) -> list[int]:
        if size > self._max_token_size:
            return [0] * size
        self._usage_count[size] = self._usage_count.get(size, 0) + 1
        if size in self._pools and self._pools[size]:
            tl = self._pools[size].pop()
            if len(tl) != size:
                tl = [0] * size
            else:
                for i in range(size):
                    tl[i] = 0
            return tl
        return [0] * size

    def return_token_list(self, token_list: list[int]) -> None:
        size = len(token_list)
        if size > self._max_token_size:
            return
        if size not in self._pools:
            self._pools[size] = []
        if len(self._pools[size]) < self._max_pool_size:
            self._pools[size].append(token_list)

    def get_stats(self) -> dict[str, object]:
        return {
            "pool_sizes": {s: len(p) for s, p in self._pools.items()},
            "usage_count": dict(self._usage_count),
            "total_pools": len(self._pools),
            "total_pooled_lists": sum(len(p) for p in self._pools.values()),
        }


_global_token_pool = TokenMemoryPool()


def get_token_pool_stats() -> dict[str, object]:
    return _global_token_pool.get_stats()


def reset_token_pool() -> None:
    global _global_token_pool
    _global_token_pool = TokenMemoryPool()


# Pool key shape: (n_tokens, embd, n_seq_max). LlamaBatch is a native
# nanobind-bound class with no public type stub, so it stays Any here.
_BatchKey = tuple[int, int, int]


class BatchMemoryPool:
    """Memory pool for efficient LlamaBatch reuse."""

    def __init__(self, max_pool_size: int = 5, max_batch_size: int = 512) -> None:
        self._pools: dict[_BatchKey, list[Any]] = {}
        self._usage_count: dict[_BatchKey, int] = {}
        self._max_pool_size = max_pool_size
        self._max_batch_size = max_batch_size

    def get_batch(self, n_tokens: int, embd: int, n_seq_max: int) -> Any:
        # Imported lazily to avoid an import cycle.
        from . import _llama_native as _N

        key: _BatchKey = (n_tokens, embd, n_seq_max)
        if n_tokens > self._max_batch_size:
            return _N.LlamaBatch(n_tokens=n_tokens, embd=embd, n_seq_max=n_seq_max)
        self._usage_count[key] = self._usage_count.get(key, 0) + 1
        if key in self._pools and self._pools[key]:
            batch = self._pools[key].pop()
            try:
                batch.reset()
            except Exception:
                pass
            return batch
        return _N.LlamaBatch(n_tokens=n_tokens, embd=embd, n_seq_max=n_seq_max)

    def return_batch(self, batch: Any) -> None:
        # Pool key is the construction-time capacity, not the live token count.
        # LlamaBatch exposes both as `_n_tokens` and `n_tokens_capacity`.
        n_tokens: int = batch._n_tokens
        embd: int = batch.embd
        n_seq_max: int = batch.n_seq_max
        key: _BatchKey = (n_tokens, embd, n_seq_max)
        if n_tokens > self._max_batch_size:
            return
        if key not in self._pools:
            self._pools[key] = []
        if len(self._pools[key]) < self._max_pool_size:
            self._pools[key].append(batch)

    def get_stats(self) -> dict[str, object]:
        return {
            "pool_configs": {str(c): len(p) for c, p in self._pools.items()},
            "usage_count": {str(c): n for c, n in self._usage_count.items()},
            "total_pools": len(self._pools),
            "total_pooled_batches": sum(len(p) for p in self._pools.values()),
        }


_global_batch_pool = BatchMemoryPool()


def get_batch_pool_stats() -> dict[str, object]:
    return _global_batch_pool.get_stats()


def reset_batch_pool() -> None:
    global _global_batch_pool
    _global_batch_pool = BatchMemoryPool()


def return_batch_to_pool(batch: Any) -> None:
    """Return a batch to the global memory pool for reuse."""
    _global_batch_pool.return_batch(batch)


def get_pooled_batch(n_tokens: int, embd: int = 0, n_seq_max: int = 1) -> Any:
    """Get a batch from the global memory pool (or create a new one)."""
    return _global_batch_pool.get_batch(n_tokens, embd, n_seq_max)


# =============================================================================
# Download API (HuggingFace + Docker)
# =============================================================================


def _get_cache_dir() -> str:
    cache_dir = os.path.expanduser("~/.cache/llama.cpp")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _split_repo_tag(hf_repo_with_tag: str) -> tuple[str, str]:
    if ":" in hf_repo_with_tag:
        idx = hf_repo_with_tag.rfind(":")
        return hf_repo_with_tag[:idx], hf_repo_with_tag[idx + 1 :]
    return hf_repo_with_tag, "latest"


def _get_model_endpoint() -> str:
    return os.environ.get("LLAMA_CACHE_MODEL_ENDPOINT", "https://huggingface.co")


def _url_request(
    url: str,
    headers: Optional[dict[str, str]] = None,
    method: str = "GET",
    timeout: int = 30,
) -> tuple[int, dict[str, str], bytes]:
    req = Request(url, method=method)
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        resp = urlopen(req, timeout=timeout)
        return resp.status, dict(resp.headers), resp.read()
    except HTTPError as e:
        return e.code, dict(e.headers), e.read() if hasattr(e, "read") else b""
    except (URLError, OSError):
        return -1, {}, b""


def get_hf_file(hf_repo_with_tag: str, bearer_token: str = "", offline: bool = False) -> dict[str, str]:
    """Resolve a HuggingFace repo+tag to its GGUF / mmproj filenames."""
    repo, tag = _split_repo_tag(hf_repo_with_tag)
    endpoint = _get_model_endpoint()
    cache_dir = _get_cache_dir()
    safe_name = repo.replace("/", "=")
    manifest_path = os.path.join(cache_dir, f"manifest={safe_name}={tag}.json")

    manifest = None
    if not offline:
        url = f"{endpoint}/v2/{repo}/manifests/{tag}"
        headers = {"Accept": "application/json", "User-Agent": "llama-cpp"}
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
        try:
            status, _, body = _url_request(url, headers=headers)
            if 200 <= status < 400:
                manifest = json.loads(body)
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f)
        except Exception:
            pass

    if manifest is None and os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

    if manifest is None:
        suffix = " (offline mode)" if offline else ""
        raise RuntimeError(f"Failed to get manifest for {repo}:{tag}{suffix}")

    gguf_file = manifest.get("ggufFile", {}).get("rfilename", "") if manifest.get("ggufFile") else ""
    mmproj_file = manifest.get("mmprojFile", {}).get("rfilename", "") if manifest.get("mmprojFile") else ""
    return {"repo": repo, "gguf_file": gguf_file, "mmproj_file": mmproj_file}


def _download_file(
    url: str,
    dest_path: str,
    headers: Optional[dict[str, str]] = None,
    max_retries: int = 3,
) -> bool:
    """Download with ETag caching, resume support, and retry."""
    if headers is None:
        headers = {}

    etag_path = dest_path + ".etag"
    tmp_path = dest_path + ".downloadInProgress"

    head_status, head_headers, _ = _url_request(url, headers=headers, method="HEAD")
    if head_status < 200 or head_status >= 400:
        return False

    remote_etag = head_headers.get("ETag", "").strip('"')
    accepts_ranges = head_headers.get("Accept-Ranges", "").lower() == "bytes"

    if os.path.exists(dest_path) and os.path.exists(etag_path):
        with open(etag_path, "r") as f:
            cached_etag = f.read().strip()
        if cached_etag == remote_etag and remote_etag:
            return True

    for attempt in range(max_retries):
        try:
            resume_from = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0

            req = Request(url)
            for k, v in headers.items():
                req.add_header(k, v)
            if resume_from > 0 and accepts_ranges:
                req.add_header("Range", f"bytes={resume_from}-")

            resp = urlopen(req, timeout=300)
            status = resp.status
            if status not in (200, 206):
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                return False

            mode = "ab" if status == 206 else "wb"
            with open(tmp_path, mode) as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

            os.replace(tmp_path, dest_path)
            if remote_etag:
                with open(etag_path, "w") as f:
                    f.write(remote_etag)
            return True

        except (HTTPError, URLError, OSError):
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            else:
                return False

    return False


def download_model(
    model_path: Optional[str] = None,
    url: Optional[str] = None,
    hf_repo: Optional[str] = None,
    hf_file: Optional[str] = None,
    docker_repo: Optional[str] = None,
    bearer_token: str = "",
    offline: bool = False,
) -> bool:
    """Download a model from a URL, HuggingFace repo, or Docker registry."""
    # Auto-detect HF repo format in model_path.
    if model_path and hf_repo is None and url is None:
        s = str(model_path)
        is_hf = (
            "/" in s
            and s.count("/") == 1
            and not s.startswith(("http://", "https://", "file://", "/"))
            and "\\" not in s
            and not os.path.exists(s)
        )
        if is_hf:
            hf_repo = model_path
            model_path = None

    if docker_repo:
        try:
            return bool(resolve_docker_model(docker_repo))
        except RuntimeError:
            return False

    if hf_repo:
        info = get_hf_file(hf_repo, bearer_token, offline)
        repo = info["repo"]
        gguf_file = hf_file or info["gguf_file"]
        if not gguf_file:
            raise ValueError(f"Could not determine GGUF file for repo: {hf_repo}")
        endpoint = _get_model_endpoint()
        url = f"{endpoint}/{repo}/resolve/main/{gguf_file}"
        if not model_path:
            model_path = os.path.join(_get_cache_dir(), gguf_file)

    if not url:
        return False

    if offline:
        return model_path is not None and os.path.exists(model_path)

    if not model_path:
        return False

    headers: dict[str, str] = {}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    return _download_file(url, model_path, headers=headers)


def list_cached_models() -> list[dict[str, object]]:
    """List all models in the local llama.cpp cache."""
    cache_dir = _get_cache_dir()
    result = []
    for manifest_path in _glob.glob(os.path.join(cache_dir, "manifest=*.json")):
        basename = os.path.basename(manifest_path)
        name = basename[len("manifest=") : -len(".json")]
        parts = name.split("=")
        if len(parts) >= 3:
            user, model, tag = parts[0], parts[1], parts[2]
        elif len(parts) == 2:
            user, model, tag = parts[0], parts[1], "latest"
        else:
            continue
        result.append(
            {
                "manifest_path": manifest_path,
                "user": user,
                "model": model,
                "tag": tag,
                "size": 0,
            }
        )
    return result


def resolve_docker_model(docker_repo: str) -> str:
    """Resolve and download a model from Docker Hub."""
    if ":" in docker_repo:
        idx = docker_repo.rfind(":")
        repo, tag = docker_repo[:idx], docker_repo[idx + 1 :]
    else:
        repo, tag = docker_repo, "latest"
    if "/" not in repo:
        repo = f"ai/{repo}"

    auth_url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
    status, _, body = _url_request(auth_url)
    if status < 200 or status >= 400:
        raise RuntimeError(f"Docker auth failed for {repo}: HTTP {status}")
    try:
        docker_token = json.loads(body)["token"]
    except (json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(f"Docker auth failed for {repo}: {e}")

    headers = {
        "Authorization": f"Bearer {docker_token}",
        "Accept": ("application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json"),
    }
    manifest_url = f"https://registry-1.docker.io/v2/{repo}/manifests/{tag}"
    status, _, body = _url_request(manifest_url, headers=headers)
    if status < 200 or status >= 400:
        raise RuntimeError(f"Docker manifest fetch failed for {repo}:{tag}: HTTP {status}")
    try:
        manifest = json.loads(body)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Docker manifest parse failed for {repo}:{tag}: {e}")

    digest = None
    for layer in manifest.get("layers", []):
        media_type = layer.get("mediaType", "")
        ann_title = layer.get("annotations", {}).get("org.opencontainers.image.title", "")
        if "gguf" in media_type or "gguf" in ann_title:
            digest = layer.get("digest", "")
            break
    if not digest:
        raise RuntimeError(f"No GGUF layer found in Docker manifest for {repo}:{tag}")
    if not re.match(r"^sha256:[0-9a-fA-F]{64}$", digest):
        raise RuntimeError(f"Invalid digest format: {digest}")

    cache_dir = _get_cache_dir()
    safe_name = repo.replace("/", "_")
    dest_path = os.path.join(cache_dir, f"{safe_name}_{tag}.gguf")
    if os.path.exists(dest_path):
        return dest_path

    blob_url = f"https://registry-1.docker.io/v2/{repo}/blobs/{digest}"
    if not _download_file(blob_url, dest_path, headers={"Authorization": f"Bearer {docker_token}"}):
        raise RuntimeError(f"Failed to download Docker blob for {repo}:{tag}")
    return dest_path


# =============================================================================
# N-gram cache (pure Python — used by drafting code)
# =============================================================================

_NGRAM_MIN = 1
_NGRAM_MAX = 4
_NGRAM_STATIC = 2
_TOKEN_NULL = -1


_Ngram = tuple[int, ...]


def _make_ngram(tokens: list[int], size: int) -> _Ngram:
    result = list(tokens[:size])
    while len(result) < _NGRAM_MAX:
        result.append(_TOKEN_NULL)
    return tuple(result)


class NgramCache:
    """N-gram cache for accelerating generation with repetitive patterns."""

    def __init__(self) -> None:
        self._data: dict[_Ngram, dict[int, int]] = {}

    def update(
        self,
        tokens: list[int],
        ngram_min: int = 2,
        ngram_max: int = 4,
        nnew: Optional[int] = None,
        print_progress: bool = False,
    ) -> None:
        if nnew is None:
            nnew = len(tokens)
        ngram_min = max(_NGRAM_MIN, min(ngram_min, _NGRAM_MAX))
        ngram_max = max(_NGRAM_MIN, min(ngram_max, _NGRAM_MAX))
        n = len(tokens)
        data = self._data
        for ngram_size in range(ngram_min, ngram_max + 1):
            i_start = max(n - nnew, ngram_size)
            for i in range(i_start, n):
                ngram = _make_ngram(tokens[i - ngram_size : i], ngram_size)
                next_token = tokens[i]
                part = data.get(ngram)
                if part is None:
                    data[ngram] = {next_token: 1}
                else:
                    part[next_token] = part.get(next_token, 0) + 1

    def draft(
        self,
        inp: list[int],
        n_draft: int = 16,
        ngram_min: int = 2,
        ngram_max: int = 4,
        context_cache: Optional["NgramCache"] = None,
        dynamic_cache: Optional["NgramCache"] = None,
        static_cache: Optional["NgramCache"] = None,
    ) -> list[int]:
        ngram_min = max(_NGRAM_MIN, min(ngram_min, _NGRAM_MAX))
        ngram_max = max(_NGRAM_MIN, min(ngram_max, _NGRAM_MAX))
        ctx_data = (context_cache if context_cache is not None else self)._data
        dyn_data = (dynamic_cache if dynamic_cache is not None else NgramCache())._data
        sta_data = (static_cache if static_cache is not None else NgramCache())._data

        draft_tokens: list[int] = [inp[-1]] if len(inp) > 0 else [0]
        min_sample_lax = [2, 2, 1, 1]
        min_percent_lax = [66, 50, 50, 50]
        min_sample_strict = [4, 3, 2, 2]
        min_percent_strict = [75, 66, 66, 66]

        while len(draft_tokens) - 1 < n_draft:
            combined_seq = list(inp) + draft_tokens[1:]
            drafted = False

            # 1. Context cache (lax thresholds)
            for ngram_size in range(ngram_max, ngram_min - 1, -1):
                idx = ngram_size - 1
                if len(combined_seq) < ngram_size:
                    continue
                ngram = _make_ngram(combined_seq[-ngram_size:], ngram_size)
                part = ctx_data.get(ngram)
                if part is None:
                    continue
                best_token = _TOKEN_NULL
                best_score = -1
                sum_count = 0
                for tok, cnt in part.items():
                    sum_count += cnt
                    sta_part = (
                        sta_data.get(_make_ngram(combined_seq[-_NGRAM_STATIC:], _NGRAM_STATIC))
                        if len(combined_seq) >= _NGRAM_STATIC
                        else None
                    )
                    sta_cnt = sta_part.get(tok, 0) if sta_part else 0
                    score = cnt * max(1, sta_cnt)
                    if score > best_score:
                        best_score = score
                        best_token = tok
                if best_token == _TOKEN_NULL:
                    continue
                max_count = part.get(best_token, 0)
                if sum_count >= min_sample_lax[idx] and 100 * max_count >= min_percent_lax[idx] * sum_count:
                    draft_tokens.append(best_token)
                    drafted = True
                    break
            if drafted:
                continue

            # 2. Dynamic cache (strict thresholds)
            for ngram_size in range(ngram_max, ngram_min - 1, -1):
                idx = ngram_size - 1
                if len(combined_seq) < ngram_size:
                    continue
                ngram = _make_ngram(combined_seq[-ngram_size:], ngram_size)
                part = dyn_data.get(ngram)
                if part is None:
                    continue
                best_token = _TOKEN_NULL
                best_score = -1
                sum_count = 0
                for tok, cnt in part.items():
                    sum_count += cnt
                    sta_part = (
                        sta_data.get(_make_ngram(combined_seq[-_NGRAM_STATIC:], _NGRAM_STATIC))
                        if len(combined_seq) >= _NGRAM_STATIC
                        else None
                    )
                    sta_cnt = sta_part.get(tok, 0) if sta_part else 0
                    score = cnt * max(1, sta_cnt)
                    if score > best_score:
                        best_score = score
                        best_token = tok
                if best_token == _TOKEN_NULL:
                    continue
                max_count = part.get(best_token, 0)
                if sum_count >= min_sample_strict[idx] and 100 * max_count >= min_percent_strict[idx] * sum_count:
                    draft_tokens.append(best_token)
                    drafted = True
                    break
            if drafted:
                continue

            # 3. Static cache only (2-gram)
            if len(combined_seq) >= _NGRAM_STATIC:
                ngram = _make_ngram(combined_seq[-_NGRAM_STATIC:], _NGRAM_STATIC)
                part = sta_data.get(ngram)
                if part:
                    best_token = _TOKEN_NULL
                    best_count = -1
                    sum_count = 0
                    for tok, cnt in part.items():
                        sum_count += cnt
                        if cnt > best_count:
                            best_count = cnt
                            best_token = tok
                    if (
                        best_token != _TOKEN_NULL
                        and sum_count >= min_sample_lax[1]
                        and 100 * best_count >= 50 * sum_count
                    ):
                        draft_tokens.append(best_token)
                        continue
            break

        return draft_tokens[1:]  # drop seed token

    def save(self, filename: str) -> None:
        with open(filename, "wb") as f:
            for ngram, part in self._data.items():
                for t in ngram:
                    f.write(struct.pack("<i", t))
                f.write(struct.pack("<i", len(part)))
                for token, count in part.items():
                    f.write(struct.pack("<i", token))
                    f.write(struct.pack("<i", count))

    @staticmethod
    def load(filename: str) -> "NgramCache":
        cache = NgramCache()
        ngram_bytes = _NGRAM_MAX * 4
        with open(filename, "rb") as f:
            while True:
                data = f.read(ngram_bytes)
                if len(data) < ngram_bytes:
                    break
                tokens = struct.unpack("<" + "i" * _NGRAM_MAX, data)
                ngram = tuple(tokens)
                ntokens_data = f.read(4)
                if len(ntokens_data) < 4:
                    break
                ntokens = struct.unpack("<i", ntokens_data)[0]
                part = {}
                for _ in range(ntokens):
                    entry = f.read(8)
                    if len(entry) < 8:
                        break
                    tok, cnt = struct.unpack("<ii", entry)
                    part[tok] = cnt
                cache._data[ngram] = part
        return cache

    def merge(self, other: "NgramCache") -> None:
        if not isinstance(other, NgramCache):
            raise TypeError("Can only merge with another NgramCache")
        for ngram, part_add in other._data.items():
            part_target = self._data.get(ngram)
            if part_target is None:
                self._data[ngram] = dict(part_add)
            else:
                for token, count in part_add.items():
                    part_target[token] = part_target.get(token, 0) + count

    def __repr__(self) -> str:
        return f"<NgramCache at {hex(id(self))}>"
