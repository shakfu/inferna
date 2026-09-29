# sqlite-vector quantization: cosine and dot

`SqliteVectorStore.quantize()` builds an 8-bit index that `search()` then scans with `vector_quantize_scan`. Two defects affected the `cosine` and `dot` metrics. Both are fixed in `src/inferna/rag/store.py`. This file records the cause, the measurements and the options not taken.

Nothing in inferna calls `quantize()`, and the default metric is `cosine`. Only callers who quantize on their own are affected.

Source references are to sqlite-vector 1.1.2 (`build/sqlite-vector/src/sqlite-vector.c`).

## How the 8-bit index works

`vector_quantize()` computes one `scale` and one `offset` for the whole column, and stores each element as `round((x - offset) * scale)` (lines 612-625, 703-713).

| `qtype` | `offset` | `scale` |
|-|-|-|
| `UINT8` | column minimum | `255 / (max - min)` |
| `INT8` | 0 | `127 / max(abs(min), abs(max))` |

Lines 2006-2010. Without an explicit `qtype`, the extension picks `INT8` if the column contains a negative value and `UINT8` otherwise (lines 1995-1996).

The scan quantizes the query with the same `scale` and `offset` (lines 3392-3409). It then computes the distance on the two integer vectors and does not convert the result back to the data's units.

## Defect 1: `UINT8` for cosine and dot

`UINT8` subtracts `offset` from every element. A common translation cancels in L2 and L1 distances. It does not cancel in a dot product:

```text
(x - m) . (q - m) = x.q - m.x - m.q + m.m
```

The `m.x` term varies by row, so the ranking changes. Cosine is affected the same way. Upstream's README says so for cosine (lines 234-238: 33.8% recall with `UINT8` against 99.5% with `INT8`).

The harm grows with the offset. On data in `[0, 1)` the minimum is near 0 and recall does not change. On data in `[0.5, 1.5)`:

| metric | `UINT8` (extension default) | `INT8` |
|-|-|-|
| cosine | 0.84 | 0.98 |
| dot | 0.70 | 0.99 |
| l2 | 0.99 | 0.98 |

Recall@10 against the exact scan: 3000 vectors, dimension 64, 30 queries, seed 0.

Most embedding models emit signed values, which already get `INT8`. The defect affects models whose output is non-negative.

**Fix:** `quantize()` passes `qtype=INT8` for `cosine` and `dot`. L2, squared L2 and L1 keep the extension's choice. Test: `TestVectorStoreQuantization::test_quantized_recall_on_offset_data`.

## Defect 2: dot scores in quantized units

With `INT8`, both the stored row and the query are multiplied by `scale`. The scan's dot product is therefore about `scale^2` times the true value, and `search()` reported it as the score. In the regression test, one score was 20584.0 against an exact 26.13. That is a factor of 788, so `scale` = 28.1 and `abs_max` = 4.5, consistent with that data. In the benchmark below, the mean absolute score error was 3.5e4 for signed and 2.5e6 for offset non-negative data.

The ranking was correct. The score values were not, which broke:

- `threshold` in `SqliteVectorStore.search()`, and in `rag.py`, which passes it through
- scores shown to users (`advanced.py`) or returned by the pipeline (`pipeline.py`)
- comparing scores across queries or stores

Hybrid search is unaffected, as it merges by rank (`HybridStore._reciprocal_rank_fusion`).

Cosine scores are not affected: normalization cancels `scale`. Measured mean absolute score error after `INT8` quantization was 0.0009 or less.

**Fix:** for a quantized `dot` store, `search()` also selects each returned row's stored vector. It then recomputes the exact dot product with `math.sumprod`, applies `threshold`, and re-sorts. Test: `TestVectorStoreQuantization::test_quantized_dot_scores_match_exact`.

## Cost

20,000 vectors, dimension 384, 50 queries, k=10, AVX-512 backend:

| | exact scan | quantized `INT8` scan | rescoring (dot only) |
|-|-|-|-|
| time per query | about 5 ms | 0.2-0.5 ms | 0.17 ms |
| recall@10 | 1.00 | 0.94-0.97 | unchanged |

Rescoring grows linearly with k and dimension. It adds 50-80% to a quantized dot query, which stays about 10x faster than the exact scan.

## Limits of the current fix

- Rescoring re-scores the scan's k rows. It cannot recover a true top-k row the scan missed.
- `threshold` filters those k rows. A query can return fewer than k rows although more rows exceed the threshold. The exact scan behaves the same way.
- A query element beyond the column's `abs_max` is clamped to 127 when quantized. This adds error for queries outside the stored range.

## Options not taken

1. **Over-fetch, then rescore.** Scan `m * k` candidates, rescore them exactly, and return the top k. This is the usual two-stage design for quantized search. It would let the rescoring cost also raise recall, and would apply to cosine too. It needs a choice of `m` and a measurement of recall against latency.

2. **Recommend cosine for normalized embeddings.** For unit-length vectors, cosine and dot give the same ranking, and cosine scores survive quantization. Most `dot` users could switch and avoid defect 2.

3. **Fix defect 2 upstream.** For `INT8`, dividing the scan's dot distance by `scale^2` would restore the data's units. That is one multiply per row in C, against inferna's per-row decode in Python. For `UINT8`, the `offset` terms prevent an exact correction without per-row sums. If upstream fixes this, remove the rescoring branch from `search()`.

## Reproducing

The two tests above reproduce both defects. To see a defect, disable its fix:

- defect 1: remove the `qtype=INT8` line in `quantize()`
- defect 2: set `rescore = False` in `search()`

Before the fixes, `test_quantized_recall_on_offset_data` failed for cosine and dot, and `test_quantized_dot_scores_match_exact` failed with 20584.0 against 26.13.
