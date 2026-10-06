whisper_decode: n_tokens and n_past are not checked before writing the batch (heap overflow)

For transparency: AI was used to help analyze and write up this issue. Found while binding the low-level whisper.cpp API in [inferna](https://github.com/shakfu/inferna).

**Environment:** whisper.cpp `v1.9.4` (`927cfce`); still present on `master` (`60c0be6`). Line numbers below are from `master`.

**Evidence:** from source reading; not reproduced under a sanitizer.

## Summary

`whisper_decode_with_state` writes `n_tokens` entries into `state->batch` before anything checks `n_tokens` (`src/whisper.cpp:4070`):

```cpp
int whisper_decode_with_state(..., const whisper_token * tokens, int n_tokens, int n_past, int n_threads) {
    whisper_batch_prep_legacy(state->batch, tokens, n_tokens, n_past, 0);
```

The batch holds `n_text_ctx` entries (`whisper_batch_init(ctx->model.hparams.n_text_ctx, ...)`, line 3567). `whisper_batch_prep_legacy` (line 517) loops to `n_tokens` and then writes `batch.logits[n_tokens - 1]`:

- `n_tokens > n_text_ctx` (448 for every released model) writes past `token`, `pos`, `n_seq_id`, `seq_id` and `logits`.
- `n_tokens == 0` writes `batch.logits[-1]`.
- `n_past + n_tokens > n_text_ctx` passes the batch, but positions index the 448-row positional embedding.

`whisper_kv_cache_find_slot` checks `n_tokens > n_ctx` (line 1037), but it runs inside `whisper_decode_internal`, after the batch was already written.

## Suggested fix

Return an error from `whisper_decode_with_state` when `n_tokens < 1`, `n_past < 0` or `n_past + n_tokens > n_text_ctx`, before `whisper_batch_prep_legacy`. A token-id check against `n_vocab` would also stop out-of-range ids from indexing the token embedding.
