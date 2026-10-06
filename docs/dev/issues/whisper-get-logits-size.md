whisper_get_logits: no way to get the array size, and only the last row is valid

For transparency: AI was used to help analyze and write up this issue. Found while binding the low-level whisper.cpp API in [inferna](https://github.com/shakfu/inferna).

**Environment:** whisper.cpp `v1.9.4` (`927cfce`); unchanged on `master` (`60c0be6`). Line numbers below are from `master`.

**Evidence:** from source reading.

## Summary

The header documents the result as `n_tokens` rows of `n_vocab` (`include/whisper.h`, above `whisper_get_logits`). Two things differ from that in practice:

1. **The caller cannot get the size.** `whisper_get_logits` returns `state->logits.data()` (line 4347). The row count is the `n_tokens` of the last `whisper_decode` call. Nothing returns it, and `whisper_full` and `whisper_lang_auto_detect` resize the same vector.
2. **Only the last row is written.** `whisper_batch_prep_legacy` sets `logits[i] = 1` only for the last token, and `whisper_decode_internal` copies only the flagged rows (line 3012):

   ```cpp
   logits_out.resize(n_tokens*n_vocab);
   for (int i = 0; i < n_tokens; i++) {
       if (batch.logits[i] == 0) {
           continue;
       }
   ```

   The other rows keep whatever the vector held before.

Also, the row stride is `hparams.n_vocab` (`whisper_model_n_vocab`), not `vocab.n_vocab` (`whisper_n_vocab`). The header does not say which one applies.

## Suggested fix

Either document that only the last row (`(n_tokens - 1) * whisper_model_n_vocab(ctx)`) is valid after `whisper_decode`, or add `whisper_n_logits(ctx)` / `_from_state` returning the row count and zero the rows that are not computed.
