whisper_encode: a negative offset reads before the mel buffer

For transparency: AI was used to help analyze and write up this issue. Found while binding the low-level whisper.cpp API in [inferna](https://github.com/shakfu/inferna).

**Environment:** whisper.cpp `v1.9.4` (`927cfce`); still present on `master` (`60c0be6`). Line numbers below are from `master`.

**Evidence:** from source reading; not reproduced under a sanitizer.

## Summary

`whisper_encode(ctx, offset, n_threads)` passes `offset` to `whisper_encode_internal` as `mel_offset`. The mel copy clamps it only from above (`src/whisper.cpp:2437`):

```cpp
const int i0 = std::min(mel_offset,           mel_inp.n_len);
const int i1 = std::min(mel_offset + 2*n_ctx, mel_inp.n_len);

for (int j = 0; j < mel_inp.n_mel; ++j) {
    for (int i = i0; i < i1; ++i) {
        dst[j*2*n_ctx + (i - i0)] = mel_inp.data[j*mel_inp.n_len + i];
```

With `offset = -k`, row `j = 0` reads `mel_inp.data[-k .. -1]`, and every other row reads the tail of the previous row.

`whisper_lang_auto_detect_with_state` rejects a negative offset before encoding; `whisper_encode` and `whisper_encode_with_state` do not.

## Suggested fix

Return an error from `whisper_encode` / `whisper_encode_with_state` when `offset < 0`, or clamp `i0` at 0.
