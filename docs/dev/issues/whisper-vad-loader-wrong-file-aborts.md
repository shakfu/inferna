whisper_vad_init_from_file_with_params aborts the process when given a whisper model

For transparency: AI was used to help analyze and write up this issue. Found while binding the VAD API in [inferna](https://github.com/shakfu/inferna).

**Environment:** whisper.cpp `v1.9.4` (`927cfce`), Apple M1, Metal; loader code unchanged on `master` (`60c0be6`). Line numbers below are from `master`.

**Evidence:** observed.

```c
struct whisper_vad_context * vctx = whisper_vad_init_from_file_with_params(
    "ggml-base.en.bin", whisper_vad_default_context_params());
```

```text
whisper_vad_init_with_params: model type:
whisper_vad_init_with_params: n_encoder_layers = 0
...
ggml/src/ggml.c:1805: GGML_ASSERT(obj_new) failed
```

The call aborts instead of returning `NULL`.

## Summary

Whisper and VAD models share the legacy ggml magic, so the magic check passes. The loader then reads a length-prefixed model-type string (`src/whisper.cpp:4944`):

```cpp
int32_t str_len;
read_safe(loader, str_len);
std::vector<char> buffer(str_len + 1, 0);
```

In a whisper model that field is `n_vocab` (51864). The loader consumes 51864 bytes as the type string, reads the following hyperparameters from misaligned data, and later allocates from them (`new int32_t[hparams.n_encoder_layers]`, line 4974). With `ggml-base.en.bin` those counts happen to be 0 and tensor creation asserts. Other files can give a negative `str_len` (`std::length_error`) or large counts.

## Suggested fix

Validate the header before allocating: bound `str_len` (e.g. 1..64), check the type string (`silero-16k` today), and bound `n_encoder_layers` and the LSTM sizes. Return `NULL` on failure.
