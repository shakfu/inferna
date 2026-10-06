llama_batch_ext_set_pos: document how many positions it reads

For transparency: AI was used to help analyze and write up this issue. Found while binding the extended batch API in [inferna](https://github.com/shakfu/inferna) and [cyllama](https://github.com/shakfu/cyllama).

**Environment:** llama.cpp `b11429` (`d812350`); unchanged on `master` (`6753a03`). Line numbers below are from `master`.

**Evidence:** from source reading.

## Summary

`llama_batch_ext_set_pos(batch, idx, const llama_pos * pos)` reads 1 position for a token entry and `n_pos_per_embd` positions for an embedding entry (`src/llama-batch.cpp:1236`):

```cpp
size_t n_pos = t.id != LLAMA_TOKEN_NULL ? 1 : n_pos_per_embd;
for (size_t i = 0; i < n_pos; ++i) {
    t.pos[i] = pos_in[i];
}
```

`n_pos_per_embd` is `hparams.n_pos_per_embd()` (`src/llama-hparams.cpp:285`): 4 when `llama_model_rope_type()` is `LLAMA_ROPE_TYPE_MROPE` or `LLAMA_ROPE_TYPE_IMROPE`, otherwise 1. The header says only "Embedding tokens must have multiple positions per token". A caller who does not know the rope-type rule passes too few positions, and the read runs past the end of `pos`.

## Suggested fix

State the count in the header comment ("4 (`GGML_MROPE_SECTIONS`) when `llama_model_rope_type()` is MROPE or IMROPE, else 1"). Optionally take the count as a parameter (`..., const llama_pos * pos, int32_t n_pos`) and return false when it is too small.
