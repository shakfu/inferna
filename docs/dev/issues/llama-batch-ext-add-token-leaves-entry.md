llama_batch_ext_add_token / add_embd keep the entry when they return -2

For transparency: AI was used to help analyze and write up this issue. Found while binding the extended batch API in [inferna](https://github.com/shakfu/inferna).

**Environment:** llama.cpp `b11429` (`d812350`); still present on `master` (`6753a03`). Line numbers below are from `master`.

**Evidence:** observed. After `llama_batch_ext_add_token(batch, 0, n_vocab + 1)` returns -2, the batch accepts only `n_batch - 1` further entries before returning -1.

## Summary

`llama_batch_ext_add_token` appends the entry first and validates the id afterwards (`src/llama-batch.cpp:1288`):

```cpp
int32_t idx = batch->add_token(seq_id);       // tokens.push_back(t)
if (idx < 0) {
    return idx;
}
if (!batch->set_token_id(idx, id)) {
    return -2;                                // entry stays in the batch
}
```

`llama_batch_ext_add_embd` (line 1299) has the same shape with `set_token_embd`. After -2 the batch holds an entry with `id = LLAMA_TOKEN_NULL`, no embedding and no position. The header (`include/llama.h`, above `llama_batch_ext_add`) documents -2 as an error code, so callers will not expect a side effect. There is no call to remove the last entry, so the only recovery is `llama_batch_ext_clear`.

## Suggested fix

Validate before appending, or pop the entry on failure:

```cpp
if (!batch->set_token_id(idx, id)) {
    batch->tokens.pop_back();
    return -2;
}
```

`add_embd` needs the same pop; for a new entry `set_token_embd` fails before touching `embd` or `n_embd`.
