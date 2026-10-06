whisper_full_parallel: n_processors < 1 and failed state allocation are not handled

For transparency: AI was used to help analyze and write up this issue. Found while binding `whisper_full_parallel` in [inferna](https://github.com/shakfu/inferna).

**Environment:** whisper.cpp `v1.9.4` (`927cfce`); still present on `master` (`60c0be6`). Line numbers below are from `master`.

**Evidence:** from source reading.

## Summary

`whisper_full_parallel` (`src/whisper.cpp:7969`) special-cases only `n_processors == 1`:

```cpp
std::vector<std::thread> workers(n_processors - 1);   // line 8004
for (int i = 0; i < n_processors - 1; ++i) {
    states.push_back(whisper_init_state(ctx));          // line 8007
```

1. `n_processors <= 0`: `n_processors - 1` converts to a huge `size_t`, so the vector constructor throws `std::length_error` (or `bad_alloc`) through the C API.
2. `whisper_init_state` returns `NULL` when allocation fails, for example when GPU memory runs out with many processors. The `NULL` state goes to `whisper_full_with_state` in a worker thread.

## Suggested fix

Return an error for `n_processors < 1`. Check each `whisper_init_state` result, and on failure free the states created so far and return an error.
