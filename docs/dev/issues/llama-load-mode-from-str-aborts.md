llama_load_mode_from_str / llama_load_mode_name abort on unknown input

For transparency: AI was used to help analyze and write up this issue. Found when a binding test that passes an invalid name crashed the test process in [inferna](https://github.com/shakfu/inferna).

**Environment:** llama.cpp `b11429` (`d812350`); unchanged on `master` (`6753a03`). Line numbers below are from `master`.

**Evidence:** observed. `llama_load_mode_from_str("bogus")` terminates the process (`Fatal Python error: Aborted` in the host).

## Summary

Both functions end in `GGML_ABORT` (`src/llama.cpp:65`, `:75`):

```cpp
enum llama_load_mode llama_load_mode_from_str(const char * str) {
    if (std::strcmp(str, "auto") == 0) { return LLAMA_LOAD_MODE_AUTO; }
    ...
    GGML_ABORT("unknown load mode: %s", str);
}
```

`llama_load_mode_from_str` parses user-supplied text, for example a CLI flag or a config value. A library function that kills the host process on bad input forces every caller to duplicate the name table to validate first.

## Suggested fix

Add an invalid value, or return `-1`, from `llama_load_mode_from_str`, and return `NULL` (or `"unknown"`, like `llama_ftype_name`) from `llama_load_mode_name`. Callers that want the abort can check the result.
