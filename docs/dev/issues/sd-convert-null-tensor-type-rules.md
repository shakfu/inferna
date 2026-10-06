convert / convert_with_components segfault when tensor_type_rules is NULL

For transparency: AI was used to help analyze and write up this issue. Found in [inferna](https://github.com/shakfu/inferna), whose `convert_model()` passed `NULL` when the caller gave no rules.

**Environment:** stable-diffusion.cpp `master-898-2bb7294`, Apple M1; still present on `master` (`3f8527a`). Line numbers below are from `master`.

**Evidence:** observed.

```c
convert("ae.safetensors", NULL, "out.gguf", SD_TYPE_F16, NULL, false);
```

```text
EXC_BAD_ACCESS (SIGSEGV) KERN_INVALID_ADDRESS at 0x0
  _platform_strlen
  convert_with_components
```

## Summary

`export_loaded_model` builds a `std::string` from the `const char*` (`src/convert.cpp:332`):

```cpp
TensorTypeRules type_rules = parse_tensor_type_rules(tensor_type_rules);
// parse_tensor_type_rules(const std::string&)  -- model_loader.h:14
```

Constructing `std::string` from `NULL` calls `strlen(NULL)`. The input path arguments accept `NULL`: `init_convert_path` skips a `NULL` path. And `validate_tensor_types` (`src/core/util.cpp`), which runs first, handles `NULL` rules with `SAFE_STR`, so `NULL` looks intended. `sd-cli` never hits the crash because it always passes `tensor_type_rules.c_str()`.

## Suggested fix

```cpp
TensorTypeRules type_rules = parse_tensor_type_rules(SAFE_STR(tensor_type_rules));
```
