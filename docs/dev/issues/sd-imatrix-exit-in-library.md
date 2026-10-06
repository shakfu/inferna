imatrix collection calls exit(1) inside the library

For transparency: AI was used to help analyze and write up this issue. Found while binding the imatrix API in [inferna](https://github.com/shakfu/inferna).

**Environment:** stable-diffusion.cpp `master-898-2bb7294`; still present on `master` (`3f8527a`). Line numbers below are from `master`.

**Evidence:** from source reading.

## Summary

`IMatrixCollector::collect_imatrix` (`src/runtime/imatrix.cpp`) runs as the backend eval callback during generation. It calls `exit(1)` in four places (lines 107, 131, 144, 156):

```cpp
} else if (e.values.size() != (size_t)src1->ne[0]) {
    LOG_WARN("inconsistent size for %s (%d vs %d)\n", ...);
    exit(1);  // GGML_ABORT("fatal error");
}
```

The other three are the expert-layout variant of the same size check and two non-finite checks.

The size check fires in ordinary use, because the collector is process-wide and has no reset:

- collecting for SD 1.5 and then SDXL in one process (they share tensor names such as `model.diffusion_model.input_blocks.1.0.in_layers.2.weight` with different shapes);
- `load_imatrix` of one model's file, then collecting for another.

`exit(1)` in a library ends the host process without unwinding and without a return path. For a GUI or server it is indistinguishable from a crash.

## Suggested fix

Record the error in the collector, stop collecting, and report it later, for example as a `false` return from `save_imatrix`. A `reset_imatrix()` would let one process collect for a second model.

`save_imatrix` currently returns `void` and writes through `std::ofstream` without checking the stream (line 230), so a failed write is silent too. A `bool` return would report both.
