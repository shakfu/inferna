Metal: OPT_STEP_ADAMW / OPT_STEP_SGD gated on simdgroup reduction; training aborts on devices without it

For transparency: AI was used to help analyze and write up this issue. The abort was reproduced in [inferna](https://github.com/shakfu/inferna), which binds llama.cpp's training API (`llama_opt_init` / `llama_opt_epoch`).

**Environment:** GitHub Actions `macos-latest` (arm64). ggml-metal reports:

```text
GPU name:   MTL0 (Apple Paravirtual device)
GPU family: MTLGPUFamilyApple5  (1005)
simdgroup reduction   = false
simdgroup matrix mul. = false
```

llama.cpp `b11146` (`7fe450e`), `GGML_METAL=ON`, static. Full log: https://github.com/shakfu/inferna/actions/runs/36276500001

## Summary

`ggml_metal_device_supports_op` reports `OPT_STEP_ADAMW` and `OPT_STEP_SGD` as unsupported when `has_simdgroup_reduction` is false:

```cpp
// ggml/src/ggml-metal/ggml-metal-device.m
case GGML_OP_OPT_STEP_ADAMW:
case GGML_OP_OPT_STEP_SGD:
    return has_simdgroup_reduction;
```

Neither kernel uses simdgroup operations. Both are elementwise, one thread per element, with no threadgroup memory (`kernels/misc.metal`, `kernel_opt_step_adamw_f32` and `kernel_opt_step_sgd_f32`). The encoders dispatch a 1-D grid over `ggml_nelements(src0)` (`ggml_metal_op_opt_step_adamw` / `_sgd`).

The step updates the weight in place: the node is a view of the weight. When the weight lives in a Metal buffer, no other backend can run it. `ggml_backend_sched_backend_id_from_cur` then aborts:

```text
ggml-backend.cpp:941: pre-allocated tensor (adamw step for output_norm.weight) in a buffer (MTL0) that cannot run the operation (OPT_STEP_ADAMW)
```

The abort fires on the first `llama_opt_epoch` after a successful `llama_opt_init`, on a 2-layer F32 llama loaded with default model params (all layers on Metal).

## Origin of the gate

The gate was added in #16529 to make `test-opt` skip Metal on the CI VM, not for the kernel's sake ([comment](https://github.com/ggml-org/llama.cpp/pull/16529#issuecomment-3395077719)):

> It might be better to add `has_simdgroup_reduction` to `OPT_STEP_ADAMW` so that both ops would end up being not supported on the virtualized device of the CI. This way `GGML_OP_SUM` would keep the simdgroup reduction requirement, which we will need later for optimization.

`test-opt` skips a backend whose `supports_op` rejects the optimizer step (`tests/test-opt.cpp`, the `skip = !ggml_backend_supports_op(...)` probe). The failure it avoided was a different op: with static graphs, `loss_sum` is pre-allocated on Metal and needs `SUM`, which does use `simd_sum`. The gate therefore turns a `test-opt` failure into a skip, while real training on the same device still aborts. #16539 copied the gate to `OPT_STEP_SGD`.

## Impact

Any training run whose trainable weights sit in a Metal buffer on a device without simdgroup reduction terminates the process. `has_simdgroup_reduction` requires `MTLGPUFamilyApple7` or `MTLGPUFamilyMetal3`, so this covers:

* paravirtualized macOS VMs, including GitHub's hosted runners (confirmed);
* Mac GPUs outside the Metal 3 set, per Apple's [Metal feature set tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf) (not tested).

Apple Silicon Macs (Apple7+) are unaffected.

## Proposed fix

1. Drop the gate on the optimizer steps, and check what the kernels do assume:

   ```cpp
   case GGML_OP_OPT_STEP_ADAMW:
   case GGML_OP_OPT_STEP_SGD:
       return op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]);
   ```

2. Make `test-opt` skip on the ops it actually needs. For example, probe `SUM` alongside the optimizer step, or check every node of the built graph with `ggml_backend_supports_op`. This keeps `SUM`'s simdgroup requirement, as #16529 intended.

With (1), the llama.cpp training path should place the step on Metal. Its other simdgroup-gated ops (`ARGMAX`, `COUNT_EQUAL`) come from `ggml_opt`'s non-static graph, are not pre-allocated, and can fall back to CPU. This is inferred from `ggml-opt.cpp` (`static_graphs` is false when `ctx_compute` is null, as in `llama_context::opt_init`), not tested on the affected device.

## Not verified

* That (1) alone makes `llama_opt_epoch` succeed on the paravirtual device. Testing requires patching ggml on a hosted runner.
* That `llama-finetune -m <f32.gguf> -ngl 99` reproduces the abort on `macos-latest`. The reproduction above uses the same `llama_opt_*` calls through inferna's bindings.

## Related: AdamW momenta ignore parameter placement

This probably belongs in a separate issue. It reproduces on any Apple Silicon Mac, with no VM.

`ggml_opt_alloc` allocates `grad_m` / `grad_v` in `ctx_static` on `ggml_backend_sched_get_backend(sched, 0)` (`ggml-opt.cpp`). In a llama context that is the model's first GPU. The `OPT_STEP_ADAMW` node is a view of the weight, so the scheduler places it with the weight. When the weight is in a CPU buffer (`n_gpu_layers = 0`, or the CPU part of a partial offload), the step runs on CPU. There `m` / `v` are split inputs copied out of the GPU buffer, and the kernel's in-place writes to them are discarded. Every step starts from zero momenta. No error is raised.

Eval loss over 4 `llama_opt_epoch` calls, 2-layer F32 llama, Apple M-series:

| placement | AdamW (lr 1e-3) | SGD (lr 1e-2) |
|-|-|-|
| no Metal device (`GGML_METAL_DEVICES=0`) | 4.291, 3.727, 3.214, 2.837 | 3.970, 2.697, 2.218, 1.112 |
| all layers on Metal | 4.291, 3.727, 3.214, 2.837 | - |
| `n_gpu_layers = 0`, Metal present | 4.480, 4.176, 3.878, 3.582 | 3.970, 2.697, 2.218, 1.112 |

SGD, which keeps no per-parameter state, matches the reference. AdamW does not. Possible fixes: allocate each parameter's momenta in a buffer the step's backend can use, or make the scheduler place `OPT_STEP_ADAMW` by its state tensors as well as the weight.

Not verified: that the partial-offload case diverges the same way. It follows from the same placement rule.

## Workaround

* Weights on a device that lacks the step: use SGD with `n_gpu_layers = 0`, or hide the GPU (`GGML_METAL_DEVICES=0`) so the model has no GPU device.
* AdamW: keep every trained weight on the model's first GPU, or hide the GPU.

inferna's `opt_init` now enforces both rules and raises instead of aborting or silently degrading.
