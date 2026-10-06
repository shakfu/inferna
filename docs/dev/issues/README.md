# Upstream issue drafts

Drafts to file upstream, one issue per file. The first line is the issue title. Each was checked against the upstream default branch on 2026-10-06: llama.cpp `6753a03`, whisper.cpp `60c0be6`, stable-diffusion.cpp `3f8527a`. "Observed" means reproduced; the rest come from reading the source. inferna guards each one in its bindings except the imatrix `exit(1)`.

| File | Repo | Severity | Evidence |
|-|-|-|-|
| [whisper-decode-unchecked-n-tokens.md](whisper-decode-unchecked-n-tokens.md) | whisper.cpp | heap overflow | source |
| [sd-convert-null-tensor-type-rules.md](sd-convert-null-tensor-type-rules.md) | stable-diffusion.cpp | segfault | observed |
| [whisper-vad-loader-wrong-file-aborts.md](whisper-vad-loader-wrong-file-aborts.md) | whisper.cpp | abort | observed |
| [sd-imatrix-exit-in-library.md](sd-imatrix-exit-in-library.md) | stable-diffusion.cpp | process exit | source |
| [sd-load-imatrix-trusts-lengths.md](sd-load-imatrix-trusts-lengths.md) | stable-diffusion.cpp | out-of-bounds write | source |
| [whisper-encode-negative-offset.md](whisper-encode-negative-offset.md) | whisper.cpp | out-of-bounds read | source |
| [llama-batch-ext-set-pos-count.md](llama-batch-ext-set-pos-count.md) | llama.cpp | undocumented count, over-read | source |
| [llama-batch-ext-add-token-leaves-entry.md](llama-batch-ext-add-token-leaves-entry.md) | llama.cpp | wrong state | observed |
| [whisper-full-parallel-n-processors.md](whisper-full-parallel-n-processors.md) | whisper.cpp | exception, NULL state | source |
| [whisper-get-logits-size.md](whisper-get-logits-size.md) | whisper.cpp | API gap, stale data | source |
| [whisper-result-getters-unchecked-index.md](whisper-result-getters-unchecked-index.md) | whisper.cpp | out-of-bounds read | source |
| [whisper-lang-auto-detect-english-only.md](whisper-lang-auto-detect-english-only.md) | whisper.cpp | wrong result | source |
| [llama-load-mode-from-str-aborts.md](llama-load-mode-from-str-aborts.md) | llama.cpp | abort | observed |
| [ggml-metal-opt-step-simdgroup-gate.md](ggml-metal-opt-step-simdgroup-gate.md) | llama.cpp (ggml) | abort | observed |

Not drafted: stable-diffusion.cpp's `sd_*_name()` functions check only `value < COUNT`. A negative value indexes before the name table only where the enum's underlying type is signed (MSVC); GCC and Clang choose an unsigned type, so the read does not happen there.
