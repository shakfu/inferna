"""Public surface of the llama.cpp wrapper.

After the Cython → nanobind migration this is a pure-Python facade that
re-exports the native bindings (``_llama_native``), pure-Python helpers
(``_python_helpers``), and the speculative decoder (``_speculative``).

External callers continue to ``from inferna.llama.llama_cpp import ...`` the
same names they used before — class identity is preserved across the
migration.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Native bindings — classes, module-level functions, integer enum constants.
# ---------------------------------------------------------------------------

from . import _llama_native as _N

# Wrapped C++ classes
LlamaModel                  = _N.LlamaModel
LlamaContext                = _N.LlamaContext
LlamaSampler                = _N.LlamaSampler
LlamaBatch                  = _N.LlamaBatch
LlamaVocab                  = _N.LlamaVocab
LlamaAdapterLora            = _N.LlamaAdapterLora

# Parameter wrappers
LlamaModelParams            = _N.LlamaModelParams
LlamaContextParams          = _N.LlamaContextParams
LlamaModelQuantizeParams    = _N.LlamaModelQuantizeParams
LlamaSamplerChainParams     = _N.LlamaSamplerChainParams

# Data wrappers
LlamaTokenData              = _N.LlamaTokenData
LlamaLogitBias              = _N.LlamaLogitBias
LlamaModelKvOverride        = _N.LlamaModelKvOverride
LlamaModelTensorBuftOverride = _N.LlamaModelTensorBuftOverride
LlamaChatMessage            = _N.LlamaChatMessage

# GGUF + ggml wrappers
GGUFContext          = _N.GGUFContext
GgmlBackend          = _N.GgmlBackend
GgmlBackendDevice    = _N.GgmlBackendDevice
GgmlTensor           = _N.GgmlTensor
GgmlThreadPool       = _N.GgmlThreadPool
GgmlThreadPoolParams = _N.GgmlThreadPoolParams

# Multimodal (mtmd)
MtmdContext          = _N.MtmdContext
MtmdContextParams    = _N.MtmdContextParams
MtmdBitmap           = _N.MtmdBitmap
MtmdInputChunk       = _N.MtmdInputChunk
MtmdInputChunks      = _N.MtmdInputChunks
MtmdInputChunkType   = _N.MtmdInputChunkType
get_default_media_marker = _N.get_default_media_marker

# Module-level functions
set_log_callback         = _N.set_log_callback
disable_logging          = _N.disable_logging
chat_builtin_templates   = _N.chat_builtin_templates

ggml_version             = _N.ggml_version
ggml_commit              = _N.ggml_commit
ggml_time_us             = _N.ggml_time_us
ggml_backend_load_all    = _N.ggml_backend_load_all
ggml_backend_unload      = _N.ggml_backend_unload
ggml_backend_reg_count   = _N.ggml_backend_reg_count
ggml_backend_reg_names   = _N.ggml_backend_reg_names
ggml_backend_dev_count   = _N.ggml_backend_dev_count
ggml_backend_dev_info    = _N.ggml_backend_dev_info

llama_backend_init       = _N.llama_backend_init
llama_backend_free       = _N.llama_backend_free
llama_numa_init          = _N.llama_numa_init
llama_time_us            = _N.llama_time_us
llama_max_devices        = _N.llama_max_devices
llama_supports_mmap      = _N.llama_supports_mmap
llama_supports_mlock     = _N.llama_supports_mlock
llama_supports_gpu_offload = _N.llama_supports_gpu_offload
llama_supports_rpc       = _N.llama_supports_rpc
llama_attach_threadpool  = _N.llama_attach_threadpool
llama_detach_threadpool  = _N.llama_detach_threadpool
llama_batch_get_one      = _N.llama_batch_get_one
llama_flash_attn_type_name = _N.llama_flash_attn_type_name

# TTS helpers (registered by register_tts; the C++ source is in helpers/tts.cpp)
rgb2xterm256             = _N.rgb2xterm256
set_xterm256_foreground  = _N.set_xterm256_foreground
save_wav16               = _N.save_wav16
save_wav16_from_list     = _N.save_wav16_from_list
fill_hann_window         = _N.fill_hann_window
twiddle_factors          = _N.twiddle_factors
irfft                    = _N.irfft
fold                     = _N.fold
convert_less_than_thousand = _N.convert_less_than_thousand
number_to_words          = _N.number_to_words
replace_numbers_with_words = _N.replace_numbers_with_words
process_text             = _N.process_text


# ---------------------------------------------------------------------------
# Integer enum constants — the long flat list that callers reference as
# `cy.LLAMA_VOCAB_TYPE_BPE`, `cy.GGML_NUMA_STRATEGY_DISTRIBUTE`, etc.
# Imported at end-of-file so they don't shadow named symbols above.
# ---------------------------------------------------------------------------

# ggml constants
GGML_DEFAULT_N_THREADS  = _N.GGML_DEFAULT_N_THREADS
GGML_MAX_DIMS           = _N.GGML_MAX_DIMS
GGML_MAX_N_THREADS      = _N.GGML_MAX_N_THREADS
GGML_MAX_NAME           = _N.GGML_MAX_NAME
GGML_MAX_OP_PARAMS      = _N.GGML_MAX_OP_PARAMS
GGML_MAX_SRC            = _N.GGML_MAX_SRC

# ggml RoPE
GGML_ROPE_TYPE_NEOX     = _N.GGML_ROPE_TYPE_NEOX
GGML_ROPE_TYPE_MROPE    = _N.GGML_ROPE_TYPE_MROPE
GGML_ROPE_TYPE_VISION   = _N.GGML_ROPE_TYPE_VISION
GGML_ROPE_TYPE_IMROPE   = _N.GGML_ROPE_TYPE_IMROPE

# ggml NUMA
GGML_NUMA_STRATEGY_DISABLED   = _N.GGML_NUMA_STRATEGY_DISABLED
GGML_NUMA_STRATEGY_DISTRIBUTE = _N.GGML_NUMA_STRATEGY_DISTRIBUTE
GGML_NUMA_STRATEGY_ISOLATE    = _N.GGML_NUMA_STRATEGY_ISOLATE
GGML_NUMA_STRATEGY_NUMACTL    = _N.GGML_NUMA_STRATEGY_NUMACTL
GGML_NUMA_STRATEGY_MIRROR     = _N.GGML_NUMA_STRATEGY_MIRROR

# ggml log levels
GGML_LOG_LEVEL_NONE  = _N.GGML_LOG_LEVEL_NONE
GGML_LOG_LEVEL_INFO  = _N.GGML_LOG_LEVEL_INFO
GGML_LOG_LEVEL_WARN  = _N.GGML_LOG_LEVEL_WARN
GGML_LOG_LEVEL_ERROR = _N.GGML_LOG_LEVEL_ERROR
GGML_LOG_LEVEL_DEBUG = _N.GGML_LOG_LEVEL_DEBUG
GGML_LOG_LEVEL_CONT  = _N.GGML_LOG_LEVEL_CONT

# ggml status
GGML_STATUS_ALLOC_FAILED = _N.GGML_STATUS_ALLOC_FAILED
GGML_STATUS_FAILED       = _N.GGML_STATUS_FAILED
GGML_STATUS_SUCCESS      = _N.GGML_STATUS_SUCCESS
GGML_STATUS_ABORTED      = _N.GGML_STATUS_ABORTED

# ggml dtypes
GGML_TYPE_F32     = _N.GGML_TYPE_F32
GGML_TYPE_F16     = _N.GGML_TYPE_F16
GGML_TYPE_Q4_0    = _N.GGML_TYPE_Q4_0
GGML_TYPE_Q4_1    = _N.GGML_TYPE_Q4_1
GGML_TYPE_Q5_0    = _N.GGML_TYPE_Q5_0
GGML_TYPE_Q5_1    = _N.GGML_TYPE_Q5_1
GGML_TYPE_Q8_0    = _N.GGML_TYPE_Q8_0
GGML_TYPE_Q8_1    = _N.GGML_TYPE_Q8_1
GGML_TYPE_Q2_K    = _N.GGML_TYPE_Q2_K
GGML_TYPE_Q3_K    = _N.GGML_TYPE_Q3_K
GGML_TYPE_Q4_K    = _N.GGML_TYPE_Q4_K
GGML_TYPE_Q5_K    = _N.GGML_TYPE_Q5_K
GGML_TYPE_Q6_K    = _N.GGML_TYPE_Q6_K
GGML_TYPE_Q8_K    = _N.GGML_TYPE_Q8_K
GGML_TYPE_IQ2_XXS = _N.GGML_TYPE_IQ2_XXS
GGML_TYPE_IQ2_XS  = _N.GGML_TYPE_IQ2_XS
GGML_TYPE_IQ3_XXS = _N.GGML_TYPE_IQ3_XXS
GGML_TYPE_IQ1_S   = _N.GGML_TYPE_IQ1_S
GGML_TYPE_IQ4_NL  = _N.GGML_TYPE_IQ4_NL
GGML_TYPE_IQ3_S   = _N.GGML_TYPE_IQ3_S
GGML_TYPE_IQ2_S   = _N.GGML_TYPE_IQ2_S
GGML_TYPE_IQ4_XS  = _N.GGML_TYPE_IQ4_XS
GGML_TYPE_I8      = _N.GGML_TYPE_I8
GGML_TYPE_I16     = _N.GGML_TYPE_I16
GGML_TYPE_I32     = _N.GGML_TYPE_I32
GGML_TYPE_I64     = _N.GGML_TYPE_I64
GGML_TYPE_F64     = _N.GGML_TYPE_F64
GGML_TYPE_IQ1_M   = _N.GGML_TYPE_IQ1_M
GGML_TYPE_BF16    = _N.GGML_TYPE_BF16
GGML_TYPE_TQ1_0   = _N.GGML_TYPE_TQ1_0
GGML_TYPE_TQ2_0   = _N.GGML_TYPE_TQ2_0
GGML_TYPE_MXFP4   = _N.GGML_TYPE_MXFP4
GGML_TYPE_COUNT   = _N.GGML_TYPE_COUNT

# ggml precision / op / triangular / scale / opt / backend-device-type
GGML_PREC_DEFAULT = _N.GGML_PREC_DEFAULT
GGML_PREC_F32     = _N.GGML_PREC_F32
GGML_OP_NONE      = _N.GGML_OP_NONE
GGML_TRI_TYPE_UPPER_DIAG = _N.GGML_TRI_TYPE_UPPER_DIAG
GGML_TRI_TYPE_UPPER      = _N.GGML_TRI_TYPE_UPPER
GGML_TRI_TYPE_LOWER_DIAG = _N.GGML_TRI_TYPE_LOWER_DIAG
GGML_TRI_TYPE_LOWER      = _N.GGML_TRI_TYPE_LOWER
GGML_SCALE_MODE_NEAREST  = _N.GGML_SCALE_MODE_NEAREST
GGML_SCALE_MODE_BILINEAR = _N.GGML_SCALE_MODE_BILINEAR
GGML_SCALE_MODE_BICUBIC  = _N.GGML_SCALE_MODE_BICUBIC
GGML_OPT_BUILD_TYPE_FORWARD = _N.GGML_OPT_BUILD_TYPE_FORWARD
GGML_OPT_BUILD_TYPE_GRAD    = _N.GGML_OPT_BUILD_TYPE_GRAD
GGML_OPT_BUILD_TYPE_OPT     = _N.GGML_OPT_BUILD_TYPE_OPT
GGML_BACKEND_DEVICE_TYPE_CPU   = _N.GGML_BACKEND_DEVICE_TYPE_CPU
GGML_BACKEND_DEVICE_TYPE_GPU   = _N.GGML_BACKEND_DEVICE_TYPE_GPU
GGML_BACKEND_DEVICE_TYPE_IGPU  = _N.GGML_BACKEND_DEVICE_TYPE_IGPU
GGML_BACKEND_DEVICE_TYPE_ACCEL = _N.GGML_BACKEND_DEVICE_TYPE_ACCEL
GGML_BACKEND_DEVICE_TYPE_META  = _N.GGML_BACKEND_DEVICE_TYPE_META

# llama vocab type
LLAMA_VOCAB_TYPE_NONE    = _N.LLAMA_VOCAB_TYPE_NONE
LLAMA_VOCAB_TYPE_SPM     = _N.LLAMA_VOCAB_TYPE_SPM
LLAMA_VOCAB_TYPE_BPE     = _N.LLAMA_VOCAB_TYPE_BPE
LLAMA_VOCAB_TYPE_WPM     = _N.LLAMA_VOCAB_TYPE_WPM
LLAMA_VOCAB_TYPE_UGM     = _N.LLAMA_VOCAB_TYPE_UGM
LLAMA_VOCAB_TYPE_RWKV    = _N.LLAMA_VOCAB_TYPE_RWKV
LLAMA_VOCAB_TYPE_PLAMO2  = _N.LLAMA_VOCAB_TYPE_PLAMO2

# llama RoPE type
LLAMA_ROPE_TYPE_NONE   = _N.LLAMA_ROPE_TYPE_NONE
LLAMA_ROPE_TYPE_NORM   = _N.LLAMA_ROPE_TYPE_NORM
LLAMA_ROPE_TYPE_NEOX   = _N.LLAMA_ROPE_TYPE_NEOX
LLAMA_ROPE_TYPE_MROPE  = _N.LLAMA_ROPE_TYPE_MROPE
LLAMA_ROPE_TYPE_VISION = _N.LLAMA_ROPE_TYPE_VISION
LLAMA_ROPE_TYPE_IMROPE = _N.LLAMA_ROPE_TYPE_IMROPE

# llama token type
LLAMA_TOKEN_TYPE_UNDEFINED    = _N.LLAMA_TOKEN_TYPE_UNDEFINED
LLAMA_TOKEN_TYPE_NORMAL       = _N.LLAMA_TOKEN_TYPE_NORMAL
LLAMA_TOKEN_TYPE_UNKNOWN      = _N.LLAMA_TOKEN_TYPE_UNKNOWN
LLAMA_TOKEN_TYPE_CONTROL      = _N.LLAMA_TOKEN_TYPE_CONTROL
LLAMA_TOKEN_TYPE_USER_DEFINED = _N.LLAMA_TOKEN_TYPE_USER_DEFINED
LLAMA_TOKEN_TYPE_UNUSED       = _N.LLAMA_TOKEN_TYPE_UNUSED
LLAMA_TOKEN_TYPE_BYTE         = _N.LLAMA_TOKEN_TYPE_BYTE

# llama token attr (bit flags)
LLAMA_TOKEN_ATTR_UNDEFINED    = _N.LLAMA_TOKEN_ATTR_UNDEFINED
LLAMA_TOKEN_ATTR_UNKNOWN      = _N.LLAMA_TOKEN_ATTR_UNKNOWN
LLAMA_TOKEN_ATTR_UNUSED       = _N.LLAMA_TOKEN_ATTR_UNUSED
LLAMA_TOKEN_ATTR_NORMAL       = _N.LLAMA_TOKEN_ATTR_NORMAL
LLAMA_TOKEN_ATTR_CONTROL      = _N.LLAMA_TOKEN_ATTR_CONTROL
LLAMA_TOKEN_ATTR_USER_DEFINED = _N.LLAMA_TOKEN_ATTR_USER_DEFINED
LLAMA_TOKEN_ATTR_BYTE         = _N.LLAMA_TOKEN_ATTR_BYTE
LLAMA_TOKEN_ATTR_NORMALIZED   = _N.LLAMA_TOKEN_ATTR_NORMALIZED
LLAMA_TOKEN_ATTR_LSTRIP       = _N.LLAMA_TOKEN_ATTR_LSTRIP
LLAMA_TOKEN_ATTR_RSTRIP       = _N.LLAMA_TOKEN_ATTR_RSTRIP
LLAMA_TOKEN_ATTR_SINGLE_WORD  = _N.LLAMA_TOKEN_ATTR_SINGLE_WORD

# llama ftype
LLAMA_FTYPE_ALL_F32          = _N.LLAMA_FTYPE_ALL_F32
LLAMA_FTYPE_MOSTLY_F16       = _N.LLAMA_FTYPE_MOSTLY_F16
LLAMA_FTYPE_MOSTLY_Q4_0      = _N.LLAMA_FTYPE_MOSTLY_Q4_0
LLAMA_FTYPE_MOSTLY_Q4_1      = _N.LLAMA_FTYPE_MOSTLY_Q4_1
LLAMA_FTYPE_MOSTLY_Q8_0      = _N.LLAMA_FTYPE_MOSTLY_Q8_0
LLAMA_FTYPE_MOSTLY_Q5_0      = _N.LLAMA_FTYPE_MOSTLY_Q5_0
LLAMA_FTYPE_MOSTLY_Q5_1      = _N.LLAMA_FTYPE_MOSTLY_Q5_1
LLAMA_FTYPE_MOSTLY_Q2_K      = _N.LLAMA_FTYPE_MOSTLY_Q2_K
LLAMA_FTYPE_MOSTLY_Q3_K_S    = _N.LLAMA_FTYPE_MOSTLY_Q3_K_S
LLAMA_FTYPE_MOSTLY_Q3_K_M    = _N.LLAMA_FTYPE_MOSTLY_Q3_K_M
LLAMA_FTYPE_MOSTLY_Q3_K_L    = _N.LLAMA_FTYPE_MOSTLY_Q3_K_L
LLAMA_FTYPE_MOSTLY_Q4_K_S    = _N.LLAMA_FTYPE_MOSTLY_Q4_K_S
LLAMA_FTYPE_MOSTLY_Q4_K_M    = _N.LLAMA_FTYPE_MOSTLY_Q4_K_M
LLAMA_FTYPE_MOSTLY_Q5_K_S    = _N.LLAMA_FTYPE_MOSTLY_Q5_K_S
LLAMA_FTYPE_MOSTLY_Q5_K_M    = _N.LLAMA_FTYPE_MOSTLY_Q5_K_M
LLAMA_FTYPE_MOSTLY_Q6_K      = _N.LLAMA_FTYPE_MOSTLY_Q6_K
LLAMA_FTYPE_MOSTLY_IQ2_XXS   = _N.LLAMA_FTYPE_MOSTLY_IQ2_XXS
LLAMA_FTYPE_MOSTLY_IQ2_XS    = _N.LLAMA_FTYPE_MOSTLY_IQ2_XS
LLAMA_FTYPE_MOSTLY_Q2_K_S    = _N.LLAMA_FTYPE_MOSTLY_Q2_K_S
LLAMA_FTYPE_MOSTLY_IQ3_XS    = _N.LLAMA_FTYPE_MOSTLY_IQ3_XS
LLAMA_FTYPE_MOSTLY_IQ3_XXS   = _N.LLAMA_FTYPE_MOSTLY_IQ3_XXS
LLAMA_FTYPE_MOSTLY_IQ1_S     = _N.LLAMA_FTYPE_MOSTLY_IQ1_S
LLAMA_FTYPE_MOSTLY_IQ4_NL    = _N.LLAMA_FTYPE_MOSTLY_IQ4_NL
LLAMA_FTYPE_MOSTLY_IQ3_S     = _N.LLAMA_FTYPE_MOSTLY_IQ3_S
LLAMA_FTYPE_MOSTLY_IQ3_M     = _N.LLAMA_FTYPE_MOSTLY_IQ3_M
LLAMA_FTYPE_MOSTLY_IQ2_S     = _N.LLAMA_FTYPE_MOSTLY_IQ2_S
LLAMA_FTYPE_MOSTLY_IQ2_M     = _N.LLAMA_FTYPE_MOSTLY_IQ2_M
LLAMA_FTYPE_MOSTLY_IQ4_XS    = _N.LLAMA_FTYPE_MOSTLY_IQ4_XS
LLAMA_FTYPE_MOSTLY_IQ1_M     = _N.LLAMA_FTYPE_MOSTLY_IQ1_M
LLAMA_FTYPE_MOSTLY_BF16      = _N.LLAMA_FTYPE_MOSTLY_BF16
LLAMA_FTYPE_MOSTLY_TQ1_0     = _N.LLAMA_FTYPE_MOSTLY_TQ1_0
LLAMA_FTYPE_MOSTLY_TQ2_0     = _N.LLAMA_FTYPE_MOSTLY_TQ2_0
LLAMA_FTYPE_MOSTLY_MXFP4_MOE = _N.LLAMA_FTYPE_MOSTLY_MXFP4_MOE
LLAMA_FTYPE_MOSTLY_NVFP4     = _N.LLAMA_FTYPE_MOSTLY_NVFP4
LLAMA_FTYPE_MOSTLY_Q1_0      = _N.LLAMA_FTYPE_MOSTLY_Q1_0
LLAMA_FTYPE_GUESSED          = _N.LLAMA_FTYPE_GUESSED

# llama RoPE scaling
LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = _N.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED
LLAMA_ROPE_SCALING_TYPE_NONE        = _N.LLAMA_ROPE_SCALING_TYPE_NONE
LLAMA_ROPE_SCALING_TYPE_LINEAR      = _N.LLAMA_ROPE_SCALING_TYPE_LINEAR
LLAMA_ROPE_SCALING_TYPE_YARN        = _N.LLAMA_ROPE_SCALING_TYPE_YARN
LLAMA_ROPE_SCALING_TYPE_LONGROPE    = _N.LLAMA_ROPE_SCALING_TYPE_LONGROPE
LLAMA_ROPE_SCALING_TYPE_MAX_VALUE   = _N.LLAMA_ROPE_SCALING_TYPE_MAX_VALUE

# llama pooling
LLAMA_POOLING_TYPE_UNSPECIFIED = _N.LLAMA_POOLING_TYPE_UNSPECIFIED
LLAMA_POOLING_TYPE_NONE        = _N.LLAMA_POOLING_TYPE_NONE
LLAMA_POOLING_TYPE_MEAN        = _N.LLAMA_POOLING_TYPE_MEAN
LLAMA_POOLING_TYPE_CLS         = _N.LLAMA_POOLING_TYPE_CLS
LLAMA_POOLING_TYPE_LAST        = _N.LLAMA_POOLING_TYPE_LAST
LLAMA_POOLING_TYPE_RANK        = _N.LLAMA_POOLING_TYPE_RANK

# llama attention / flash-attn / split
LLAMA_ATTENTION_TYPE_UNSPECIFIED = _N.LLAMA_ATTENTION_TYPE_UNSPECIFIED
LLAMA_ATTENTION_TYPE_CAUSAL      = _N.LLAMA_ATTENTION_TYPE_CAUSAL
LLAMA_ATTENTION_TYPE_NON_CAUSAL  = _N.LLAMA_ATTENTION_TYPE_NON_CAUSAL
LLAMA_FLASH_ATTN_TYPE_AUTO     = _N.LLAMA_FLASH_ATTN_TYPE_AUTO
LLAMA_FLASH_ATTN_TYPE_DISABLED = _N.LLAMA_FLASH_ATTN_TYPE_DISABLED
LLAMA_FLASH_ATTN_TYPE_ENABLED  = _N.LLAMA_FLASH_ATTN_TYPE_ENABLED
LLAMA_SPLIT_MODE_NONE   = _N.LLAMA_SPLIT_MODE_NONE
LLAMA_SPLIT_MODE_LAYER  = _N.LLAMA_SPLIT_MODE_LAYER
LLAMA_SPLIT_MODE_ROW    = _N.LLAMA_SPLIT_MODE_ROW
LLAMA_SPLIT_MODE_TENSOR = _N.LLAMA_SPLIT_MODE_TENSOR

# llama token sentinel
LLAMA_TOKEN_NULL = _N.LLAMA_TOKEN_NULL


# ---------------------------------------------------------------------------
# Pure-Python helpers (memory pools, downloads, n-gram cache).
# ---------------------------------------------------------------------------

from ._python_helpers import (
    # memory pools
    TokenMemoryPool,
    BatchMemoryPool,
    get_token_pool_stats,
    reset_token_pool,
    get_batch_pool_stats,
    reset_batch_pool,
    return_batch_to_pool,
    get_pooled_batch,
    # download API
    download_model,
    get_hf_file,
    list_cached_models,
    resolve_docker_model,
    # n-gram cache
    NgramCache,
)


# ---------------------------------------------------------------------------
# Speculative decoding (rewritten on top of native bindings).
# ---------------------------------------------------------------------------

from ._speculative import SpeculativeParams, Speculative


# ---------------------------------------------------------------------------
# JSON Schema -> GBNF grammar (pure Python; no native dep).
# ---------------------------------------------------------------------------

from inferna.utils.json_schema_to_grammar import json_schema_to_grammar
