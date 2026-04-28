# Inferna - an experiment

**WARNING**: inferna is an experimental recent automatic ai-driven translation and is buggy, not stable, and not working with different gpu architectures. Don't use it. Use [cyllama](https://pypi.org/project/cyllama) instead which is much more mature and tested.

---

Inferna is a Python library for running local AI models across text, speech, and image modalities. It wraps three established C++ inference engines behind a single high-level API:

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** - Text generation, chat, embeddings, and text-to-speech
- **[whisper.cpp](https://github.com/ggerganov/whisper.cpp)** - Speech-to-text transcription and translation
- **[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)** - Image and video generation

The bindings are built with [nanobind](https://github.com/wjakob/nanobind), and the package itself has no required Python dependencies.

**[Documentation](https://shakfu.github.io/inferna/)** | **[PyPI](https://pypi.org/project/inferna/)** | **[Changelog](CHANGELOG.md)**

Inferna is a nanobind-based rewrite of its sibling project, cyllama — a [cython](https://cython.org/) wrapper of the same .cpp ecosystem that remains actively maintained. The migration was motivated by the promise of nanobind's lower binding overhead and simpler C++ integration, and by the desire for a different development and packaging trajectory.

How inferna differs from cyllama:

| | inferna | cyllama |
|---|---|---|
| **Binding layer** | nanobind | Cython |
| **Wheel format** | stable ABI (`abi3`), one wheel per platform | per-Python-version wheels |
| **Minimum Python** | 3.12 | 3.10 |
| **Release cadence** | tracks major upstream releases of `llama.cpp` / `stable-diffusion.cpp` | tracks bleeding-edge `llama.cpp` / `stable-diffusion.cpp`, updated frequently |
| **Release lineage** | `0.1.0` corresponds to cyllama `0.2.14` | -- |

## Features

- High-level API -- `complete()`, `chat()`, `LLM` class for quick prototyping / text generation.

- Streaming -- token-by-token output with callbacks

- Batch processing -- process multiple prompts in parallel

- GPU acceleration -- Metal (macOS), CUDA (NVIDIA), ROCm (AMD), Vulkan (cross-platform), SYCL (Intel)

- Speculative decoding -- accelerate generation with draft models

- Agent framework -- ReActAgent, ConstrainedAgent, ContractAgent with tool calling

- RAG -- retrieval-augmented generation with local embeddings and [sqlite-vector](https://github.com/sqliteai/sqlite-vector)

- Speech recognition -- whisper.cpp transcription and translation

- Image/Video generation -- stable-diffusion.cpp handles image, image-edit and video models.

- OpenAI-compatible servers -- EmbeddedServer (C/Mongoose) and PythonServer with chat completions and embeddings endpoints

- Framework integrations -- OpenAI API client, LangChain LLM interface

## Installation

### From PyPI

```sh
pip install inferna
```

This installs the CPU backend for Linux and Windows. For macOS, the Metal backend is installed by default to take advantage of Apple Silicon.

### GPU-Accelerated Variants (DISABLED FOR NOW)

GPU variants are NOT YET available on PyPI as separate dynamically linked packages:

```sh
pip install inferna-cuda12   # NVIDIA GPU (CUDA 12.4)        -- Linux x86_64, Windows x86_64
pip install inferna-cuda13   # NVIDIA GPU (CUDA 13.1)        -- Windows x86_64
pip install inferna-rocm     # AMD GPU (ROCm 6.3)            -- Linux x86_64 (requires glibc >= 2.35)
pip install inferna-sycl     # Intel GPU (oneAPI SYCL 2025.3) -- Linux x86_64
pip install inferna-vulkan   # Cross-platform GPU (Vulkan)   -- Linux x86_64, Windows x86_64, macOS x86_64 (Intel)
```

All variants install the same `inferna` Python package -- only the compiled backend differs. Install one at a time (they replace each other). GPU variants require the corresponding driver/runtime installed on your system.

You can verify which backend is active after installation:

```sh
inferna info
```

You can also query the backend configuration at runtime:

```python
from inferna import _backend
print(_backend.cuda)   # True if built with CUDA
print(_backend.metal)  # True if built with Metal
```

### Build from source with a specific backend

A source install is a two-phase build: the third-party C++ libraries
(`llama.cpp`, `whisper.cpp`, `stable-diffusion.cpp`) must be built first
because they are intentionally excluded from the sdist (see
`pyproject.toml`'s `sdist.exclude` entry for `thirdparty/*/lib/`). The
nanobind extensions in `pip install` then link against those prebuilt
libraries.

```sh
# 1. Clone the repo and build the third-party deps in place.
git clone https://github.com/shakfu/inferna && cd inferna
GGML_CUDA=1 python scripts/manage.py build --all --deps-only --no-sd-examples

# 2. Build and install the wheel against the prebuilt deps.
GGML_CUDA=1 pip install . --no-build-isolation
```

`pip install inferna --no-binary inferna` (sdist-only, no clone) will
**not** work because the deps build step has no place to run. Use the
prebuilt wheels from PyPI, or follow the two-phase flow above. CI uses
the same `manage.py build --deps-only` step via cibuildwheel's
`before-build` hook.

## Command-Line Interface

inferna provides a unified CLI for all major functionality:

```bash
# Text generation
inferna gen -m models/llama.gguf -p "What is Python?" --stream
inferna gen -m models/llama.gguf -p "Write a haiku" --temperature 0.9 --json

# Chat (single-turn or interactive)
inferna chat -m models/llama.gguf -p "Explain gravity" -s "You are a physicist"
inferna chat -m models/llama.gguf                      # interactive mode
inferna chat -m models/llama.gguf -n 1024              # interactive, up to 1024 tokens per response
inferna chat -m models/llama.gguf --stats              # show session stats on exit

# Embeddings
inferna embed -m models/bge-small.gguf -t "hello world" -t "another text"
inferna embed -m models/bge-small.gguf --dim                        # print dimensions
inferna embed -m models/bge-small.gguf --similarity "cats" -f corpus.txt --threshold 0.5

# Other commands
inferna rag -m models/llama.gguf -e models/bge-small.gguf -d docs/ -p "How do I configure X?"
inferna rag -m models/llama.gguf -e models/bge-small.gguf -f file.md   # interactive mode
inferna rag -m models/llama.gguf -e models/bge-small.gguf -d docs/ --db docs.sqlite -p "..."  # index to persistent DB
inferna rag -m models/llama.gguf -e models/bge-small.gguf --db docs.sqlite -p "..."           # reuse existing DB, no re-indexing
inferna server -m models/llama.gguf --port 8080
inferna transcribe -m models/ggml-base.en.bin audio.wav
inferna tts -m models/tts.gguf -p "Hello world"
inferna sd txt2img --model models/sd.gguf --prompt "a sunset"
inferna info       # build and backend information
inferna memory -m models/llama.gguf  # GPU memory estimation
```

Run `inferna --help` or `inferna <command> --help` for full usage. See [CLI Cheatsheet](docs/cli-cheatsheet.md) for the complete reference.

## Quick Start

```python
from inferna import complete

# One line is all you need
response = complete(
    "Explain quantum computing in simple terms",
    model_path="models/llama.gguf",
    temperature=0.7,
    max_tokens=200
)
print(response)
```

## Key Features

### High-Level API

**High-Level API**:

```python
from inferna import complete, chat, LLM

# One-shot completion
response = complete("What is Python?", model_path="model.gguf")

# Multi-turn chat
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"}
]
response = chat(messages, model_path="model.gguf")

# Reusable LLM instance (faster for multiple prompts)
llm = LLM("model.gguf")
response1 = llm("Question 1")
response2 = llm("Question 2")  # Model stays loaded!
```

**Streaming Support** - Token-by-token output:

```python
for chunk in complete("Tell me a story", model_path="model.gguf", stream=True):
    print(chunk, end="", flush=True)
```

### Performance Features

**Batch Processing** - Process multiple prompts in parallel:

```python
from inferna import batch_generate

prompts = ["What is 2+2?", "What is 3+3?", "What is 4+4?"]
responses = batch_generate(prompts, model_path="model.gguf")
```

**Speculative Decoding** - Use a draft model to accelerate generation:

```python
from inferna.llama.llama_cpp import Speculative, SpeculativeParams

params = SpeculativeParams(n_max=16, p_min=0.75)
spec = Speculative(params, ctx_target)
draft_tokens = spec.draft(prompt_tokens, last_token)
```

**Memory Optimization** - GPU layer allocation:

```python
from inferna import estimate_gpu_layers

estimate = estimate_gpu_layers(
    model_path="model.gguf",
    available_vram_mb=8000
)
print(f"Recommended GPU layers: {estimate.n_gpu_layers}")
```

**N-gram Cache** - Reuse n-gram matches as draft tokens for repetitive text:

```python
from inferna.llama.llama_cpp import NgramCache

cache = NgramCache()
cache.update(tokens, ngram_min=2, ngram_max=4)
draft = cache.draft(input_tokens, n_draft=16)
```

**Response Caching** - Cache LLM responses for repeated prompts:

```python
from inferna import LLM

# Enable caching with 100 entries and 1 hour TTL
llm = LLM("model.gguf", cache_size=100, cache_ttl=3600, seed=42)

response1 = llm("What is Python?")  # Cache miss - generates response
response2 = llm("What is Python?")  # Cache hit - returns cached response instantly

# Check cache statistics
info = llm.cache_info()  # ResponseCacheInfo(hits=1, misses=1, maxsize=100, currsize=1, ttl=3600)

# Clear cache when needed
llm.cache_clear()
```

Note: Caching requires a fixed seed (not the default random sentinel) since random seeds produce non-deterministic output. Streaming responses are not cached.

### Framework Integrations

**OpenAI-Compatible API**:

```python
from inferna.integrations import OpenAIClient

client = OpenAIClient(model_path="model.gguf")

response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7
)
print(response.choices[0].message.content)
```

**LangChain Integration** - LangChain `LLM` interface:

```python
from inferna.integrations import InfernaLLM
from langchain.chains import LLMChain

llm = InfernaLLM(model_path="model.gguf", temperature=0.7)
chain = LLMChain(llm=llm, prompt=prompt_template)
result = chain.run(topic="AI")
```

### Agent Framework

Inferna includes a zero-dependency agent framework with three agent architectures:

**ReActAgent** - Reasoning + Acting agent with tool calling:

```python
from inferna import LLM
from inferna.agents import ReActAgent, tool
from simpleeval import simple_eval

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression safely."""
    return str(simple_eval(expression))

llm = LLM("model.gguf")
agent = ReActAgent(llm=llm, tools=[calculate])
result = agent.run("What is 25 * 4?")
print(result.answer)
```

**ConstrainedAgent** - Grammar-enforced tool calling (guarantees valid tool-call syntax):

```python
from inferna.agents import ConstrainedAgent

agent = ConstrainedAgent(llm=llm, tools=[calculate])
result = agent.run("Calculate 100 / 4")  # Guaranteed valid tool calls
```

**ContractAgent** - Contract-based agent with C++26-inspired pre/post conditions:

```python
from inferna.agents import ContractAgent, tool, pre, post, ContractPolicy

@tool
@pre(lambda args: args['x'] != 0, "cannot divide by zero")
@post(lambda r: r is not None, "result must not be None")
def divide(a: float, x: float) -> float:
    """Divide a by x."""
    return a / x

agent = ContractAgent(
    llm=llm,
    tools=[divide],
    policy=ContractPolicy.ENFORCE,
    task_precondition=lambda task: len(task) > 10,
    answer_postcondition=lambda ans: len(ans) > 0,
)
result = agent.run("What is 100 divided by 4?")
```

See [Agents Overview](docs/agents_overview.md) for detailed agent documentation.

### Speech Recognition

**Whisper Transcription** - Transcribe audio files with timestamps:

```python
from inferna.whisper import WhisperContext, WhisperFullParams
import numpy as np

# Load model and audio
ctx = WhisperContext("models/ggml-base.en.bin")
samples = load_audio_as_16khz_float32("audio.wav")  # Your audio loading function

# Transcribe
params = WhisperFullParams()
ctx.full(samples, params)

# Get results
for i in range(ctx.full_n_segments()):
    start = ctx.full_get_segment_t0(i) / 100.0
    end = ctx.full_get_segment_t1(i) / 100.0
    text = ctx.full_get_segment_text(i)
    print(f"[{start:.2f}s - {end:.2f}s] {text}")
```

See [Whisper docs](docs/whisper.md) for full documentation.

### Stable Diffusion

**Image Generation** - Generate images from text using stable-diffusion.cpp:

```python
from inferna.sd import text_to_image

# Simple text-to-image
image = text_to_image(
    model_path="models/sd_xl_turbo_1.0.q8_0.gguf",
    prompt="a photo of a cute cat",
    width=512,
    height=512,
    sample_steps=4,
    cfg_scale=1.0
)
image.save("output.png")
```

**Advanced Generation** - Full control with SDContext:

```python
from inferna.sd import SDContext, SDContextParams

params = SDContextParams()
params.model_path = "models/sd_xl_turbo_1.0.q8_0.gguf"
params.n_threads = 4

ctx = SDContext(params)
# sample_method / scheduler / eta / wtype default to auto-resolve
# sentinels (SD C-library defaults) -- pass explicitly only to override.
images = ctx.generate(
    prompt="a beautiful mountain landscape",
    negative_prompt="blurry, ugly",
    width=512,
    height=512,
)
```

**CLI Tool** - Command-line interface:

```bash
# Text to image
inferna sd txt2img \
    --model models/sd_xl_turbo_1.0.q8_0.gguf \
    --prompt "a beautiful sunset" \
    --output sunset.png

# Image to image
inferna sd img2img \
    --model models/sd-v1-5.gguf \
    --init-img input.png \
    --prompt "oil painting style" \
    --strength 0.7

# Show system info
inferna sd info
```

Supports SD 1.x/2.x, SDXL, SD3, FLUX, FLUX2, z-image-turbo, video generation (Wan/CogVideoX), LoRA, ControlNet, inpainting, and ESRGAN upscaling. See [Stable Diffusion docs](docs/stable_diffusion.md) for full documentation.

### RAG (Retrieval-Augmented Generation)

**CLI** - Query your documents from the command line:

```bash
# Single query against a directory of docs
inferna rag -m models/llama.gguf -e models/bge-small.gguf \
    -d docs/ -p "How do I configure X?" --stream

# Interactive mode with source display
inferna rag -m models/llama.gguf -e models/bge-small.gguf \
    -f guide.md -f faq.md --sources

# Persistent vector store: index once, reuse across runs
inferna rag -m models/llama.gguf -e models/bge-small.gguf \
    -d docs/ --db docs.sqlite -p "How do I configure X?"   # first run: indexes to docs.sqlite
inferna rag -m models/llama.gguf -e models/bge-small.gguf \
    --db docs.sqlite -p "Another question?"                # later runs: reuse index, no re-embedding
```

**Simple RAG** - Query your documents with LLMs:

```python
from inferna.rag import RAG

# Create RAG instance with embedding and generation models
rag = RAG(
    embedding_model="models/bge-small-en-v1.5-q8_0.gguf",
    generation_model="models/llama.gguf"
)

# Add documents
rag.add_texts([
    "Python is a high-level programming language.",
    "Machine learning is a subset of artificial intelligence.",
    "Neural networks are inspired by biological neurons."
])

# Query
response = rag.query("What is Python?")
print(response.text)
```

**Load Documents** - Support for multiple file formats:

```python
from inferna.rag import RAG, load_directory

rag = RAG(
    embedding_model="models/bge-small-en-v1.5-q8_0.gguf",
    generation_model="models/llama.gguf"
)

# Load all documents from a directory
documents = load_directory("docs/", glob="**/*.md")
rag.add_documents(documents)

response = rag.query("How do I configure the system?")
```

**Hybrid Search** - Combine vector and keyword search:

```python
from inferna.rag import RAG, HybridStore, Embedder

embedder = Embedder("models/bge-small-en-v1.5-q8_0.gguf")
store = HybridStore("knowledge.db", embedder)

store.add_texts(["Document content..."])

# Hybrid search with configurable weights
results = store.search("query", k=5, vector_weight=0.7, fts_weight=0.3)
```

**Embedding Cache** - Speed up repeated queries with LRU caching:

```python
from inferna.rag import Embedder

# Enable cache with 1000 entries
embedder = Embedder("models/bge-small-en-v1.5-q8_0.gguf", cache_size=1000)

embedder.embed("hello")  # Cache miss
embedder.embed("hello")  # Cache hit - instant return

info = embedder.cache_info()
print(f"Hits: {info.hits}, Misses: {info.misses}")
```

**Agent Integration** - Use RAG as an agent tool:

```python
from inferna import LLM
from inferna.agents import ReActAgent
from inferna.rag import RAG, create_rag_tool

rag = RAG(
    embedding_model="models/bge-small-en-v1.5-q8_0.gguf",
    generation_model="models/llama.gguf"
)
rag.add_texts(["Your knowledge base..."])

# Create a tool from the RAG instance
search_tool = create_rag_tool(rag)

llm = LLM("models/llama.gguf")
agent = ReActAgent(llm=llm, tools=[search_tool])
result = agent.run("Find information about X in the knowledge base")
```

Supports text chunking, multiple embedding pooling strategies, LRU caching for repeated queries, async operations, reranking, and SQLite-vector for persistent storage. See [RAG Overview](docs/rag_overview.md) for full documentation.

### Common Utilities

**GGUF File Manipulation** - Inspect and modify model files:

```python
from inferna.llama.llama_cpp import GGUFContext

ctx = GGUFContext.from_file("model.gguf")
metadata = ctx.get_all_metadata()
print(f"Model: {metadata['general.name']}")
```

**Structured Output** - JSON schema to grammar conversion (pure Python, no C++ dependency):

```python
from inferna.llama.llama_cpp import json_schema_to_grammar

schema = {"type": "object", "properties": {"name": {"type": "string"}}}
grammar = json_schema_to_grammar(schema)
```

**Huggingface Model Downloads**:

```python
from inferna.llama.llama_cpp import download_model, list_cached_models, get_hf_file

# Download from HuggingFace (saves to ~/.cache/llama.cpp/)
download_model("bartowski/Llama-3.2-1B-Instruct-GGUF:latest")

# Or with explicit parameters
download_model(hf_repo="bartowski/Llama-3.2-1B-Instruct-GGUF:latest")

# Download specific file to custom path
download_model(
    hf_repo="bartowski/Llama-3.2-1B-Instruct-GGUF",
    hf_file="Llama-3.2-1B-Instruct-Q8_0.gguf",
    model_path="./models/my_model.gguf"
)

# Get file info without downloading
info = get_hf_file("bartowski/Llama-3.2-1B-Instruct-GGUF:latest")
print(info)  # {'repo': '...', 'gguf_file': '...', 'mmproj_file': '...'}

# List cached models
models = list_cached_models()
```

## What's Inside

### Text Generation (llama.cpp)

- [x] **llama.cpp API** - nanobind wrapper

- [x] **High-Level API** - `LLM`, `complete`, `chat`

- [x] **Streaming Support** - Token-by-token generation with callbacks

- [x] **Batch Processing** - Parallel inference

- [x] **Multimodal** - LLAVA and vision-language models

- [x] **Speculative Decoding** - Draft-model-based generation

### Speech Recognition (whisper.cpp)

- [x] **whisper.cpp API** - nanobind wrapper

- [x] **High-Level API** - `transcribe()` function

- [x] **Multiple Formats** - WAV, MP3, FLAC, and more

- [x] **Language Detection** - Automatic or specified language

- [x] **Timestamps** - Word and segment-level timing

### Image & Video Generation (stable-diffusion.cpp)

- [x] **stable-diffusion.cpp API** - nanobind wrapper

- [x] **Text-to-Image** - SD 1.x/2.x, SDXL, SD3, FLUX, FLUX2, Z-Image

- [x] **Image-to-Image** - Transform existing images

- [x] **Inpainting** - Mask-based editing

- [x] **ControlNet** - Guided generation with edge/pose/depth

- [x] **Video Generation** - Wan, CogVideoX models

- [x] **Upscaling** - ESRGAN 4x upscaling

### Cross-Cutting Features

- [x] **GPU Acceleration** - Metal, CUDA, ROCm, Vulkan, SYCL backends

- [x] **Memory Optimization** - GPU layer allocation

- [x] **Agent Framework** - ReActAgent, ConstrainedAgent, ContractAgent

- [x] **Framework Integration** - OpenAI API, LangChain, FastAPI

## Status

**Build System**: scikit-build-core + CMake. See [pyproject.toml](pyproject.toml) for the current `inferna` version and [scripts/manage.py](scripts/manage.py) for pinned `llama.cpp` / `whisper.cpp` / `stable-diffusion.cpp` / `sqlite-vector` versions.

### Platform & GPU Availability

Pre-built wheels on PyPI:

| Package | Backend | Platform | Arch | Linking |
|---|---|---|---|---|
| `inferna` | CPU | Linux | x86_64 | static |
| `inferna` | CPU | Windows | x86_64 | static |
| `inferna` | Metal | macOS | arm64 (Apple Silicon) | static |
| `inferna` | Metal | macOS | x86_64 (Intel) | static |
| `inferna-cuda12` | CUDA | Linux | x86_64 | dynamic |
| `inferna-cuda12` | CUDA | Windows | x86_64 | dynamic |
| `inferna-cuda13` | CUDA | Windows | x86_64 | dynamic |
| `inferna-rocm` | ROCm | Linux | x86_64 | dynamic |
| `inferna-sycl` | Intel SYCL | Linux | x86_64 | dynamic |
| `inferna-vulkan` | Vulkan | Linux | x86_64 | dynamic |
| `inferna-vulkan` | Vulkan | Windows | x86_64 | dynamic |
| `inferna-vulkan` | Vulkan | macOS | x86_64 (Intel, MoltenVK) | dynamic |

Additional platforms (Windows SYCL / HIP, ARM64, Linux ROCm prebuilt, OpenVINO) are tracked in [TODO.md](TODO.md).

Build from source (any platform with a C++ toolchain):

| Backend | macOS | Linux | Windows |
|---|---|---|---|
| CPU | `make build-cpu` | `make build-cpu` | `make build-cpu` |
| Metal | `make build-metal` (default) | -- | -- |
| CUDA | -- | `make build-cuda` | `make build-cuda` |
| ROCm (HIP) | -- | `make build-hip` | -- |
| Vulkan | `make build-vulkan` | `make build-vulkan` | `make build-vulkan` |
| SYCL | -- | `make build-sycl` | -- |
| OpenCL | `make build-opencl` | `make build-opencl` | `make build-opencl` |

All source builds support both static (`make build-<backend>`) and dynamic (`make build-<backend>-dynamic`) linking.

## Building from Source

To build `inferna` from source:

1. A recent version of `python3` (currently testing on python 3.13)

2. Git clone the latest version of `inferna`:

    ```sh
    git clone https://github.com/shakfu/inferna.git
    cd inferna
    ```

3. We use [uv](https://github.com/astral-sh/uv) for package management:

    If you don't have it see the link above to install it, otherwise:

    ```sh
    uv sync
    ```

4. Type `make` in the terminal.

    This will:

    1. Download and build `llama.cpp`, `whisper.cpp` and `stable-diffusion.cpp`
    2. Install them into the `thirdparty` folder
    3. Build `inferna` using scikit-build-core + CMake

### Build Commands

```sh
# Full build (default: static linking, builds llama.cpp from source)
make              # Build dependencies + editable install

# Dynamic linking (downloads pre-built llama.cpp release)
make build-dynamic  # No source compilation needed for llama.cpp

# Build wheel for distribution
make wheel        # Creates wheel in dist/
make dist         # Creates sdist + wheel in dist/

# Backend-specific builds (static)
make build-cpu    # CPU only
make build-metal  # macOS Metal (default on macOS)
make build-cuda   # NVIDIA CUDA
make build-vulkan # Vulkan (cross-platform)
make build-hip    # AMD ROCm
make build-sycl   # Intel SYCL
make build-opencl # OpenCL

# Backend-specific builds (dynamic -- shared libs)
make build-cpu-dynamic
make build-cuda-dynamic
make build-vulkan-dynamic
make build-metal-dynamic
make build-hip-dynamic
make build-sycl-dynamic
make build-opencl-dynamic

# Backend-specific wheels (static and dynamic)
make wheel-cuda           # Static wheel
make wheel-cuda-dynamic   # Dynamic wheel with shared libs

# Clean and rebuild
make clean        # Remove build artifacts + dynamic libs
make reset        # Full reset including thirdparty and .venv
make remake       # Clean rebuild with tests

# Code quality
make lint         # Lint with ruff (auto-fix)
make format       # Format with ruff
make typecheck    # Type check with mypy
make qa           # Run all: lint, typecheck, format

# Memory leak detection
make leaks        # RSS-growth leak check (10 cycles, 20% threshold)

# Publishing
make check        # Validate wheels with twine
make publish      # Upload to PyPI
make publish-test # Upload to TestPyPI
```

### GPU Acceleration

By default, inferna builds with Metal support on macOS and CPU-only on Linux. To enable other GPU backends (CUDA, Vulkan, etc.):

```sh
# Static builds (all libs compiled in)
make build-cuda
make build-vulkan

# Dynamic builds (shared libs installed alongside extension)
make build-cuda-dynamic
make build-vulkan-dynamic

# Multiple backends
export GGML_CUDA=1 GGML_VULKAN=1
make build
```

See [Build Backends](docs/build_backends.md) for comprehensive backend build instructions.

### Multi-GPU Configuration

For systems with multiple GPUs, inferna provides full control over GPU selection and model splitting:

```python
from inferna import LLM, GenerationConfig

# Use a specific GPU (GPU index 1)
llm = LLM("model.gguf", main_gpu=1)

# Multi-GPU with layer splitting (default mode)
llm = LLM("model.gguf", split_mode=1, n_gpu_layers=-1)

# Multi-GPU with tensor parallelism (row splitting)
llm = LLM("model.gguf", split_mode=2, n_gpu_layers=-1)

# Custom tensor split: 30% GPU 0, 70% GPU 1
llm = LLM("model.gguf", tensor_split=[0.3, 0.7])

# Full configuration via GenerationConfig
config = GenerationConfig(
    main_gpu=0,
    split_mode=1,          # 0=NONE, 1=LAYER, 2=ROW
    tensor_split=[1, 2],   # 1/3 GPU0, 2/3 GPU1
    n_gpu_layers=-1
)
llm = LLM("model.gguf", config=config)
```

**Split Modes:**

- `0` (NONE): Single GPU only, uses `main_gpu`

- `1` (LAYER): Split layers and KV cache across GPUs (default)

- `2` (ROW): Tensor parallelism - split layers with row-wise distribution

## Testing

The `tests` directory in this repo provides extensive examples of using inferna.

However, as a first step, you should download a smallish llm in the `.gguf` model from [huggingface](https://huggingface.co/models?search=gguf). A good small model to start and which is assumed by tests is [Llama-3.2-1B-Instruct-Q8_0.gguf](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf). `inferna` expects models to be stored in a `models` folder in the cloned `inferna` directory. So to create the `models` directory if doesn't exist and download this model, you can just type:

```sh
make download
```

This basically just does:

```sh
cd inferna
mkdir models && cd models
wget https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf
```

Now you can test it using `llama-cli` or `llama-simple`:

```sh
bin/llama-cli -c 512 -n 32 -m models/Llama-3.2-1B-Instruct-Q8_0.gguf \
 -p "Is mathematics discovered or invented?"
```

Run the full pytest suite:

```sh
make test
```

You can also explore interactively:

```python
python3 -i scripts/start.py

>>> from inferna import complete
>>> response = complete("What is 2+2?", model_path="models/Llama-3.2-1B-Instruct-Q8_0.gguf")
>>> print(response)
```

## Documentation

Full documentation is available at [https://shakfu.github.io/inferna/](https://shakfu.github.io/inferna/) (built with MkDocs).

To serve docs locally: `make docs-serve`

- **[User Guide](docs/user_guide.md)** - Comprehensive guide covering all features

- **[CLI Cheatsheet](docs/cli-cheatsheet.md)** - Complete CLI reference for all commands

- **[API Reference](docs/api_reference.md)** - Complete API documentation

- **[RAG Overview](docs/rag_overview.md)** - Retrieval-augmented generation guide

- **[Cookbook](docs/cookbook.md)** - Practical recipes and patterns

- **[Changelog](CHANGELOG.md)** - Complete release history

- **Examples** - See `tests/examples/` for working code samples

## Contributing

Contributions are welcome! Please see the [User Guide](docs/user_guide.md) for development guidelines.

## License

This project wraps [llama.cpp](https://github.com/ggml-org/llama.cpp), [whisper.cpp](https://github.com/ggml-org/whisper.cpp), and [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) which all follow the MIT licensing terms, as does inferna.
