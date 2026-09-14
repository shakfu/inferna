# Inferna - a multimodal Python inference library

Inferna runs local text, speech, and image models from Python. It wraps three C++ inference engines behind one high-level API:

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** - text generation, chat, embeddings, and text-to-speech

- **[whisper.cpp](https://github.com/ggml-org/whisper.cpp)** - speech-to-text transcription and translation

- **[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)** - image and video generation

The bindings use [nanobind](https://github.com/wjakob/nanobind). The package has no required Python dependencies.

**[Documentation](https://shakfu.github.io/inferna/)** | **[PyPI](https://pypi.org/project/inferna/)** | **[Changelog](CHANGELOG.md)**

## Relation to cyllama

Inferna is a nanobind rewrite of [cyllama](https://github.com/shakfu/cyllama), a [Cython](https://cython.org/) wrapper of the same `.cpp` stack. Both projects are maintained. They pin the same upstream revisions, share their build scaffolding, and publish the same wheel format (see [Python version and wheels](#python-version-and-wheels)).

They differ in:

| | inferna | cyllama |
|---|---|---|
| **Binding layer** | nanobind | Cython |
| **Release lineage** | `0.1.0` corresponds to cyllama `0.2.14` | -- |
| **Embedded web UI** | opt-in chat UI, a rebrand of llama.cpp's [llama-server webui](https://github.com/ggml-org/llama.cpp/tree/master/tools/server/webui); `inferna server -w` or `ServerConfig(serve_webui=True)` | built-in servers are API-only |
| **macOS Intel** | not supported | `cyllama`, `cyllama-vulkan` |
| **Windows GPU variants** | CI can build them; not released | `cyllama-cuda12`, `cyllama-vulkan` |

Inferna publishes fewer wheels than cyllama, and a release may not include every variant. Check [PyPI](https://pypi.org/project/inferna/#files) for the files in a given release.

## Features

- High-level API -- `complete()`, `chat()`, and the `LLM` class

- Streaming -- token-by-token output with callbacks

- Batch processing -- multiple prompts in parallel

- GPU acceleration -- Metal (macOS), CUDA (NVIDIA), ROCm (AMD), Vulkan (cross-platform), SYCL (Intel)

- Speculative decoding -- draft-model-accelerated generation

- Agent framework -- ReActAgent, ConstrainedAgent, ContractAgent with tool calling; multi-agent composition (`agent_as_tool`, `TieredAgentTeam`); JSON-Schema constraints on tool args via `Annotated[]` markers; per-tool timeouts and coercion

- RAG -- retrieval-augmented generation with local embeddings and [sqlite-vector](https://github.com/sqliteai/sqlite-vector)

- Speech recognition -- whisper.cpp transcription and translation

- Image and video generation -- stable-diffusion.cpp image, image-edit, and video models

- OpenAI-compatible servers -- EmbeddedServer (C/Mongoose) with SSE streaming and the opt-in web UI, plus a pure-Python PythonServer; both expose chat-completions and embeddings endpoints

- Framework integrations -- OpenAI-style client, LangChain LLM interface

## Installation

### From PyPI

```sh
pip install inferna
```

This installs the CPU backend on Linux and Windows, and the Metal backend on macOS (Apple Silicon).

### GPU variants

GPU variants are separate PyPI packages, dynamically linked, Linux x86_64 only:

```sh
pip install inferna-cuda12   # NVIDIA GPU (CUDA 12.4)
pip install inferna-rocm     # AMD GPU (ROCm 6.3, requires glibc >= 2.35)
pip install inferna-sycl     # Intel GPU (oneAPI SYCL 2025.3)
pip install inferna-vulkan   # Vulkan
```

Every variant installs the same `inferna` Python package; only the compiled backend differs. Install one at a time, since each replaces the others. Each requires the matching driver or runtime on the host.

Check the active backend:

```sh
inferna info
```

Or at runtime:

```python
from inferna._internal import build_config

print(build_config.backend_enabled("cuda"))   # True if built with CUDA
print(build_config.backend_enabled("metal"))  # True if built with Metal
print(build_config.backend())                 # full per-backend config dict
```

### Python version and wheels

Inferna publishes only **abi3** wheels, built against the CPython stable ABI. Each wheel is tagged `cp312-abi3-<platform>` and imports on Python 3.12, 3.13, 3.14, and later. One wheel per platform replaces one per Python version.

Python 3.10 and 3.11 are not supported. cyllama moved to the same format in `0.3.0`; its `0.2.18` release is the last with per-version wheels for 3.10-3.14.

### Optional integrations

Inferna has no hard dependencies beyond its compiled core. Features that use third-party libraries import them lazily, so install only what you use.

**PDF parsing** (`inferna.rag.PDFLoader`) supports four backends:

| Backend    | Install                          | Strengths                                      | Capabilities                                  |
|------------|----------------------------------|------------------------------------------------|-----------------------------------------------|
| `pypdf`    | `pip install pypdf`              | Pure-Python, lightweight, per-page text        | `per_page`                                    |
| `pymupdf`  | `pip install pymupdf`            | Fast, per-page text, table/image awareness     | `per_page`, `tables`, `images`                |
| `pdfminer` | `pip install pdfminer.six`       | Pure-Python, layout-aware extraction           | `layout`                                      |
| `docling`  | `pip install docling`            | Highest quality; OCR, tables, layout, markdown | `ocr`, `tables`, `images`, `layout`, `markdown` (heavy; pulls in torch + CV stack) |

`PDFLoader(backend="auto")`, the default, picks the first installed backend in table order. Select one with `PDFLoader(backend="docling")`, or filter by capability with `PDFLoader(require={"ocr"})`. `inferna.rag.available_pdf_backends()` and `pdf_backend_info(name)` report what is installed.

**Other optional integrations**:

| Feature             | Install                          |
|---------------------|----------------------------------|
| Qdrant vector store | `pip install qdrant-client`      |

### Build from source with a specific backend

A source install has two phases. The sdist excludes the prebuilt `llama.cpp`, `whisper.cpp`, and `stable-diffusion.cpp` libraries (`sdist.exclude` in `pyproject.toml`), so build them first, then build the extension against them:

```sh
# 1. Clone and build the third-party deps in place.
git clone https://github.com/shakfu/inferna && cd inferna
GGML_CUDA=1 python scripts/manage.py build --all --deps-only --no-sd-examples

# 2. Build and install against the prebuilt deps.
GGML_CUDA=1 pip install . --no-build-isolation
```

`pip install inferna --no-binary inferna` does **not** work: an sdist-only install has no step that builds the deps. CI runs the same `manage.py build --deps-only` step in cibuildwheel's `before-all` hook.

A plain source build produces a version-specific extension. See [Building from source](#building-from-source) for the abi3 wheel target.

## Command-line interface

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

# RAG
inferna rag -m models/llama.gguf -e models/bge-small.gguf -d docs/ -p "How do I configure X?"
inferna rag -m models/llama.gguf -e models/bge-small.gguf -f file.md   # interactive mode
inferna rag -m models/llama.gguf -e models/bge-small.gguf -d docs/ --db docs.sqlite -p "..."  # index to persistent DB
inferna rag -m models/llama.gguf -e models/bge-small.gguf --db docs.sqlite -p "..."           # reuse DB, no re-indexing

# Servers
inferna server -m models/llama.gguf --port 8080         # OpenAI-compatible API only
inferna server -m models/llama.gguf --port 8080 -w      # API + browser chat UI at http://127.0.0.1:8080/

# Speech, image, diagnostics
inferna transcribe -m models/ggml-base.en.bin -f audio.wav
inferna tts -m models/tts.gguf -mv models/vocoder.gguf -p "Hello world"
inferna sd txt2img --model models/sd.gguf --prompt "a sunset"
inferna info                          # build and backend information
inferna memory models/llama.gguf      # GPU memory estimation
```

Run `inferna --help` or `inferna <command> --help` for usage. The [CLI Cheatsheet](docs/cli-cheatsheet.md) is the full reference.

## Quick start

```python
from inferna import complete

response = complete(
    "Explain quantum computing in simple terms",
    model_path="models/llama.gguf",
    temperature=0.7,
    max_tokens=200,
)
print(response)
```

## Usage

### High-level API

```python
from inferna import complete, chat, LLM

# One-shot completion
response = complete("What is Python?", model_path="model.gguf")

# Multi-turn chat
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"},
]
response = chat(messages, model_path="model.gguf")

# Reusable instance: the model stays loaded between calls
llm = LLM("model.gguf")
response1 = llm("Question 1")
response2 = llm("Question 2")
```

**Streaming**:

```python
for chunk in complete("Tell me a story", model_path="model.gguf", stream=True):
    print(chunk, end="", flush=True)
```

### Performance

**Batch processing**:

```python
from inferna import batch_generate

prompts = ["What is 2+2?", "What is 3+3?", "What is 4+4?"]
responses = batch_generate(prompts, model_path="model.gguf")
```

**Speculative decoding** with a draft model:

```python
from inferna.llama.llama_cpp import Speculative, SpeculativeParams

params = SpeculativeParams(n_max=16, p_min=0.75)
spec = Speculative(params, ctx_target)
draft_tokens = spec.draft(prompt_tokens, last_token)
```

**GPU layer estimation**:

```python
from inferna import estimate_gpu_layers

estimate = estimate_gpu_layers(model_path="model.gguf", available_vram_mb=8000)
print(f"Recommended GPU layers: {estimate.n_gpu_layers}")
```

**N-gram cache** -- reuse n-gram matches as draft tokens for repetitive text:

```python
from inferna.llama.llama_cpp import NgramCache

cache = NgramCache()
cache.update(tokens, ngram_min=2, ngram_max=4)
draft = cache.draft(input_tokens, n_draft=16)
```

**Response cache**:

```python
from inferna import LLM

# 100 entries, 1 hour TTL
llm = LLM("model.gguf", cache_size=100, cache_ttl=3600, seed=42)

response1 = llm("What is Python?")  # miss: generates
response2 = llm("What is Python?")  # hit: returns cached response

info = llm.cache_info()  # ResponseCacheInfo(hits=1, misses=1, maxsize=100, currsize=1, ttl=3600)
llm.cache_clear()
```

Caching requires a fixed seed; the default random seed makes output non-deterministic. Streaming responses are not cached.

### Framework integrations

**OpenAI-style client**:

```python
from inferna.integrations import OpenAIClient

client = OpenAIClient(model_path="model.gguf")

response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
)
print(response.choices[0].message.content)
```

**LangChain**:

```python
from inferna.integrations import InfernaLLM
from langchain.chains import LLMChain

llm = InfernaLLM(model_path="model.gguf", temperature=0.7)
chain = LLMChain(llm=llm, prompt=prompt_template)
result = chain.run(topic="AI")
```

### Agent framework

Three agent architectures, with no extra dependencies.

**ReActAgent** -- reasoning and acting with tool calls:

```python
from inferna import LLM
from inferna.agents import ReActAgent
from inferna.agents.tools import calculator  # safe arithmetic, no eval

llm = LLM("model.gguf")
agent = ReActAgent(llm=llm, tools=[calculator])
result = agent.run("What is 25 * 4?")
print(result.answer)
```

**ConstrainedAgent** -- a grammar enforces valid tool-call syntax:

```python
from inferna.agents import ConstrainedAgent

agent = ConstrainedAgent(llm=llm, tools=[calculate])
result = agent.run("Calculate 100 / 4")
```

**ContractAgent** -- pre- and post-conditions modeled on C++26 contracts:

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
    task_preconditions=[lambda task: len(task) > 10],
    answer_postconditions=[lambda ans: len(ans) > 0],
)
result = agent.run("What is 100 divided by 4?")
```

**Schema constraints via `Annotated[]`** -- `Ge`, `Le`, `MultipleOf`, `MinLen`, `MaxLen`, and `Pattern` attach JSON-Schema bounds to type hints. Dispatch enforces them before the tool runs:

```python
from typing import Annotated, Literal
from inferna.agents import tool, Ge, Le, Pattern

@tool
def fetch(
    table: Annotated[str, Pattern(r"^[a-z_]+$")],
    limit: Annotated[int, Ge(1), Le(1000)],
    mode: Literal["preview", "full"] = "preview",
) -> list[dict]: ...
```

**Multi-agent composition** -- `agent_as_tool` wraps an agent as a tool. `TieredAgentTeam` pairs a supervisor with worker agents, which can use smaller models:

```python
from inferna.agents import agent_as_tool, AgentRole, TieredAgentTeam

team = TieredAgentTeam(
    supervisor=ReActAgent(llm=LLM("models/strong.gguf"), tools=[]),
    workers=[
        AgentRole("researcher", researcher, "Find facts."),
        AgentRole("coder", coder, "Modify code."),
    ],
)
result = team.run("Refactor X using technique Y.")
```

See [Agents Overview](docs/agents_overview.md), and [Contract Recipes](docs/agents/contracts.md) for nine patterns contrasting schema constraints with contracts.

### Speech recognition

```python
from inferna.whisper import WhisperContext, WhisperFullParams

ctx = WhisperContext("models/ggml-base.en.bin")
samples = load_audio_as_16khz_float32("audio.wav")  # your audio loader

params = WhisperFullParams()
ctx.full(samples, params)

for i in range(ctx.full_n_segments()):
    start = ctx.full_get_segment_t0(i) / 100.0
    end = ctx.full_get_segment_t1(i) / 100.0
    text = ctx.full_get_segment_text(i)
    print(f"[{start:.2f}s - {end:.2f}s] {text}")
```

See [Whisper docs](docs/whisper.md).

### Stable Diffusion

**Text to image**:

```python
from inferna.sd import text_to_image

image = text_to_image(
    model_path="models/sd_xl_turbo_1.0.q8_0.gguf",
    prompt="a photo of a cute cat",
    width=512,
    height=512,
    sample_steps=4,
    cfg_scale=1.0,
)
image.save("output.png")
```

**SDContext** for full control:

```python
from inferna.sd import SDContext, SDContextParams

params = SDContextParams()
params.model_path = "models/sd_xl_turbo_1.0.q8_0.gguf"
params.n_threads = 4

ctx = SDContext(params)
# sample_method, scheduler, eta, and wtype default to the SD library's
# auto-resolved values; pass them only to override.
images = ctx.generate(
    prompt="a beautiful mountain landscape",
    negative_prompt="blurry, ugly",
    width=512,
    height=512,
)
```

**CLI**:

```bash
inferna sd txt2img \
    --model models/sd_xl_turbo_1.0.q8_0.gguf \
    --prompt "a beautiful sunset" \
    --output sunset.png

inferna sd img2img \
    --model models/sd-v1-5.gguf \
    --init-img input.png \
    --prompt "oil painting style" \
    --strength 0.7

inferna sd info
```

Supports SD 1.x/2.x, SDXL, SD3, FLUX, FLUX2, Z-Image, video (Wan, CogVideoX), LoRA, ControlNet, inpainting, and ESRGAN upscaling. See [Stable Diffusion docs](docs/stable_diffusion.md).

### RAG

**CLI**:

```bash
# Single query against a directory
inferna rag -m models/llama.gguf -e models/bge-small.gguf \
    -d docs/ -p "How do I configure X?" --stream

# Interactive, showing sources
inferna rag -m models/llama.gguf -e models/bge-small.gguf \
    -f guide.md -f faq.md --sources

# Persistent store: the first run indexes, later runs reuse the index
inferna rag -m models/llama.gguf -e models/bge-small.gguf \
    -d docs/ --db docs.sqlite -p "How do I configure X?"
inferna rag -m models/llama.gguf -e models/bge-small.gguf \
    --db docs.sqlite -p "Another question?"
```

**Python**:

```python
from inferna.rag import RAG

rag = RAG(
    embedding_model="models/bge-small-en-v1.5-q8_0.gguf",
    generation_model="models/llama.gguf",
)

rag.add_texts([
    "Python is a high-level programming language.",
    "Machine learning is a subset of artificial intelligence.",
    "Neural networks are inspired by biological neurons.",
])

response = rag.query("What is Python?")
print(response.text)
```

**Loading documents**:

```python
from inferna.rag import RAG, load_directory

rag = RAG(
    embedding_model="models/bge-small-en-v1.5-q8_0.gguf",
    generation_model="models/llama.gguf",
)
documents = load_directory("docs/", glob="**/*.md")
rag.add_documents(documents)

response = rag.query("How do I configure the system?")
```

**Hybrid search** -- weighted vector and full-text search:

```python
from inferna.rag import HybridStore, Embedder

embedder = Embedder("models/bge-small-en-v1.5-q8_0.gguf")
store = HybridStore("knowledge.db", embedder)
store.add_texts(["Document content..."])

results = store.search("query", k=5, vector_weight=0.7, fts_weight=0.3)
```

**Embedding cache** (LRU):

```python
from inferna.rag import Embedder

embedder = Embedder("models/bge-small-en-v1.5-q8_0.gguf", cache_size=1000)

embedder.embed("hello")  # miss
embedder.embed("hello")  # hit

info = embedder.cache_info()
print(f"Hits: {info.hits}, Misses: {info.misses}")
```

**RAG as an agent tool**:

```python
from inferna import LLM
from inferna.agents import ReActAgent
from inferna.rag import RAG, create_rag_tool

rag = RAG(
    embedding_model="models/bge-small-en-v1.5-q8_0.gguf",
    generation_model="models/llama.gguf",
)
rag.add_texts(["Your knowledge base..."])

llm = LLM("models/llama.gguf")
agent = ReActAgent(llm=llm, tools=[create_rag_tool(rag)])
result = agent.run("Find information about X in the knowledge base")
```

RAG also supports chunking, several embedding pooling strategies, async operations, and reranking. See [RAG Overview](docs/rag_overview.md).

### Utilities

**GGUF inspection and editing**:

```python
from inferna.llama.llama_cpp import GGUFContext

ctx = GGUFContext.from_file("model.gguf")
metadata = ctx.get_all_metadata()
print(f"Model: {metadata['general.name']}")
```

**JSON schema to grammar** (pure Python):

```python
from inferna.llama.llama_cpp import json_schema_to_grammar

schema = {"type": "object", "properties": {"name": {"type": "string"}}}
grammar = json_schema_to_grammar(schema)
```

**Hugging Face downloads**:

```python
from inferna.llama.llama_cpp import download_model, list_cached_models, get_hf_file

# Saves to ~/.cache/llama.cpp/
download_model("bartowski/Llama-3.2-1B-Instruct-GGUF:latest")

# A specific file to a custom path
download_model(
    hf_repo="bartowski/Llama-3.2-1B-Instruct-GGUF",
    hf_file="Llama-3.2-1B-Instruct-Q8_0.gguf",
    model_path="./models/my_model.gguf",
)

# File info without downloading
info = get_hf_file("bartowski/Llama-3.2-1B-Instruct-GGUF:latest")
print(info)  # {'repo': '...', 'gguf_file': '...', 'mmproj_file': '...'}

models = list_cached_models()
```

### Multi-GPU

```python
from inferna import LLM, GenerationConfig

llm = LLM("model.gguf", main_gpu=1)                       # use GPU 1
llm = LLM("model.gguf", split_mode=1, n_gpu_layers=-1)    # layer split (default)
llm = LLM("model.gguf", split_mode=2, n_gpu_layers=-1)    # row split (tensor parallel)
llm = LLM("model.gguf", tensor_split=[0.3, 0.7])          # 30% GPU 0, 70% GPU 1

config = GenerationConfig(
    main_gpu=0,
    split_mode=1,          # 0=NONE, 1=LAYER, 2=ROW
    tensor_split=[1, 2],   # 1/3 GPU 0, 2/3 GPU 1
    n_gpu_layers=-1,
)
llm = LLM("model.gguf", config=config)
```

Split modes:

- `0` (NONE): one GPU, selected by `main_gpu`

- `1` (LAYER): layers and KV cache split across GPUs (default)

- `2` (ROW): layers split row-wise across GPUs

## Platforms

`pyproject.toml` holds the `inferna` version. [scripts/manage.py](scripts/manage.py) pins the `llama.cpp`, `whisper.cpp`, `stable-diffusion.cpp`, and `sqlite-vector` versions. The build uses scikit-build-core and CMake.

### Prebuilt wheels on PyPI

All wheels are `cp312-abi3`.

| Package | Backend | Platform | Arch | Linking |
|---|---|---|---|---|
| `inferna` | CPU | Linux | x86_64 | static |
| `inferna` | CPU | Windows | x86_64 | static |
| `inferna` | Metal | macOS | arm64 | static |
| `inferna-cuda12` | CUDA 12.4 | Linux | x86_64 | dynamic |
| `inferna-rocm` | ROCm 6.3 | Linux | x86_64 | dynamic |
| `inferna-sycl` | SYCL (oneAPI 2025.3) | Linux | x86_64 | dynamic |
| `inferna-vulkan` | Vulkan | Linux | x86_64 | dynamic |

macOS Intel is not supported. Windows GPU variants are not released; build them from source. Planned platforms are tracked in [TODO.md](TODO.md).

### Source builds

| Backend | macOS | Linux | Windows |
|---|---|---|---|
| CPU | `make build-cpu` | `make build-cpu` | `make build-cpu` |
| Metal | `make build-metal` (default) | -- | -- |
| CUDA | -- | `make build-cuda` | `make build-cuda` |
| ROCm (HIP) | -- | `make build-hip` | -- |
| Vulkan | `make build-vulkan` | `make build-vulkan` | `make build-vulkan` |
| SYCL | -- | `make build-sycl` | -- |
| OpenCL | `make build-opencl` | `make build-opencl` | `make build-opencl` |

Each backend also has a dynamic variant, `make build-<backend>-dynamic`.

## Building from source

Requirements: Python 3.12+, a C++ toolchain, CMake, and [uv](https://github.com/astral-sh/uv).

```sh
git clone https://github.com/shakfu/inferna.git
cd inferna
uv sync
make
```

`make` downloads and builds `llama.cpp`, `whisper.cpp`, and `stable-diffusion.cpp` into `thirdparty/`, then builds an editable `inferna` install.

### Build commands

```sh
# Build
make                  # deps + editable install (static linking)
make build-dynamic    # deps as shared libs

# Wheels
make wheel            # version-specific wheel in dist/
make wheel-abi3       # cp312-abi3 wheel in dist/, the format published to PyPI
make dist             # sdist + wheel in dist/
make wheel-cuda                # backend-specific, static
make wheel-cuda-dynamic        # backend-specific, shared libs bundled
make wheel-cuda-dynamic-abi3   # backend-specific, shared libs bundled, abi3

# Backend-specific builds (static)
make build-cpu
make build-metal      # default on macOS
make build-cuda
make build-vulkan
make build-hip
make build-sycl
make build-opencl

# Multiple backends
GGML_CUDA=1 GGML_VULKAN=1 make build

# Clean
make clean            # build artifacts + dynamic libs
make reset            # also thirdparty/ and .venv
make remake           # clean rebuild, then tests

# Code quality
make lint             # ruff, auto-fix
make format           # ruff format
make typecheck        # mypy
make qa               # lint + typecheck + format

# Leak check
make leaks            # RSS growth over 10 cycles, 20% threshold

# Publishing
make check            # validate wheels with twine
make publish          # upload to PyPI
make publish-test     # upload to TestPyPI
```

See [Build Backends](docs/build_backends.md) for per-backend instructions.

## Testing

The tests expect `models/Llama-3.2-1B-Instruct-Q8_0.gguf` ([Hugging Face](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf)). Download it into `models/`:

```sh
make download
```

Run the suite:

```sh
make test
```

The `tests/` directory doubles as a set of usage examples.

## Documentation

Full documentation: [shakfu.github.io/inferna](https://shakfu.github.io/inferna/) (MkDocs). Serve locally with `make docs-serve`.

- **[User Guide](docs/user_guide.md)** - all features

- **[CLI Cheatsheet](docs/cli-cheatsheet.md)** - every CLI command

- **[API Reference](docs/api_reference.md)**

- **[RAG Overview](docs/rag_overview.md)**

- **[Cookbook](docs/cookbook.md)** - recipes and patterns

- **[Changelog](CHANGELOG.md)** - release history

- **Examples** - `tests/examples/`

## Contributing

Contributions are welcome. See the [User Guide](docs/user_guide.md) for development guidelines.

## License

MIT. The wrapped projects, [llama.cpp](https://github.com/ggml-org/llama.cpp), [whisper.cpp](https://github.com/ggml-org/whisper.cpp), and [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), are also MIT-licensed.
