# Installation

This guide covers installing inferna on different platforms.

## Requirements

- Python 3.10 or later

- C++ compiler (clang or gcc)

- CMake 3.21+

- Git

### Platform-Specific Requirements

**macOS:**

```bash
xcode-select --install  # Xcode Command Line Tools
```

**Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3-dev
```

**Fedora/RHEL:**

```bash
sudo dnf install -y gcc-c++ cmake git python3-devel
```

## Install from PyPI

```bash
pip install inferna
```

### GPU-Accelerated Variants

GPU variants are available on PyPI as separate packages (dynamically linked, Linux x86_64 only):

```bash
pip install inferna-cuda12   # NVIDIA GPU (CUDA 12.4)
pip install inferna-rocm     # AMD GPU (ROCm 6.3, requires glibc >= 2.35)
pip install inferna-sycl     # Intel GPU (oneAPI SYCL 2025.3)
pip install inferna-vulkan   # Cross-platform GPU (Vulkan)
```

All GPU variants install the same `inferna` Python package -- only the compiled backend differs. Install one at a time (they replace each other). GPU variants require the corresponding driver/runtime installed on your system.

You can verify which backend is active after installation:

```bash
inferna info
```

## Build from Source

```bash
# Clone repository
git clone https://github.com/shakfu/inferna.git
cd inferna

# Build everything (downloads llama.cpp, whisper.cpp, builds inferna)
make

# Download a test model
make download

# Verify installation
python -c "from inferna import complete; print('OK')"
```

## Build Options

### Default Build

The default build enables GPU acceleration appropriate for your platform:

- **macOS**: Metal (Apple GPU)

- **Linux**: CPU-only (GPU backends optional)

```bash
make build
```

### GPU Backends

Build with specific GPU support (static or dynamic):

```bash
# Static builds (all libs compiled into the extension)
make build-cpu       # CPU only
make build-cuda      # NVIDIA CUDA
make build-vulkan    # Vulkan (cross-platform)

# Dynamic builds (shared libs installed alongside extension)
make build-cpu-dynamic
make build-cuda-dynamic
make build-vulkan-dynamic

# Multiple backends
GGML_CUDA=1 GGML_VULKAN=1 make build
```

See [Building with Different Backends](build_backends.md) for detailed GPU setup instructions.

### Optional Components

**Stable Diffusion support:**

```bash
WITH_STABLEDIFFUSION=1 make build

# Opt into sharing llama.cpp's ggml (not recommended for GPU backends)
SD_USE_VENDORED_GGML=0 make build
```

**Whisper support** (included by default):

```bash
make build  # Whisper is built automatically
```

## Build System

Inferna uses **scikit-build-core** with CMake for building the nanobind extensions. The build process:

1. **Dependencies**: `make` downloads and builds llama.cpp, whisper.cpp (and optionally stable-diffusion.cpp)
2. **nanobind compilation**: CMake compiles the hand-written `_*_native.cpp` files (one extension per upstream)
3. **Extension linking**: C++ extensions are linked against the static libraries
4. **Installation**: Extensions are installed in editable mode

### Build Commands

| Command | Description |
|---------|-------------|
| `make` | Full build (dependencies + editable install) |
| `make build-<backend>` | Static build for a specific backend (e.g. `build-cuda`) |
| `make build-<backend>-dynamic` | Dynamic build for a specific backend (e.g. `build-cuda-dynamic`) |
| `make wheel` | Build wheel for distribution |
| `make wheel-<backend>` | Static wheel for a specific backend |
| `make wheel-<backend>-dynamic` | Dynamic wheel for a specific backend |
| `make clean` | Remove build artifacts and dynamic libs |
| `make reset` | Full reset including thirdparty and .venv |
| `make remake` | Clean rebuild with tests |
| `make leaks` | RSS-growth memory leak detection |

### Wheel Distribution

To build a distributable wheel:

```bash
make wheel
# Creates wheel in dist/
```

The wheel includes all compiled extensions and can be installed on systems with matching platform/Python version.

## Installing Models

### LLM Models (GGUF format)

Download the default test model:

```bash
make download
# Downloads: models/Llama-3.2-1B-Instruct-Q8_0.gguf
```

Or download manually from [Hugging Face](https://huggingface.co/models?search=gguf):

```bash
# Example: Download a model
curl -L -o models/llama.gguf \
  "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf"
```

### Whisper Models

Download from [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp):

```bash
curl -L -o models/ggml-base.en.bin \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
```

### Stable Diffusion Models

Download SDXL Turbo or other SD models in GGUF or safetensors format.

## Verification

### Test Installation

```bash
# Run test suite
make test

# Quick smoke test
python -c "
from inferna import complete
print(complete('Hello', model_path='models/Llama-3.2-1B-Instruct-Q8_0.gguf', max_tokens=10))
"
```

### Check GPU Support

```python
from inferna.llama.llama_cpp import ggml_backend_load_all

# Load all available backends
ggml_backend_load_all()

# Check what's available
from inferna.llama.llama_cpp import LlamaModel, LlamaModelParams
params = LlamaModelParams()
params.n_gpu_layers = -1  # Offload all layers to GPU
# If GPU is available, layers will be offloaded
```

## Troubleshooting

### "No module named 'inferna'"

Make sure you're in the project directory or have installed inferna:

```bash
cd inferna
make  # or: uv pip install -e .
```

### Build Errors

Clean and rebuild:

```bash
make reset  # Full clean
make build
```

### Metal Not Working (macOS)

Ensure Xcode Command Line Tools are installed:

```bash
xcode-select --install
```

### CUDA Not Found (Linux)

Add CUDA to your PATH:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Development Install

For development with editable install:

```bash
git clone https://github.com/shakfu/inferna.git
cd inferna
make  # Builds dependencies and installs in editable mode
```

For manual editable install (after dependencies are built):

```bash
uv pip install -e .
```

## Next Steps

- [User Guide](user_guide.md) - Learn the API

- [Cookbook](cookbook.md) - Common patterns and recipes

- [Building with Different Backends](build_backends.md) - GPU setup details
