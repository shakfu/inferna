# Whisper.cpp Integration

Inferna wraps [whisper.cpp](https://github.com/ggerganov/whisper.cpp) to provide automatic speech recognition (ASR) capabilities in Python.

## Overview

The whisper module provides Python bindings to whisper.cpp, enabling:

- Speech-to-text transcription

- Multi-language support (100+ languages)

- Translation to English

- Word-level timestamps

- Voice activity detection (VAD)

- GPU acceleration (Metal, CUDA)

## Quick Start

### Basic Transcription

```python
from inferna.whisper import WhisperContext, WhisperFullParams
import numpy as np

# Load model
ctx = WhisperContext("models/ggml-base.en.bin")

# Load audio as float32 samples at 16kHz
# (Use your preferred audio library: scipy, soundfile, librosa, etc.)
samples = load_audio_as_float32("audio.wav")  # Shape: (n_samples,)

# Transcribe
params = WhisperFullParams()
ctx.full(samples, params)

# Get results
n_segments = ctx.full_n_segments()
for i in range(n_segments):
    text = ctx.full_get_segment_text(i)
    t0 = ctx.full_get_segment_t0(i)  # Start time in centiseconds
    t1 = ctx.full_get_segment_t1(i)  # End time in centiseconds
    print(f"[{t0/100:.2f}s - {t1/100:.2f}s] {text}")
```

### With Language Detection

```python
from inferna.whisper import WhisperContext, WhisperFullParams

ctx = WhisperContext("models/ggml-base.bin")  # Multilingual model

params = WhisperFullParams()
params.language = None  # Auto-detect language

ctx.full(samples, params)

# Get detected language
lang_id = ctx.full_lang_id()
lang_name = ctx.lang_str_full(lang_id)
print(f"Detected language: {lang_name}")
```

### Translation to English

```python
params = WhisperFullParams()
params.translate = True  # Translate to English
params.language = "de"   # Source language (German)

ctx.full(samples, params)
```

## API Reference

### Constants

```python
from inferna.whisper import WHISPER

WHISPER.SAMPLE_RATE   # 16000 - Required sample rate
WHISPER.N_FFT         # FFT size
WHISPER.HOP_LENGTH    # Hop length for STFT
WHISPER.CHUNK_SIZE    # Chunk size for processing
```

### WhisperContext

The main context class for model loading and inference.

```python
from inferna.whisper import WhisperContext, WhisperContextParams

# Basic loading
ctx = WhisperContext("models/ggml-base.bin")

# With parameters
params = WhisperContextParams()
params.use_gpu = True
params.flash_attn = True
params.gpu_device = 0

ctx = WhisperContext("models/ggml-base.bin", params)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `full(samples, params)` | Run full transcription pipeline |
| `full_n_segments()` | Get number of transcribed segments |
| `full_get_segment_text(i)` | Get text of segment i |
| `full_get_segment_t0(i)` | Get start time of segment i (centiseconds) |
| `full_get_segment_t1(i)` | Get end time of segment i (centiseconds) |
| `full_n_tokens(i)` | Get number of tokens in segment i |
| `full_get_token_text(i, j)` | Get text of token j in segment i |
| `full_get_token_id(i, j)` | Get ID of token j in segment i |
| `full_get_token_p(i, j)` | Get probability of token j in segment i |
| `full_get_token_t0(i, j)`, `full_get_token_t1(i, j)` | Token times (centiseconds, VAD-mapped); needs `token_timestamps` |
| `full_get_segment_speaker_turn_next(i)` | Next segment is a speaker turn; needs `tdrz_enable` |
| `full_n_vad_segments()` | VAD speech segments in the last `full()`; 0 without VAD |
| `full_get_vad_segment_t0(i)`, `full_get_vad_segment_t1(i)` | VAD segment bounds (centiseconds) |
| `get_timings()` | Mean ms per sample/encode/decode/batchd/prompt call |
| `full_lang_id()` | Get detected language ID |

Segment and token getters raise `IndexError` on an index outside the last `full()` result, and `RuntimeError` while another thread runs `full()` on the same context or state.

**Model Information:**

| Method | Description |
|--------|-------------|
| `is_multilingual()` | Check if model supports multiple languages |
| `n_vocab()` | Get vocabulary size |
| `n_text_ctx()` | Get text context size |
| `n_audio_ctx()` | Get audio context size |
| `model_type_readable()` | Get model type as string ("base", "small", etc.) |

**Tokenization:**

| Method | Description |
|--------|-------------|
| `tokenize(text)` | Convert text to token IDs |
| `token_to_str(id)` | Convert token ID to text |
| `token_count(text)` | Count tokens in text |

`full_parallel(samples, params, n_processors=2)` splits the audio into `n_processors` chunks, runs them on parallel states, and merges the results into the context. Words at chunk boundaries can be lost.

### WhisperState

A `WhisperState` holds its own buffers and results on a shared `WhisperContext`. The model loads once; each state runs `full()` independently, so separate states can transcribe on separate threads. A state has the same `full_*` result getters as `WhisperContext`.

| Member | Description |
|--------|-------------|
| `WhisperState(ctx)` | Create a state on `ctx` |
| `full(samples, params)` | Transcribe into this state |
| `close()` | Free the state's buffers |
| `is_valid`, `ctx` | Open flag; parent context |

One state runs one `full()` at a time; a second concurrent call raises `RuntimeError`. `WhisperContext.close()` raises while any of its states is open.

### Step-by-step pipeline

`full()` runs the whole pipeline. `WhisperContext` and `WhisperState` also expose each step:

| Method | Description |
|--------|-------------|
| `pcm_to_mel(samples, n_threads=1)` | Log mel spectrogram of 16 kHz mono samples |
| `set_mel(mel)` | Use your own mel, float32 of shape `(model_n_mels(), n_len)` |
| `n_len()` | Mel length in 10 ms frames |
| `encode(offset=0, n_threads=1)` | Run the encoder from mel frame `offset` |
| `decode(tokens, n_past=0, n_threads=1)` | Decode `tokens` after `n_past` cached tokens |
| `get_logits()` | Logits of the last token of the last `decode()`, shape `(model_n_vocab(),)` |
| `lang_auto_detect(offset_ms=0, n_threads=1)` | `(lang_id, probs)`; needs a multilingual model |

Greedy decoding without timestamps:

```python
ctx.pcm_to_mel(samples)
ctx.encode()
prompt = [ctx.token_sot(), ctx.token_not()]
ctx.decode(prompt)
out = []
while len(out) < 100:
    token = int(np.argmax(ctx.get_logits()))
    if token == ctx.token_eot():
        break
    out.append(token)
    ctx.decode([token], n_past=len(prompt) + len(out) - 1)
print("".join(ctx.token_to_str(t) for t in out))
```

`decode()` raises `ValueError` when `n_past + len(tokens)` exceeds `n_text_ctx()` or a token id is out of range.

### WhisperContextParams

Configuration for context creation.

```python
from inferna.whisper import WhisperContextParams

params = WhisperContextParams()
params.use_gpu = True           # Use GPU acceleration
params.flash_attn = True        # Use flash attention
params.gpu_device = 0           # GPU device index
params.dtw_token_timestamps = False  # Enable DTW for precise timestamps
```

### WhisperFullParams

Configuration for transcription.

```python
from inferna.whisper import WhisperFullParams, WhisperSamplingStrategy

params = WhisperFullParams()

# Sampling strategy
params.strategy = WhisperSamplingStrategy.GREEDY  # or BEAM_SEARCH

# Threading
params.n_threads = 4

# Language
params.language = "en"    # Set language (None for auto-detect)
params.translate = False  # Translate to English

# Timing
params.offset_ms = 0      # Start offset in milliseconds
params.duration_ms = 0    # Duration (0 = full audio)

# Output control
params.no_timestamps = False
params.single_segment = False
params.print_progress = False
params.print_realtime = False
params.print_timestamps = True

# Token timestamps
params.token_timestamps = False
params.temperature = 0.0
```

### WhisperVadParams

Voice activity detection parameters.

```python
from inferna.whisper import WhisperVadParams

vad = WhisperVadParams()
vad.threshold = 0.6              # VAD threshold (0-1)
vad.min_speech_duration_ms = 250 # Minimum speech duration
vad.min_silence_duration_ms = 100  # Minimum silence duration
vad.max_speech_duration_s = 30.0 # Maximum speech segment
vad.speech_pad_ms = 30           # Padding around speech
vad.samples_overlap = 0.0        # Sample overlap
```

### WhisperVadContext

Standalone Silero VAD, without a whisper model. `full()` with `params.vad = True` runs the same detector internally.

```python
from inferna.whisper import WhisperVadContext

vad = WhisperVadContext("models/ggml-silero-v5.1.2.bin")
vad.segments_from_samples(samples)   # [(29.0, 221.0), (330.0, 377.0), ...] in centiseconds
```

| Method | Description |
|--------|-------------|
| `WhisperVadContext(model_path, params=None)` | Load a VAD model; `params` is a `WhisperVadContextParams` (`n_threads`, `use_gpu`, `gpu_device`) |
| `detect_speech(samples, reset=True)` | Speech probability per window; `reset=False` keeps the LSTM state for streaming |
| `reset_state()` | Clear the LSTM state between utterances |
| `probs()` | Probabilities from the last `detect_speech()` |
| `segments_from_probs(params=None)` | Segments from those probabilities, using `WhisperVadParams` |
| `segments_from_samples(samples, params=None)` | `detect_speech()` then `segments_from_probs()` |
| `close()`, `is_valid` | Free the model; open flag |

A whisper transcription model passed as `model_path` raises `ValueError`.

### Sampling Strategies

```python
from inferna.whisper import WhisperSamplingStrategy

WhisperSamplingStrategy.GREEDY      # Fast, deterministic
WhisperSamplingStrategy.BEAM_SEARCH # Better quality, slower
```

### Language Functions

```python
from inferna.whisper import lang_id, lang_str, lang_str_full, lang_max_id

# Get language ID from code
id = lang_id("en")  # Returns 0

# Get language code from ID
code = lang_str(0)  # Returns "en"

# Get full language name
name = lang_str_full(0)  # Returns "english"

# Get maximum language ID
max_id = lang_max_id()  # Returns ~100
```

### Module Functions

```python
from inferna.whisper import version, print_system_info

# Get whisper.cpp version
ver = version()

# Get system info (CPU features, etc.)
info = print_system_info()
```

## Audio Preparation

Whisper requires:

- **Sample rate**: 16000 Hz (mono)

- **Format**: Float32 normalized to [-1.0, 1.0]

### Using scipy

```python
from scipy.io import wavfile
import numpy as np

def load_audio(path: str) -> np.ndarray:
    rate, data = wavfile.read(path)

    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Convert to float32
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0

    # Resample to 16kHz if needed
    if rate != 16000:
        from scipy import signal
        num_samples = int(len(data) * 16000 / rate)
        data = signal.resample(data, num_samples)

    return data.astype(np.float32)
```

### Using soundfile

```python
import soundfile as sf
import numpy as np

def load_audio(path: str) -> np.ndarray:
    data, rate = sf.read(path, dtype='float32')

    # Convert to mono
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Resample if needed
    if rate != 16000:
        import resampy
        data = resampy.resample(data, rate, 16000)

    return data.astype(np.float32)
```

## Common Patterns

### Transcription with Timestamps

```python
def transcribe_with_timestamps(audio_path: str, model_path: str) -> list:
    ctx = WhisperContext(model_path)
    samples = load_audio(audio_path)

    params = WhisperFullParams()
    params.print_timestamps = True

    ctx.full(samples, params)

    results = []
    for i in range(ctx.full_n_segments()):
        results.append({
            "start": ctx.full_get_segment_t0(i) / 100.0,
            "end": ctx.full_get_segment_t1(i) / 100.0,
            "text": ctx.full_get_segment_text(i).strip()
        })

    return results
```

### Word-Level Timestamps

```python
def transcribe_with_word_timestamps(audio_path: str, model_path: str) -> list:
    params = WhisperContextParams()
    params.dtw_token_timestamps = True

    ctx = WhisperContext(model_path, params)
    samples = load_audio(audio_path)

    fparams = WhisperFullParams()
    fparams.token_timestamps = True

    ctx.full(samples, fparams)

    words = []
    for i in range(ctx.full_n_segments()):
        for j in range(ctx.full_n_tokens(i)):
            token_data = ctx.full_get_token_data(i, j)
            text = ctx.full_get_token_text(i, j)
            if text.strip():
                words.append({
                    "word": text,
                    "start": token_data.t0 / 100.0,
                    "end": token_data.t1 / 100.0,
                    "probability": token_data.p
                })

    return words
```

### Batch Processing

```python
def transcribe_batch(audio_paths: list, model_path: str) -> dict:
    ctx = WhisperContext(model_path)
    params = WhisperFullParams()

    results = {}
    for path in audio_paths:
        samples = load_audio(path)
        ctx.full(samples, params)

        text = ""
        for i in range(ctx.full_n_segments()):
            text += ctx.full_get_segment_text(i)

        results[path] = text.strip()

    return results
```

To transcribe files concurrently, give each thread its own `WhisperState` on one context:

```python
from concurrent.futures import ThreadPoolExecutor

ctx = WhisperContext(model_path)

def transcribe(path):
    state = WhisperState(ctx)
    state.full(load_audio(path), WhisperFullParams())
    text = "".join(state.full_get_segment_text(i) for i in range(state.full_n_segments()))
    state.close()
    return text.strip()

with ThreadPoolExecutor(max_workers=2) as pool:
    results = dict(zip(audio_paths, pool.map(transcribe, audio_paths)))
```

### Streaming Transcription

For real-time or streaming audio, process in chunks:

```python
def transcribe_stream(audio_stream, model_path: str, chunk_seconds: float = 30.0):
    ctx = WhisperContext(model_path)
    params = WhisperFullParams()
    params.single_segment = True

    chunk_samples = int(16000 * chunk_seconds)
    buffer = np.array([], dtype=np.float32)

    for chunk in audio_stream:
        buffer = np.concatenate([buffer, chunk])

        if len(buffer) >= chunk_samples:
            ctx.full(buffer[:chunk_samples], params)

            for i in range(ctx.full_n_segments()):
                yield ctx.full_get_segment_text(i)

            # Keep overlap for continuity
            buffer = buffer[chunk_samples - 1600:]  # 100ms overlap
```

## Model Selection

| Model | Size | Memory | Speed | Quality |
|-------|------|--------|-------|---------|
| tiny | 75 MB | ~400 MB | Fastest | Basic |
| base | 142 MB | ~500 MB | Fast | Good |
| small | 466 MB | ~1 GB | Medium | Better |
| medium | 1.5 GB | ~2.5 GB | Slow | Great |
| large-v3 | 3 GB | ~5 GB | Slowest | Best |
| large-v3-turbo | 1.6 GB | ~3 GB | Medium | Great |

**English-only models** (`.en` suffix) are faster and more accurate for English:

- `ggml-tiny.en.bin`

- `ggml-base.en.bin`

- `ggml-small.en.bin`

- `ggml-medium.en.bin`

Download models from [Hugging Face](https://huggingface.co/ggerganov/whisper.cpp).

## Performance Tips

1. **Use GPU**: Enable `use_gpu=True` in context params

2. **Use Flash Attention**: Enable `flash_attn=True` for faster inference

3. **Match model to task**: Use `.en` models for English-only content

4. **Batch by length**: Group similar-length audio for consistent memory usage

5. **Thread count**: Set `n_threads` to match physical CPU cores

## Troubleshooting

### Model Loading Errors

```python
# Check model path exists
import os
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

# Check model format
if not model_path.endswith('.bin'):
    print("Warning: Whisper models should be .bin format (ggml)")
```

### Audio Issues

```python
# Verify audio format
print(f"Sample rate: {rate}")
print(f"Dtype: {samples.dtype}")
print(f"Shape: {samples.shape}")
print(f"Range: [{samples.min():.2f}, {samples.max():.2f}]")

# Should be: 16000, float32, (N,), [-1.0, 1.0]
```

### Memory Issues

```python
# Use smaller model
ctx = WhisperContext("models/ggml-tiny.bin")

# Disable GPU if VRAM limited
params = WhisperContextParams()
params.use_gpu = False
```

## See Also

- [whisper.cpp repository](https://github.com/ggerganov/whisper.cpp)

- [Whisper model card](https://huggingface.co/openai/whisper-large-v3)
