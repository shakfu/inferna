"""Public surface of the whisper.cpp wrapper.

After the Cython → nanobind migration this is a pure-Python facade that
re-exports the native bindings (``_whisper_native``). External callers
continue to ``from inferna.whisper.whisper_cpp import ...`` the same names
they used before.
"""

from ._whisper_native import *  # noqa: F401,F403
from . import _whisper_native as _N

# Classes
WhisperContext = _N.WhisperContext
WhisperContextParams = _N.WhisperContextParams
WhisperFullParams = _N.WhisperFullParams
WhisperVadParams = _N.WhisperVadParams
WhisperTokenData = _N.WhisperTokenData
WhisperState = _N.WhisperState

# Constant containers
WHISPER = _N.WHISPER
WhisperSamplingStrategy = _N.WhisperSamplingStrategy
WhisperAheadsPreset = _N.WhisperAheadsPreset
WhisperGretype = _N.WhisperGretype

# Module-level functions
ggml_backend_load_all = _N.ggml_backend_load_all
disable_logging = _N.disable_logging
version = _N.version
print_system_info = _N.print_system_info
lang_max_id = _N.lang_max_id
lang_id = _N.lang_id
lang_str = _N.lang_str
lang_str_full = _N.lang_str_full

# Streaming API (rolling-window real-time transcription).
from .streaming import StreamSegment, WhisperStreamer, transcribe_stream  # noqa: E402, F401
