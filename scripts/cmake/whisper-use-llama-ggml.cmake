# Injected into whisper.cpp via CMAKE_PROJECT_whisper.cpp_INCLUDE by scripts/manage.py.
# Defines the `ggml` target from llama.cpp's tree, so whisper's `if (NOT TARGET ggml)`
# skips its bundled copy and compiles against the ggml inferna links.
add_subdirectory("${INFERNA_GGML_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/ggml")
