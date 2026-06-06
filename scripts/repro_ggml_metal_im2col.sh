#!/usr/bin/env bash
#
# Standalone, from-scratch reproduction of the ggml-metal `im2col_ext`
# miscomputation that makes whisper.cpp emit an EMPTY transcript on the Apple
# Metal backend.
#
# The single variable isolated here is *which ggml whisper links against*:
#
#   build A: whisper.cpp + its OWN vendored ggml      -> correct transcript
#   build B: whisper.cpp + llama.cpp's ggml (b9528)   -> EMPTY transcript (GPU)
#   build B on CPU (-ng)                              -> correct transcript
#
# Everything else is identical: same whisper-cli source (v1.8.6), same model,
# same audio, same machine, same Metal. The only difference is the linked ggml,
# so a correct->empty flip pins the defect to llama's ggml-metal.
#
# Root cause (see docs/dev/ggml-metal-issue.md): llama's
# ggml/src/ggml-metal/ggml-metal-device.cpp selects `kernel_im2col_ext` when
# ne00*ne01 > 1024 (= KW*IC for a 1-D conv), but ggml-metal-ops.cpp picks the
# dispatch geometry off KH*KW. whisper's encoder conv2 (KW=3, IC=512, KH=1) has
# KW*IC = 1536 > 1024 (selects _ext) yet KH*KW = 3 (base geometry) -> the _ext
# kernel runs under the base grid, mis-indexes, writes garbage -> the decoder
# sees noise -> zero segments. LLM inference never uses im2col, so it is
# invisible upstream.
#
# Requirements: macOS on Apple Silicon (Metal), git, cmake, a C/C++ toolchain
# (Xcode CLT), curl. Network access (clones two repos, downloads ~147MB model).
#
# Usage:
#   scripts/repro_ggml_metal_im2col.sh
#   WORKDIR=/tmp/repro scripts/repro_ggml_metal_im2col.sh   # custom scratch dir
#   FORCE=1 scripts/repro_ggml_metal_im2col.sh              # rebuild everything
#
# Exit code: 0 if the bug reproduced (correct->empty flip observed), 1 if not
# (e.g. an upstream fix landed, or the environment differs), 2 on setup error.

set -euo pipefail

# ---------------------------------------------------------------------------
# Config (override via environment)
# ---------------------------------------------------------------------------
LLAMA_VER="${LLAMA_VER:-b9528}"
WHISPER_VER="${WHISPER_VER:-v1.8.6}"
WORKDIR="${WORKDIR:-$PWD/ggml-metal-repro}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
FORCE="${FORCE:-0}"

LLAMA_REPO="https://github.com/ggml-org/llama.cpp"
WHISPER_REPO="https://github.com/ggml-org/whisper.cpp"

PREFIX="$WORKDIR/llama-ggml-install"     # llama's ggml, installed (find_package-able)
B_LLAMA_GGML="$WORKDIR/build-llama-ggml"
B_WH_OWN="$WORKDIR/build-whisper-own-ggml"
B_WH_LLAMA="$WORKDIR/build-whisper-llama-ggml"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 2; }

[ "$(uname -s)" = "Darwin" ] || die "this reproduction is macOS/Metal-specific (uname=$(uname -s))"
[ "$(uname -m)" = "arm64" ]  || note "WARNING: not arm64 ($(uname -m)); Metal path may differ"
for tool in git cmake curl; do command -v "$tool" >/dev/null || die "missing required tool: $tool"; done

if [ "$FORCE" = "1" ]; then
    say "FORCE=1 -- removing previous build trees (keeping clones + model)"
    rm -rf "$PREFIX" "$B_LLAMA_GGML" "$B_WH_OWN" "$B_WH_LLAMA"
fi

mkdir -p "$WORKDIR"

# ---------------------------------------------------------------------------
# 1. Clone (shallow, pinned tags)
# ---------------------------------------------------------------------------
clone_pinned() {
    local repo="$1" tag="$2" dir="$3"
    if [ -d "$dir/.git" ]; then
        note "already cloned: $dir"
    else
        say "Cloning $repo @ $tag"
        git clone --depth 1 --branch "$tag" "$repo" "$dir"
    fi
}
clone_pinned "$LLAMA_REPO"   "$LLAMA_VER"   "$WORKDIR/llama.cpp"
clone_pinned "$WHISPER_REPO" "$WHISPER_VER" "$WORKDIR/whisper.cpp"

# ---------------------------------------------------------------------------
# 2. Build + install llama.cpp (Metal, embedded shaders), which installs its
#    ggml as a find_package(ggml) package. This is the ggml that contains the
#    defective im2col_ext selection. We build the full tree (examples/tools/
#    server/tests OFF) rather than ggml standalone, because llama.cpp's
#    vendored ggml/ subtree is stripped of the tests/examples/ggml.pc.in that
#    ggml's standalone (top-level) CMake path requires.
# ---------------------------------------------------------------------------
if [ -f "$PREFIX/lib/cmake/ggml/ggml-config.cmake" ] && [ "$FORCE" != "1" ]; then
    note "llama ggml already installed at $PREFIX"
else
    say "Building + installing llama.cpp ($LLAMA_VER) ggml with Metal"
    cmake -S "$WORKDIR/llama.cpp" -B "$B_LLAMA_GGML" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_METAL=ON \
        -DGGML_METAL_EMBED_LIBRARY=ON \
        -DGGML_BLAS=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TOOLS=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_BUILD_APP=OFF \
        -DLLAMA_CURL=OFF \
        -DCMAKE_INSTALL_PREFIX="$PREFIX"
    cmake --build "$B_LLAMA_GGML" -j "$JOBS"
    cmake --install "$B_LLAMA_GGML"
    [ -f "$PREFIX/lib/cmake/ggml/ggml-config.cmake" ] || \
        die "llama ggml install did not produce a find_package(ggml) config at $PREFIX"
fi

# ---------------------------------------------------------------------------
# 3. Build whisper-cli TWICE -- identical source, different ggml.
# ---------------------------------------------------------------------------
build_whisper_cli() {
    local builddir="$1"; shift
    cmake -S "$WORKDIR/whisper.cpp" -B "$builddir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DWHISPER_BUILD_TESTS=OFF \
        -DWHISPER_BUILD_SERVER=OFF \
        "$@"
    cmake --build "$builddir" -j "$JOBS" --target whisper-cli
}

cli_path() { find "$1" -name whisper-cli -type f -perm -u+x 2>/dev/null | head -1; }

# build A: whisper's OWN ggml (vendored subdirectory) -- the correct baseline
if [ -n "$(cli_path "$B_WH_OWN")" ] && [ "$FORCE" != "1" ]; then
    note "whisper-cli (own ggml) already built"
else
    say "Building whisper-cli against whisper's OWN ggml (baseline)"
    build_whisper_cli "$B_WH_OWN" \
        -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_BLAS=ON
fi

# build B: llama's ggml via WHISPER_USE_SYSTEM_GGML + find_package -> the defect
if [ -n "$(cli_path "$B_WH_LLAMA")" ] && [ "$FORCE" != "1" ]; then
    note "whisper-cli (llama ggml) already built"
else
    say "Building whisper-cli against llama.cpp's ggml (WHISPER_USE_SYSTEM_GGML)"
    build_whisper_cli "$B_WH_LLAMA" \
        -DWHISPER_USE_SYSTEM_GGML=ON \
        -DCMAKE_PREFIX_PATH="$PREFIX"
fi

CLI_OWN="$(cli_path "$B_WH_OWN")";   [ -n "$CLI_OWN" ]   || die "whisper-cli (own ggml) not found"
CLI_LLAMA="$(cli_path "$B_WH_LLAMA")"; [ -n "$CLI_LLAMA" ] || die "whisper-cli (llama ggml) not found"

# ---------------------------------------------------------------------------
# 4. Model + audio (both come from the whisper.cpp clone)
# ---------------------------------------------------------------------------
MODEL="$WORKDIR/whisper.cpp/models/ggml-base.en.bin"
AUDIO="$WORKDIR/whisper.cpp/samples/jfk.wav"
[ -f "$AUDIO" ] || die "sample audio missing: $AUDIO (expected in the whisper.cpp clone)"
if [ ! -f "$MODEL" ]; then
    say "Downloading whisper base.en model"
    bash "$WORKDIR/whisper.cpp/models/download-ggml-model.sh" base.en "$WORKDIR/whisper.cpp/models"
fi
[ -f "$MODEL" ] || die "model download failed: $MODEL"

# ---------------------------------------------------------------------------
# 5. Run: A on GPU, B on GPU, B on CPU. Capture transcripts (-nt = text only).
# ---------------------------------------------------------------------------
run_cli() {  # <cli> <outfile> [extra flags...]
    local cli="$1" out="$2"; shift 2
    "$cli" -m "$MODEL" -f "$AUDIO" -l en -nt "$@" >"$out" 2>"$out.log" || true
    # keep only non-empty, non-bracketed transcription lines
    grep -vE '^\s*$|^\[' "$out" | tr -d '\r' | sed 's/^[[:space:]]*//' > "$out.clean" || true
}

say "Running whisper-cli (own ggml)  on Metal/GPU"
run_cli "$CLI_OWN"   "$WORKDIR/own.gpu"
say "Running whisper-cli (llama ggml) on Metal/GPU"
run_cli "$CLI_LLAMA" "$WORKDIR/llama.gpu"
say "Running whisper-cli (llama ggml) on CPU (-ng)"
run_cli "$CLI_LLAMA" "$WORKDIR/llama.cpu" -ng

PHRASE='ask not|my fellow americans|do for your country'
has_phrase()  { grep -iqE "$PHRASE" "$1.clean"; }
show()        { local t; t="$(tr '\n' ' ' < "$1.clean" | sed 's/  */ /g; s/^ //; s/ $//')"; printf '        "%s"\n' "${t:-<EMPTY>}"; }

# ---------------------------------------------------------------------------
# 6. Verdict
# ---------------------------------------------------------------------------
say "Transcripts"
note "A) whisper own ggml,  GPU:"; show "$WORKDIR/own.gpu"
note "B) llama ggml,        GPU:"; show "$WORKDIR/llama.gpu"
note "B) llama ggml,        CPU:"; show "$WORKDIR/llama.cpu"

own_gpu_ok=0;  has_phrase "$WORKDIR/own.gpu"   && own_gpu_ok=1
llama_gpu_ok=0; has_phrase "$WORKDIR/llama.gpu" && llama_gpu_ok=1
llama_cpu_ok=0; has_phrase "$WORKDIR/llama.cpu" && llama_cpu_ok=1

say "Result"
note "A whisper-own-ggml GPU transcribed : $([ $own_gpu_ok = 1 ] && echo YES || echo 'NO (empty)')"
note "B llama-ggml       GPU transcribed : $([ $llama_gpu_ok = 1 ] && echo YES || echo 'NO (empty)')"
note "B llama-ggml       CPU transcribed : $([ $llama_cpu_ok = 1 ] && echo YES || echo 'NO (empty)')"
echo

if [ "$own_gpu_ok" = 1 ] && [ "$llama_gpu_ok" = 0 ] && [ "$llama_cpu_ok" = 1 ]; then
    printf '\033[1;31mBUG REPRODUCED\033[0m: whisper on llama.cpp ggml-metal yields an EMPTY\n'
    printf 'transcript on the GPU while (a) the SAME whisper on its own ggml works on the\n'
    printf 'GPU and (b) the SAME llama-ggml build works on the CPU. The defect is in\n'
    printf "llama's ggml-metal im2col path, exercised only on the GPU.\n"
    exit 0
fi

printf '\033[1;33mNOT REPRODUCED\033[0m with these pins (llama=%s whisper=%s).\n' "$LLAMA_VER" "$WHISPER_VER"
printf 'Either an upstream ggml fix landed, the build differs, or the environment is not\n'
printf 'Metal. Inspect the transcripts and *.log files under: %s\n' "$WORKDIR"
exit 1
