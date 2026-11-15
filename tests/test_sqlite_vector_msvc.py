"""The staged sqlite-vector sources must compile under MSVC (`sqlite-vector-inferna.patch`)."""

import re
from pathlib import Path

import pytest

VENDORED = Path(__file__).resolve().parent.parent / "thirdparty" / "sqlite-vector"

# Sources CMakeLists.txt compiles for MSVC x64. distance-neon.c / distance-rvv.c
# are arch-guarded and never reach MSVC x64.
MSVC_SOURCES = ["sqlite-vector.c", "distance-cpu.c", "distance-avx2.c", "distance-avx512.c", "distance-sse2.c"]


def _unguarded_builtins(text: str) -> list[int]:
    """Line numbers of `__builtin_*` calls not inside an `#if` on __GNUC__ / __clang__."""
    stack: list[list] = []  # [condition, in_first_branch]
    bad = []
    for n, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        m = re.match(r"#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)", s)
        if m:
            kw, cond = m.groups()
            if kw in ("if", "ifdef", "ifndef"):
                stack.append([("!" if kw == "ifndef" else "") + cond, True])
            elif kw == "elif" and stack:
                stack[-1] = [cond, False]
            elif kw == "else" and stack:
                stack[-1][1] = False
            elif kw == "endif" and stack:
                stack.pop()
            continue
        if "__builtin_" in s.split("//")[0]:
            if not any(first and re.search(r"__GNUC__|__clang__", c) and not c.startswith("!") for c, first in stack):
                bad.append(n)
    return bad


@pytest.mark.skipif(not VENDORED.is_dir(), reason="sqlite-vector sources not staged")
@pytest.mark.parametrize("name", MSVC_SOURCES)
def test_vendored_sources_guard_gcc_builtins(name):
    # MSVC has no __builtin_*; an unguarded call links on gcc/clang and fails with
    # LNK2019 on Windows only. Upstream's distance-avx2.c does this; the patch guards it.
    text = (VENDORED / name).read_text(encoding="utf-8")
    assert _unguarded_builtins(text) == []


def test_builtin_scan_detects_unguarded_call():
    assert _unguarded_builtins("int f(int x) { return __builtin_popcount(x); }\n") == [1]
    guarded = "#if defined(__GNUC__) || defined(__clang__)\n__builtin_popcount(x);\n#else\nslow(x);\n#endif\n"
    assert _unguarded_builtins(guarded) == []
    assert _unguarded_builtins("#ifdef __GNUC__\n#else\n__builtin_popcount(x);\n#endif\n") == [3]
