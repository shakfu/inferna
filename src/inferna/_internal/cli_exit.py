"""Fast-exit context manager for inferna CLI entrypoints.

Skips Python interpreter finalization on the way out, which avoids
segfaults in ggml GPU backend static destructors during ``dlclose`` at
shutdown. The issue is not driver-specific — it reproduces across CUDA,
Vulkan, and (by the same dlopen/dlclose teardown pattern) is expected on
ROCm and SYCL too. Library users who import inferna are unaffected — only
the ``python -m inferna ...`` style entrypoints opt in.
"""

from __future__ import annotations

import os
import sys
import traceback
from contextlib import contextmanager


@contextmanager
def cli_runtime():
    """Wrap a CLI ``main()`` call; flush stdio and ``os._exit`` on the way out.

    Usage:

        if __name__ == "__main__":
            with cli_runtime():
                sys.exit(main())
    """
    rc = 0
    try:
        yield
    except SystemExit as e:
        code = e.code
        if code is None:
            rc = 0
        elif isinstance(code, int):
            rc = code
        else:
            print(code, file=sys.stderr)
            rc = 1
    except KeyboardInterrupt:
        rc = 130
    except BaseException:
        traceback.print_exc()
        rc = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
