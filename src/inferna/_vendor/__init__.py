"""Vendored third-party libraries.

This package contains pure-Python copies of upstream libraries that
inferna depends on internally. Vendoring (rather than declaring them as
runtime dependencies in ``pyproject.toml``) keeps inferna wheels
self-contained: ``pip install inferna`` resolves as a single package
with no transitive Python deps.

The vendored libraries here are loaded under the ``inferna._vendor.*``
namespace so they cannot collide with whatever versions the user has
installed in their own environment.

**Do not modify the contents of vendored sub-packages directly.** Any
local changes will be lost the next time the libraries are re-vendored
from upstream. If a vendored library needs a fix, file the fix upstream
and re-vendor (see ``scripts/vendor_jinja2.sh``).

See ``README.md`` in this directory for the current vendored versions
and the re-vendoring procedure.
"""
