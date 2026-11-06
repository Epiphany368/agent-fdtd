"""MCP server for the torchfdtd-platform.

The server exposes six read/execute tools.  It never calls a model and it never
decides acceptance: ``submit_run`` returns the process exit status, and the
harness's own exit vocabulary is what says whether a run passed.

Importing this package must not import the solver.  ``torchfdtd`` is loaded by
``tools/run.sh`` in its own process, only after a run is requested.
"""

__version__ = "0.1.0"
