# MCP test fixtures

`exit57.py` is the A3 driver. It is not a simulation and it imports no solver;
its whole result is the process exit status `57`, which the shared vocabulary
calls a finding. The test copies it into the isolated project root at
`src/exit57.py`, because `tools/launch.py` snapshots only `src/`, `checks/`,
`tools/` and `configs/` and runs the driver from that snapshot.

`exit57.json` is the matching config. Its `driver` field is a path under the
isolated root, not this directory, for the same reason.

`invalid-config.json` omits `task`, which the project's own gate
(`checks/pre_environment.py`) refuses. It is used to prove that a refusal is
visible in the payload and that no process starts.
