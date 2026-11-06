# torchfdtd-platform-mcp (`agent-fdtd-mcp` v1.0.1)

An MCP server for **AgentFDTD** (`torchfdtd-platform`). It exposes **11 tools** over stdio
(local clients, auto-configured via `.mcp.json`) and streamable HTTP (remote clients):

| tool | returns |
|---|---|
| `create_project(...)` | declaratively builds a 2D/3D optical project (`configs/projects/<id>.json`) with an auto-synthesized 4/6-surface Poynting conservation box |
| `validate_project(...)` | multi-tier pre-flight check (`dt_fs`, total Yee cells, memory admission, Poynting box closure, Pydantic schema) |
| `submit_run(config, dry_run=True)` | the exact command and pre-flight verdict, or the run's `{ok, exit_status, run_id, ledger_row, stderr_tail}` |
| `resume_run(run_id, dry_run=True)` | bit-identically resumes a checkpointed run from its dual-slot `ResidentRestartJournal` |
| `inspect_breakpoint(run_id)` | inspects checkpoint journal, completed/total steps, dual ping-pong slot health (`slot_a`, `slot_b`), and lock status |
| `get_certificate(run_id)` | verifies and returns the post-flight notary certificate (`verification_certificate.json`) with SHA-256 telemetry/readout cross-check |
| `list_configs()` | config/driver pairs recorded in `runs/ledger.jsonl`, with a one-line purpose |
| `describe_run(config)` | driver, solver role, and the criteria quoted from the leg's `plan.md` / `docs/leg-criteria.json` |
| `validate_config(config)` | `{ok, errors[]}` from the project's own `checks/pre_environment.py` gate |
| `read_ledger(limit)` | the last N ledger rows, in file order |
| `explain_exit(code)` | the shared exit vocabulary; unknown codes stay unknown |

```bash
pip install -e .                 # installs torchfdtd-platform-mcp
torchfdtd-platform-mcp           # stdio
torchfdtd-platform-mcp --http --host 0.0.0.0 --port 8787   # HTTP at /mcp
```

The design rule is one sentence: **the model proposes, the harness disposes.**
No tool calls a model. `submit_run` executes only `tools/run.sh`, and `ok` is
derived from the process exit status. Exit `57` is a lawful finding and comes
back as `ok: false, kind: "finding"`, never as success.
