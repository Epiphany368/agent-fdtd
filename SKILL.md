---
name: agent-fdtd
description: AgentFDTD (v1.0.1) — Machine-verified electromagnetic FDTD simulation, photonic inverse design, and post-flight notary platform built on TorchFDTD v1.1.7. Use this skill when modeling 2D/3D optical structures, running forward or adjoint FDTD simulations, resuming checkpoints, or verifying energy conservation and cross-solver parity via MCP tools or CLI.
---

# AgentFDTD (`agent-fdtd` v1.0.1) — Agent Skill Guide

This repository provides a **zero-trust, machine-verified FDTD photonic simulation and inverse-design platform** driven via an MCP (Model Context Protocol) server (`agent-fdtd-mcp` / `torchfdtd-platform-mcp`) or the single-entry launcher (`./tools/run.sh`).

## 1. One-Command Agent Bootstrap

After cloning this repository, run the automated bootstrap script once to initialize the Python virtual environment (`.venv`), link or initialize the bundled/local `torchfdtd` solver (`src/torchfdtd`), install the `torchfdtd-platform-mcp` CLI entry point, and run the spine self-check:

```bash
./tools/setup_agent.sh
```

If you have an external `torchfdtd` checkout, pass its path explicitly:

```bash
./tools/setup_agent.sh /path/to/torchfdtd
```

## 2. MCP Server (`torchfdtd-platform-mcp`) — 11 Tools

When opening this repository in an MCP-capable coding agent (Claude Code, Antigravity, Cursor, OpenCode, Codex), `.mcp.json` automatically mounts the `agent-fdtd` server over `stdio`. You can also start it manually:

```bash
.venv/bin/agent-fdtd-mcp                                # stdio transport
.venv/bin/agent-fdtd-mcp --http --port 8787             # streamable HTTP transport
```

### Declarative Lifecycle Workflow (Recommended for Agents)

1. **`create_project(...)`** — Build a 2D or 3D optical scene declaratively (no Python driver script needed). Automatically synthesizes a 4-surface (2D) or 6-surface (3D) Poynting flux conservation box and writes `configs/projects/<project_id>.json`.
2. **`validate_project(config=...)`** — Run multi-tier pre-flight validation (CFL Courant time step `dt_fs`, total Yee cell count against resident ceiling, memory admission, and Poynting box closure).
3. **`submit_run(config=..., dry_run=False)`** — Execute `./tools/run.sh` under exclusive lock with source snapshotting and post-flight notary verification (`checks/post_notary.py`).
4. **`inspect_breakpoint(run_id=...)` / `resume_run(run_id=..., dry_run=False)`** — Inspect checkpoint state and dual ping-pong slots, or resume a checkpointed/partial run bit-identically from its dual-slot `ResidentRestartJournal`.
5. **`get_certificate(run_id=...)`** — Inspect and verify `runs/<run_id>/verification_certificate.json` (`verified: true`, SHA-256 telemetry-readout integrity, passivity, V2-box closure, energy decay, or inverse-design end-to-end FDTD Taylor/M3-realized gates).

### Complete Tool Reference

| Tool | Purpose | Key Parameters |
|---|---|---|
| `create_project` | Declaratively create a 2D/3D optical simulation config with auto-synthesized Poynting box | `name`, `dimension` (`"2d"`/`"3d"`), `domain`, `grid`, `steps`, `materials`, `structures`, `grating_teeth`, `source`, `poynting_box` |
| `validate_project` | Pre-flight physics, CFL, memory, and Poynting box validation | `config` (path) or inline `project` dict |
| `submit_run` | Run or dry-run a simulation via `./tools/run.sh` | `config`, `dry_run` (`True`/`False`), `resume_from` (optional) |
| `resume_run` | Bit-identically resume a run from its `ResidentRestartJournal` | `run_id`, `dry_run` (`True`/`False`) |
| `inspect_breakpoint` | Inspect checkpoint journal, completed/total steps, and dual slot health | `run_id` |
| `get_certificate` | Verify and return the post-flight notary certificate for a run | `run_id` |
| `list_configs` | List registered `(config, driver)` pairs from the ledger | none |
| `describe_run` | Inspect a config's driver, solver role, and pre-declared acceptance criteria | `config` |
| `validate_config` | Run `checks/pre_environment.py` pre-flight check without starting a simulation | `config` |
| `read_ledger` | Read the last `N` rows from `runs/ledger.jsonl` | `limit` (default `20`) |
| `explain_exit` | Decode any platform/driver exit status into structured meaning | `code` (integer) |

## 3. Direct CLI Execution & Key Capabilities

All simulations execute through `./tools/run.sh <config> <script> [args...]`:

```bash
# Spine self-test (audit mode, fast CPU verification)
./tools/run.sh configs/selftest-spine.json src/selftest_spine.py

# Inverse design with end-to-end FDTD gradient verification, M3 discrete re-simulation & Armijo line search
./tools/run.sh configs/leg49-inverse-design.json src/leg49_inverse_design.py

# Cross-solver reconciliation (TorchFDTD vs Meep & TORCWA RCWA)
./tools/run.sh configs/leg48-reconciliation.json src/leg48_reconciliation.py

# Run platform verification & MCP test suites
.venv/bin/python -m unittest discover -s checks
.venv/bin/python -m unittest discover -s mcptests
```

### Solver Capabilities Included (`src/torchfdtd`)
- **Differentiable Inverse Design & M3 Fabrication Verification (`src/leg49_inverse_design.py`, `src/leg49_m3_constraints.py`)**: 3-tier M3 conic filtering, smooth $\tanh$ projection, discrete morphological opening/closing ($\ge 150\text{ nm}$, zero violations), end-to-end FDTD directional Taylor gradient verification ($\le 10^{-6}$), and FDTD re-simulation of the realized discrete binary grating.
- **Reversible Lorentz Dispersive Adjoint (`v1.1.7`)**: Exact time-reversal and periodic/recorded CPML adjoint gradients for isotropic Lorentz media and two-material density mixtures (`torchfdtd/reversible_lorentz.py`), plus discrete FDTD dispersion fitting (`fit_discrete_lorentz` in `torchfdtd/material_fit.py`).
- **Sources & Boundaries**: Plane-wave, Bloch periodic, Angular Spectrum Method (ASM) oblique Gaussian beam (`src/oblique_asm.py`, Vitali non-paraxial Gaussian in `torchfdtd/injection.py`), uniform & graded modal launch (`torchfdtd/mode_branches.py`), and vendor-aligned SC-PML (`torchfdtd/vendor_scpml.py`).
- **Dual-Slot Checkpoint Journal**: Crash-safe, bit-identical checkpointing and resume (`torchfdtd/resident_restart.py`).

## 4. Exit Status Contract

Never judge a run by the presence of output files; **acceptance is the process exit status**:

- **`0`**: All pre-declared criteria and post-flight notary checks (`checks/post_notary.py`) passed; `verification_certificate.json` issued with `verified: true`.
- **`53`–`59`**: Lawful scientific findings (e.g., `53` non-monotone convergence ladder, `57` threshold violation, `59` admission OOM refusal).
- **`64` / `65` / `69`**: Usage error / pre-flight refusal (`checks/pre_environment.py` or `checks/pre_resume.py`) / Python unavailable — no simulation was executed.
- **`66`**: Post-flight notary rejection (`checks/post_notary.py` detected telemetry tampering, passivity violation, unclosed Poynting box, or cross-solver discrepancy).
