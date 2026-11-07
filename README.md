# AgentFDTD (`agent-fdtd`) — v2.0.0

**Language / 语言:** [English](README.md) | [简体中文](README_zh.md)

**An agent-built, machine-verified FDTD simulation and photonic inverse-design platform.**
AI proposes a simulation or scientific claim; a deterministic process exit status and post-flight notary decide whether it passed.

[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)
[![Version: 2.0.0](https://img.shields.io/badge/version-2.0.0-blue.svg)](pyproject.toml)

---

## Why this exists

Photonic design runs on simulation, and simulation results are famously easy to produce and
famously hard to trust. A transmission number comes out of a solver, goes into a paper or a
tape-out review, and the question nobody can answer six months later is: *what exactly was
checked, and against what?*

This platform inverts the usual order. **Nothing is claimed without a criterion that can
fail.** Every capability ships with:

- a **pre-declared** criterion and a **pre-declared exit status**, written down before the run;
- a **negative control** — a case constructed so the check *must* fail, because a gate whose
  failure mode has never been produced is a description, not a gate;
- a **ledger row** recording what ran, against which solver commit, with which exit;
- a **long-form evidence document** stating the numbers, the commands, and — explicitly —
  what the leg could **not** determine.

---

## Quick start (One-command Agent & Developer Setup)

```bash
git clone https://github.com/nillinjay/agent-fdtd
cd agent-fdtd
./tools/setup_agent.sh      # creates .venv, links/initialises TorchFDTD solver, installs MCP server, runs spine check
```

If you prefer manual setup with an external TorchFDTD checkout:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Link an external TorchFDTD checkout:
tools/bootstrap_solver.sh /opt/torchfdtd
```

Run the baseline verification and test suites — and read the exit status:

```bash
./tools/run.sh configs/selftest-spine.json src/selftest_spine.py    # exit 0 expected
.venv/bin/python -m unittest discover -s checks                     # platform verification & negative controls
.venv/bin/python -m unittest discover -s mcptests                   # MCP server test suite
```

---

## Drive it from an AI client (MCP + `SKILL.md`)

The repository ships a root `SKILL.md` (Agent Skill manifest) and `.mcp.json` (zero-config MCP
discovery), so an AI coding agent (Claude Code, Antigravity, Cursor, OpenCode, Codex) that clones
this project can directly invoke the platform's **MCP tools**:

```bash
pip install -e .                        # installs `agent-fdtd-mcp` (and `torchfdtd-platform-mcp`)
agent-fdtd-mcp                          # MCP over stdio (local clients)
agent-fdtd-mcp --http --port 8787       # streamable HTTP (remote clients)
```

### Core Simulation & Verification Tools

| Tool | Declared Capability |
|---|---|
| `create_project(...)` | Declaratively creates a 2D/3D optical simulation scene (`configs/projects/<id>.json`) with automated energy-conservation monitors |
| `validate_project(...)` | Multi-tier pre-flight admission check (time-step bounds, grid dimensions, memory admission, monitor closure) |
| `submit_run(config, dry_run=True)` | `dry_run` returns the exact command and pre-flight verdict; otherwise executes under lock and returns **`{ok, exit_status, run_id, kind}`** |
| `resume_run(run_id, dry_run=True)` | Bit-identically resumes a checkpointed simulation from its crash-safe restart journal |
| `inspect_breakpoint(run_id)` | Inspects checkpoint progress, completed/target steps, slot integrity, and lock status |
| `get_certificate(run_id)` | Verifies and returns the post-flight notary certificate (`verification_certificate.json`) |
| `list_configs()` | Lists registered runnable configurations |
| `describe_run(config)` | Returns the driver, solver role, and pre-declared acceptance criteria |
| `validate_config(config)` | Runs structural and environment validators without launching a simulation |
| `read_ledger(limit)` | Reads the last N tracked rows from `runs/ledger.jsonl` |
| `explain_exit(code)` | Explains the platform exit-code vocabulary (`0`, `53`–`59`, `61`–`66`, `69`) |

### v2.0 Experiment, Claim & Delivery Tools (Opt-in Experiment Surface)

| Tool | Declared Capability |
|---|---|
| `create_experiment(...)` | Registers a solver-agnostic photonic experiment definition (`TorchFDTD`, `RCWA`, `Meep`, `GDS`) |
| `evaluate_claim(...)` | Evaluates a comparative scientific claim under a two-tier numerical uncertainty gate (`SUPPORTED`, `REFUTED`, `INCONCLUSIVE`, `UNSUPPORTED`) |
| `export_gds(...)` | Exports photonic device geometry to `.gds` layout with PDK design-rule checking (DRC) and physical readback verification |
| `export_evidence_bundle(...)` | Packages experiment runs, lineage records, and verification certificates into a portable RO-Crate 1.1 archive |

**Design rule: the model proposes, the harness disposes.** No tool calls a model, and `ok` is
derived strictly from the process exit status.

---

## Implemented Capabilities in `v2.0.0`

**Declared and hardware-verified capabilities (through `LEG 57`):**

- **Zero-trust execution & post-flight notary (`LEG 43`, `LEG 54`, `LEG 56`):** Exclusive execution lock, source snapshotting, manifest generation, solver telemetry cross-checking, and deterministic post-flight verification (`checks/post_notary.py`, returning `exit 66` on tampering or contract violations).
- **Five-tier physical verification hierarchy (`V1`–`V5`, `LEG 44`–`LEG 48`):**
  - `V1`: Closed-form analytical benchmarks (vacuum propagation, Fabry-Perot slab, dispersive reflection, waveguide mode launch);
  - `V2`: Passivity and closed-box energy conservation verification;
  - `V3`: Multi-rung mesh convergence ladders and asymptotic error estimation;
  - `V4`: Cross-solver reconciliation against independent open-source electromagnetic solvers (**Meep** FDTD and **TORCWA** RCWA);
  - `V5`: Automated unit and power normalization verification.
- **GPU execution, multi-instance batching & crash-safe checkpointing (`LEG 45`, `LEG 46`, `LEG 51`, `LEG 52`):**
  - GPU-accelerated forward and adjoint execution with empirical hardware characterization across grid scales, graph capture widths, and batch sizes;
  - Shared-topology multi-structure batched forward and differentiable adjoint evaluation with automated checkpoint memory-budget management;
  - Crash-safe checkpointing with bit-identical mid-run restart.
- **Calibrated mode overlap, fabrication tolerance & differentiable inverse design (`LEG 49`, `LEG 53`):**
  - End-to-end differentiable photonic inverse design with minimum feature-size constraints and independent forward re-simulation of both continuous and binarized structures;
  - Unit-power-calibrated vector mode overlap evaluation and continuous fabrication etch-bias ($\pm \Delta w$) sensitivity evaluation.
- **Scientific claim verification & multi-backend experiment management (`LEG 54`, `LEG 55`):**
  - Two-tier numerical uncertainty budgeting (`estimated` vs. `certified`) and automated 4-state scientific claim adjudication;
  - Solver-agnostic experiment representation, crash-safe experiment lineage tracking, and standard RO-Crate 1.1 research bundle export.
- **GDSII / PDK layout delivery & physical readback verification (`LEG 56`):**
  - SOI layer-stack mapping, design-rule checking (DRC), `.gds` layout export/import, and independent physical FDTD equivalence verification on readback layout polygons.
- **Isolated-evaluator scientific falsification benchmark (`LEG 57`):**
  - 24-task positive/negative control benchmark suite with an isolated evaluator measuring empirical False Verification Rate (FVR) and ablation baselines.

**Scope boundaries (what this platform does not claim):**

- **Truncated vertical windows on graded meshes (`LEG 47 M4`, `partial`, `exit 57`):** When a vertical simulation window truncates a substrate interface, high-aspect-ratio graded meshes can exceed tight reflection thresholds (`exit 57`); full-stack windows are required for sub-$10^{-3}$ reflection accuracy.
- **Hardware-isolated cryptographic non-repudiation against a privileged local writer:** Local SHA-256 and HMAC checks catch metric spoofing and single-file tampering, whereas defending against a process with full write access to the run directory requires an external signing control plane.
- **A GUI CAD suite or hosted cloud service:** This repository is a code-first, agent-native simulation and verification platform.

---

## Exit statuses you will meet

| code | meaning |
|---|---|
| `0` | all pre-declared acceptance criteria and post-flight notary checks passed |
| `53` | a pre-declared stop condition fired — a lawful scientific outcome |
| `54` | all executed criteria hold, verdict `partial` |
| `55` | a declared assumption cannot be relaxed; stopped rather than approximated |
| `56` | a control case did not reproduce its inputs; comparison invalid |
| `57` | measured quantity violates its threshold — a protected **finding** |
| `58` | a realised resource bound was exceeded |
| `59` | out of memory; stops cleanly without altering solver parameters |
| `61` `62` `63` | criterion violated / negative control missing or vacuous / no criteria block |
| `64` `65` `69` | usage / pre-flight refusal / interpreter missing — **not** an executed run |
| `66` | **post-flight notary refusal**: telemetry mismatch, physics violation, or unverified claim/delivery artifact |

---

## Layout

| path | contains |
|---|---|
| `src/` | simulation drivers, inverse design, experiment engine, claim gate, GDS delivery, platform modules |
| `checks/` | pre-flight validators, post-flight notary (`post_notary.py`), and unit/adversarial test suites |
| `benchmarks/` | isolated scientific falsification benchmark tasks and harness |
| `configs/` | declarative run configurations |
| `tools/` | controlled launcher (`run.sh` / `launch.py`) and environment bootstrap scripts |
| `mcp_server/` | MCP server (stdio + streamable HTTP) |
| `mcptests/` | MCP server automated test suite |
| `docs/` | long-form evidence records per leg |
| `runs/ledger.jsonl` | tracked ledger of executed runs and exit statuses |
| `PROVENANCE.md` | external references, literature citations, and pinned solver commits |

## Licence

MIT — see `LICENSE`. **TorchFDTD** is a separate MIT-licensed project used as a dependency; its
copyright stays with it. No vendor-proprietary material or binary is required.
