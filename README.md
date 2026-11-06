# AgentFDTD (`agent-fdtd`) — v1.0.1

**An agent-built, machine-verified FDTD simulation and photonic inverse-design platform.**
AI proposes a simulation; a process exit status and deterministic post-flight notary decide whether it passed.

[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)
[![Version: 1.0.1](https://img.shields.io/badge/version-1.0.1-blue.svg)](pyproject.toml)

---

## Why this exists

Photonic design runs on simulation, and simulation results are famously easy to produce and
famously hard to trust. A transmission number comes out of a solver, goes into a paper or a
tape-out review, and the question nobody can answer six months later is: *what exactly was
checked, and against what?* Commercial tools are closed, so the answer is usually "the vendor
says so". Open tools are open, but verifying a result still means writing a one-off comparison
script that is thrown away afterwards.

This project started from a grating-coupler problem — the kind of structure where a mode has
to be launched into a sub-micron periodic region, the mesh has to grade, and the answer has to
survive a reviewer asking "how do you know". The author is an optical engineer (PhD, grating
couplers by trade) who got tired of re-deriving the same confidence from scratch for every
design.

So the platform inverts the usual order. **Nothing is claimed without a criterion that can
fail.** Every capability ships with:

- a **pre-declared** criterion and a **pre-declared exit status**, written down before the run;
- a **negative control** — a case constructed so the check *must* fail, because a gate whose
  failure mode has never been produced is a description, not a gate;
- a **ledger row** recording what ran, against which solver commit, with which exit;
- a **long-form evidence document** stating the numbers, the commands, and — explicitly —
  what the leg could **not** determine.

And the whole thing was built the same way it is meant to be used: by AI agents working under
that contract, with an independent reviewer from a different model family checking each batch,
and a human deciding only the four things a human should decide.

That is the unusual part. Not "AI wrote some code" — plenty of projects can say that — but
**an agentic workflow whose acceptance is machine-checkable and re-runnable by a stranger.**
If you do not believe a number in `docs/`, you can re-run its acceptance and watch the exit
status yourself.

> **On novelty, carefully.** We are not aware of another open-source FDTD platform that ships
> its own acceptance contract per capability, with negative controls and a re-runnable exit
> status for each. That is a statement about what we found, not a survey of the field — treat
> it as an invitation to send a counterexample, not as a priority claim.

---

## Quick start (One-command Agent & Developer Setup)

```bash
git clone https://github.com/nillinjay/agent-fdtd
cd agent-fdtd
./tools/setup_agent.sh      # creates .venv, links/initialises TorchFDTD v1.1.7 solver, installs MCP server, runs spine check
```

If you prefer manual setup with an external TorchFDTD checkout:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Link an external TorchFDTD checkout (or let setup_agent.sh use bundled vendor/torchfdtd):
git clone https://github.com/<torchfdtd>/torchfdtd /opt/torchfdtd
tools/bootstrap_solver.sh /opt/torchfdtd      # links it, then imports it to prove it works
```

Run something, and read the exit status — not the log:

```bash
./tools/run.sh configs/selftest-spine.json src/selftest_spine.py    # exit 0 expected
.venv/bin/python -m unittest discover -s checks                     # ~400 tests, seconds
```

---

## Drive it from an AI client (MCP + `SKILL.md`)

The repository ships a root `SKILL.md` (Agent Skill manifest) and `.mcp.json` (zero-config MCP
discovery), so an AI coding agent (Claude Code, Antigravity, Cursor, OpenCode, Codex) that clones
this project immediately mounts the **11 MCP tools**:

```bash
pip install -e .            # installs the `agent-fdtd-mcp` (and `torchfdtd-platform-mcp`) entry points
agent-fdtd-mcp                       # MCP over stdio (local clients)
agent-fdtd-mcp --http --port 8787    # streamable HTTP (remote clients)
```

| tool | does |
|---|---|
| `create_project(...)` | declaratively creates a 2D/3D optical simulation scene (`configs/projects/<id>.json`) with an auto-synthesized Poynting conservation box |
| `validate_project(...)` | multi-tier pre-flight validation (`dt_fs`, total Yee cells, memory admission, Poynting box closure) |
| `submit_run(config, dry_run=True)` | `dry_run` returns the exact command and pre-flight verdict; otherwise runs it and returns **`{ok, exit_status, run_id, kind}`** |
| `resume_run(run_id, dry_run=True)` | bit-identically resumes a checkpointed run from its dual-slot `ResidentRestartJournal` |
| `inspect_breakpoint(run_id)` | inspects checkpoint journal, completed/total steps, dual ping-pong slot integrity (`slot_a`, `slot_b`), and lock status |
| `get_certificate(run_id)` | verifies and returns the post-flight notary certificate (`verification_certificate.json`) with SHA-256 telemetry/readout cross-check |
| `list_configs()` | what can be run, one line each |
| `describe_run(config)` | the driver, the solver role, and the leg's pre-declared criteria |
| `validate_config(config)` | the project's own validators → `{ok, errors[]}`; runs nothing |
| `read_ledger(limit)` | the last N ledger rows, verbatim |
| `explain_exit(code)` | the exit-code vocabulary, including "this is a finding, not a crash" |

**The design rule is one sentence: the model proposes, the harness disposes.** No tool calls a
model, and `ok` is derived from the process exit status. A run that returns `57` — a measured
violation of a pre-declared threshold — comes back as `ok: false, kind: "finding"`. It cannot
come back as a success, because the tool would have to lie about an exit code to do it.

MCP is an open protocol with no vendor tie: the stdio transport works with local clients, and
the HTTP transport exists because **hosted** clients that support remote MCP servers — OpenAI's
Responses API among them — cannot spawn a local process. `pyproject.toml` declares a
`[project.optional-dependencies] mcp` extra; the server has **no model SDK dependency** of any
kind, so it stays provider-neutral.

---

## What 1.0.1 claims, and what it does not

Understating the scope is deliberate: an over-claimed release is worse than a narrow honest one.

**It claims (verified through `LEG 50` + `TorchFDTD v1.1.7` on RTX 5070 Ti `sm_120` GPU):**

- **Zero-trust execution spine (`./tools/run.sh`, `LEG 43`):** exclusive lock, source snapshot, manifest, `effective.json`, kernel-emitted `solver_telemetry.json`, deterministic post-flight integrity & physics notary (`checks/post_notary.py`, SHA-256 telemetry-readout cross-check and optional HMAC-SHA256 signing, exit `66` on any tampering or physics violation), and append-only `runs/ledger.jsonl`.
- **Verification hierarchy (`V1`–`V5`):**
  - `V1` closed-form analytic cases (vacuum propagation, Fabry-Perot slab, dispersive Lorentz reflection, waveguide mode launch);
  - `V2` passivity and 4-surface/6-surface closed Poynting box conservation (`box_closure_residual <= tau_box`);
  - `V3` monotone 3-rung 3D waveguide convergence ladder (`LEG 47 M1`: $T = [0.645227, 0.574380, 0.569926]$, Richardson order $p = 3.9916$; $n_{\text{eff}} = [2.107171, 2.077889, 2.065529]$, $p = 1.2443$);
  - `V4` dual-track cross-solver reconciliation (`LEG 48`: TorchFDTD vs. open-source **Meep 1.34.0** FDTD with `max_discrepancy = 9.04e-5 <= 0.015`, and vs. **TORCWA RCWA** rigorous coupled-wave analysis with `4.12e-3 <= 0.025`);
  - `V5` unit/normalisation ledger and coverage inventory.
- **Closed-loop differentiable inverse design (`LEG 49` + `TorchFDTD v1.1.7`):**
  - 3-tier `M3` conic spatial filtering, smooth $\tanh$ continuation projection, and discrete morphological opening/closing (`src/leg49_m3_constraints.py`, `src/leg49_inverse_design.py`) with projected Adam + genuine Armijo backtracking line search;
  - End-to-end FDTD objective directional Taylor finite-difference verification (`|dJ_FDTD/d(rho) * v - FD| <= 1e-6`, measured `4.23e-9`) alongside analytic filter-projection Jacobian verification (`1.07e-9 <= 1e-6`);
  - Independent FDTD re-simulation of both the continuous projected structure (`0.8537 -> 0.8965`, `+0.0428`) and the realized discrete `M3`-projected binary grating (`0.7019 -> 0.7335`, `+0.0317`, `0` violations at `>= 150 nm`);
  - Exact time-reversible Lorentz dispersive adjoint gradients (`torchfdtd/reversible_lorentz.py` + raw CUDA C++ kernels) and discrete FDTD material dispersion fitting (`fit_discrete_lorentz` in `torchfdtd/material_fit.py`).
- **Crash-safe checkpointing & GPU acceleration (`LEG 45` & `LEG 46`):** dual-slot `ResidentRestartJournal` with bit-identical (`0.0` field/DFT diff) mid-run resume, and fused CUDA/CuPy kernels delivering `1.70x–2.08x` GPU speedup on RTX 5070 Ti while maintaining float64 bit-identity (`<= 1.04e-33`).
- **Declarative 11-tool MCP Server v2 (`LEG 50`):** `create_project`, `validate_project`, `submit_run`, `resume_run`, `inspect_breakpoint`, `get_certificate`, `list_configs`, `describe_run`, `validate_config`, `read_ledger`, and `explain_exit`.

**It does not claim:**

- **Full-window graded-mesh modal launch on truncated vertical windows (`LEG 47 M4`, `partial`, `exit 57`).** On the tight `N8` vertical window (`y in [-1.5, +1.5] um`), the Silicon substrate interface at `y = -2.0 um` lies outside the domain, so the 17,347-step graded-mesh collar shifts reflection by `|\Delta R|/R = 2.13% > 0.1%` (`exit 57`). Extending the vertical window to cover the full stack (`y in [-2.6, +2.25] um`) reduces the 17,347-step GPU error to `|\Delta T|/T = 1.64e-4 < 1e-3` and `|\Delta R|/R = 5.90e-4 < 1e-3`.
- **Hardware-isolated cryptographic non-repudiation against a privileged local writer.** `checks/post_notary.py` cross-checks kernel telemetry against driver readouts via SHA-256 digests (and optional `TORCHFDTD_NOTARY_SECRET` HMAC-SHA256) to catch driver hallucination, metric spoofing, or single-file tampering; full isolation against a process with write access to the run directory requires an external control plane (e.g. Helix) holding a separate signing key.
- **A CAD GUI or hosted cloud service.** This is a code-first, agent-native simulation and verification harness.

`STATUS.md` is the derived, current position; `plan.md` is the internal contract and decision
log (deliberately **not** published in the standalone distribution — `docs/` carries the complete long-form evidence).

---

## Exit statuses you will meet

Exit codes are chosen by the driver and pre-declared per leg; the launcher only forwards them.

| code | meaning |
|---|---|
| `0` | the criteria that command carries hold |
| `53` | a pre-declared stop condition fired — a lawful scientific outcome |
| `54` | all criteria true, verdict `partial` |
| `55` | a declared assumption cannot be relaxed; stopped rather than approximated |
| `56` | a control case did not reproduce its inputs; the comparison is invalid |
| `57` | the measured quantity violates its threshold — a **finding**, and it may still be accepted as one |
| `58` | a realised (not requested) bound was exceeded |
| `59` | out of memory; the run stops and reports, and never rewrites solver source |
| `61` `62` `63` | criterion violated / negative control missing or vacuous / no criteria block |
| `64` `65` `69` | usage / pre-flight refusal / interpreter missing — **not** an executed run |

---

## Layout

| path | contains |
|---|---|
| `src/` | drivers, checks, platform code |
| `checks/` | the suite, including the negative controls |
| `configs/` | one config per leg; a run is a config plus a driver |
| `tools/` | launcher, solver bootstrap, public-tree exporter |
| `mcp_server/` | the MCP server (stdio + streamable HTTP) |
| `mcptests/` | the server's own tests — no network, no GPU |
| `docs/` | one long-form evidence document per leg — the reason to believe a number |
| `output/` | status page and maps |
| `runs/ledger.jsonl` | every run that happened, with its exit status |
| `PROVENANCE.md` | what came from where, pinned to commits |

## Reproducing a claim

Pick a leg in `docs/`, read its registration and its commands, re-run them, compare the exit
status. `runs/ledger.jsonl` indexes every run by run id, exit status, config, script and the
solver commit that was imported.

## Licence

MIT — see `LICENSE`. **TorchFDTD** is a separate MIT-licensed project used as a dependency; its
copyright stays with it. No vendor-proprietary material is included, and no vendor binary is
required to build or run this software.

---

**Author.** Optical engineer, PhD — grating couplers. Built this because the confidence
question kept costing more than the simulation.
