# torchfdtd-platform

**An agent-built, machine-verified FDTD platform.**
AI proposes a simulation; a process exit status decides whether it passed.

[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)

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
git clone https://github.com/<you>/torchfdtd-platform
cd torchfdtd-platform
./tools/setup_agent.sh      # creates .venv, links/initialises TorchFDTD solver, installs MCP server, runs spine check
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
this project immediately mounts the **10 MCP tools**:

```bash
pip install -e .            # installs the `torchfdtd-platform-mcp` entry point
torchfdtd-platform-mcp                       # MCP over stdio (local clients)
torchfdtd-platform-mcp --http --port 8787    # streamable HTTP (remote clients)
```

| tool | does |
|---|---|
| `create_project(...)` | declaratively creates a 2D/3D optical simulation scene (`configs/projects/<id>.json`) with an auto-synthesized Poynting conservation box |
| `validate_project(...)` | multi-tier pre-flight validation (`dt_fs`, total Yee cells, memory admission, Poynting box closure) |
| `submit_run(config, dry_run=True)` | `dry_run` returns the exact command and pre-flight verdict; otherwise runs it and returns **`{ok, exit_status, run_id, kind}`** |
| `resume_run(run_id, dry_run=True)` | bit-identically resumes a checkpointed run from its dual-slot `ResidentRestartJournal` |
| `get_certificate(run_id)` | verifies and returns the cryptographic post-flight notary certificate (`verification_certificate.json`) |
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

## What 1.0 claims, and what it does not

Understating the scope is deliberate: an over-claimed 1.0 is worse than a narrow honest one.

**It claims:** a forward pipeline from structure to report through one launcher (lock, source
snapshot, manifest, effective config, ledger row); a verification harness with closed-form
analytic cases, a convergence ladder, energy and passivity checks, a unit and normalisation
ledger and a coverage inventory; self-checks including the negative controls above; and an
admission model that predicts device-memory fit **before** a run, gated on the **reserved**
peak with the allocated peak reported beside it.

**It does not claim:**

- **Inverse design.** Registered, not started. No working adjoint optimisation loop here.
- **Mesh convergence for the documented device.** The ladder came back **non-monotone**: one
  observable rose then fell under refinement. No fit, no extrapolation, **no convergence
  conclusion** — and that is the recorded result, not a gap we hid.
- **Graded-mesh accuracy for the modal launch.** An opt-in path launches a mode on a
  non-uniform grid. Default path is bit-identical to the baseline; the graded path executes but
  its accuracy criterion **fails** — transmission off by ~`0.6 %`, reflection by ~`2.2 %` after
  one refinement control, against a `0.1 %` threshold. Residual cause not isolated. Treat as
  **experimental**.
- **Cross-code validation.** Parked; needs a second code and a matched fixture.
- **A production-grade UI or cloud service.** This is a harness you run yourself.

`STATUS.md` is the derived, current position; `plan.md` is the internal contract and decision
log (deliberately **not** published — the evidence documents carry what matters).

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
