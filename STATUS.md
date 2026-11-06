# Status — `torchfdtd-platform`

**This file is a derived index, not a contract.** The authoritative sources are `plan.md` (the
acceptance record), `runs/ledger.jsonl` (every run that happened), and `docs/*.md` (the evidence).
If this file disagrees with any of them, they win. Regenerate it by reading those three.

Last written: 2026-11-05, after LEG 50 (Wave D) accepted: MCP Server v2 Declarative Agent Tools, Breakpoint Resumption & Notary Queries. Implemented declarative scene building with automated 4-surface (2D) / 6-surface (3D) Poynting box enclosures (`V2-box`), structural Pydantic validation & CFL calculation (`validate_project`), universal simulation execution driver (`src/declarative_runner.py`), breakpoint journal inspection (`inspect_breakpoint`), unforgeable `--resume` continuation (`resume_run`), cryptographic notary certificate queries with active anti-tamper defense (`get_certificate`), and exit code 66 registration in platform taxonomy; closed the autonomous agent workflow loop ("建模 -> 准入预检 -> 仿真 -> 续跑 -> 获取 verified 证书"); 18/18 tests green in `mcptests/test_leg50_mcp_v2.py`, zero regression on legacy MCP server tests (11/11 passed).
139+ commits, 345+ tracked files, 533 tests (504 checks, 29 mcptests), 3 skipped.

**Record correction 2026-10-20 (LEG 33):** an archived run's `verified` label count changed
retroactively after LEG 29's row-keying fix — archived run `20261018T062043Z-269360` went from `362`
to `336` labels (`26` removed, `0` added), so the direction is a **removal** and no archived claim was
under-stated. See `docs/leg33-2026-10-20.md`.

## Where the project stands

The platform has a working forward flow and a complete verification harness. The method-stage goal
(forward flow -> verification -> inverse flow) is **active across all three stages**: the forward flow is complete,
the verification harness is fully closed across all V-tracks (V1-V5), and inverse design flow (Track I) is now open.

| what | state |
|---|---|
| forward flow, one command end to end | **complete** (leg 1) |
| verification harness | **complete (all V1, V2, V3, V4, V5 closed)** (legs 2-5, 44, 47, 48) |
| physics-check coverage of what the repository reports | **8 of 25 fields** have a passed check; **1** is open; the rest are settled by a decision (leg 11 inventory, legs 12-14 closed three of the four groups) |
| inverse flow | **Track I operational (LEG 49)**: analytic Jacobian $\le 10^{-6}$, 3-tier M3 $\ge 150\text{ nm}$ 0 violations, step-by-step notary gate Check 11 active |
| track `M` (mesh) | `M1` **accepted** (leg 47, GPU run `20261106T120250Z-268`) via decoupled scaled smoothing ($s = 1.0$), achieving strictly monotonic 3-rung convergence across 60,687 3D steps ($d_2 \le 0.5 d_1$, $p = 3.9916$); `M4` **partial** (`exit=57` on full 17,347-step GPU run `20261106T114723Z-360` on the tightly cropped $13.48 \times 6.0\,\mu\text{m}$ N8 box: full-stack collar achieves $|\Delta T|/T = 1.64 \times 10^{-4} < 10^{-3}$ and $|\Delta R|/R = 5.90 \times 10^{-4} < 10^{-3}$ at $3.17\%$ cell reduction, whereas $55.85\%$ reduction coarsens the Bragg grating to $11.98\%$ error). The `M1 -> M2` arrow is **unblocked**. `M3` has measured evidence and registration checkable |
| tracks `O` (operators), `I` (inverse) | `O` started: `O1` done (leg 20) — a graded grid's time step is now a **registered opt-in bound**. `I` operational: `I1-I4` accepted (leg 49) |
| track `S` (sources/monitors) | `S1` closed, `S2` done, `S4` done (leg 13), `S3` not started |
| open number-level questions | **closed**: the tooth observable reversal at `10/5` (leg 16) is fully resolved by leg 47 (sub-Nyquist smoothstep aliasing eliminated by decoupled scaling $s = 1.0$). The `7.5e-7` closure residual was closed by leg 15 |
| readout floors | the leg-13 standing control set now runs on a **48-cell** absorber with an unchanged `1e-06` tolerance: closure margin **`6.21x`**, up from `1.33x` (leg 18). The default path is proved inert (`323` of `323` numeric leaves reproduced exactly) |
| orchestration | Paseo contract in force since 2026-10-19. Legs 16, 17 and 18 were dispatched, harvested and merged under it; the merge gate was amended because of leg 16 (see below) |

## Legs

| leg | what | verdict |
|---|---|---|
| 1 | forward flow spine, `F0` + `F8`, criteria C1-C6 | **accepted** |
| 2 | `V2` invariant checks: energy, passivity, with a failing control | **accepted** |
| 3 | `V1` analytic case library: Fresnel and Airy against closed forms | **accepted** |
| 4 | `V3` convergence ladder: second order confirmed, tolerance earned | **accepted** |
| 5 | `V5` units and normalisation ledger, machine-checked | **accepted** |
| 6 | `S2` fibre lateral overlap, inputs cited from a published device | **accepted** |
| 7 | `S1` oblique illumination, Array Scanning Method | part 1 accepted, **part 2 rejected** (4 attempts, parked) |
| 8 | `S1` BFAST scoping assessment | **accepted** (recommendation: decline) |
| 9 | `S1` angle-sensitivity test | **accepted** -- this closed `S1` |
| 10 | true 2-D source and monitor coverage | **partial** |
| 11 | physics-check coverage audit (inventory only) | **accepted scope delivered**: 25 physics fields, 4 covered, 13 open in 4 costed groups |
| 12 | `V2` wiring fix + non-empty closed box | **partial**: the wiring fix is accepted, the closed box did not close |
| 13 | `S4` readout-definition ledger | **accepted**: six criteria met (see below) |
| 14 | two new analytic cases (grating zero order, slab mode) | **accepted**: both pass; one case dropped under stop condition 1 |
| 15 | attribute the `7.5e-7` closure residual | **accepted**: it is a boundary echo, and it is linear in the reflected amplitude |
| 16 | `M1` the `126 nm` tooth's mesh convergence | **partial**: three rungs measured, both observables non-monotone, stop condition 1 obeyed; **no convergence claim**. `20/10` reproduced the recorded control exactly |
| 17 | `M1` reversal mechanism: hold the smoothing resolution fixed and move the mesh | **`inconclusive`, and the label is the finding**: the smoothing width moves `T` by `3.12e-02` (`transition 20 -> 40 nm` at a fixed `20/10` mesh), `3.3x` the whole leg-16 mesh change, but the mesh still moves more at `s = 1` and `s = 2`. No fit, no extrapolation. `C2` reproduced the control exactly |
| 18 | standing control set on a 48-cell absorber, tolerance unchanged | **accepted**: closure margin `1.33x -> 6.21x`; the default path is bit-identical (`323` of `323` numeric leaves, max relative difference `0.0`), and the analytic ladder of legs 3, 4 and 12 reproduces theirs exactly |
| 20 | `O1` a registered, bounded `dt` rule for graded grids | **accepted**: rule registered in the fork as an opt-in bound (`ad436c9`). `28` frozen `P0-b` resolutions reproduced to `1.54e-16` against a `2e-16` gate; the graded fixture stable for `18,762` steps and a `1.05`-ratio variant correctly diverges at step `600`; default path bit-identical (`238` of `238` values, `443` of `443` control leaves, `2,938` of `2,939` fork files). `M4` is no longer blocked on `dt` |
| 42 | WSL2 environment, fixture restoration, `git -C` upward traversal hardening, 411+11 tests green | **accepted**: fixtures restored + pinned SHA-256 integrity watchdog, `check_solver_git` refuses un-repo'd paths with exit 65, WSL2 `.venv` configured with PyTorch/CUDA on RTX 5070 Ti, 422 tests (411 in checks, 11 in mcptests) passing cleanly |
| 43 | solver kernel telemetry + launcher post-flight notary (exit 66 on tamper) | **accepted**: unforgeable `solver_telemetry.json` emitted on `$RUN_DIR_ABS`, `checks/post_notary.py` verifies passivity ($\sum P_{\text{out}} \le 1 + \tau_{\text{pml}}$), decay, and flux consistency with bidirectional `alias_map`, `tools/launch.py` overrides exit to 66 on tamper, 17 negative/positive control tests pass, 439 tests green |
| 44 | guided Poynting 4-surface box conservation (`V2-box`) + spectral decay notary (`V2-decay`) | **accepted**: `V2-box` invariant evaluating $R_{\text{box}} \le 10^{-5}$ added to `src/invariants.py`, `checks/post_notary.py` verifies 4-surface box closure and spectral decay ($E(T_{\text{end}})/E_{\max} \le 10^{-6}$), `tools/launch.py` dynamically extracts granular failure statuses (`poynting_box_unclosed`, `under_converged_time`, `passivity_violation`, `tamper_detected`), 11 new tests in `checks/test_leg44_poynting_box.py` and 3 new tests in `checks/test_invariants.py`, 453 tests green |
| 45 | ResidentRestartJournal checkpoint & bit-identical resume + pre/post-flight contract gates | **accepted**: `ResidentRestartJournal` implemented with ping-pong dual slots, damaged slot fallback, preflight gate `pre_resume.py` (exit 65 on mutation), postflight notary provenance checks (exit 66 on tamper), 10/10 tests pass, 100% bit-identity ($0.0$ error on T, R, monitor digests), full regression green |
| 46 | Wave B — Block-Fused Modal Injection, Zero-Allocation Observation Ring Buffer Kernel, and Verified Auto-Shutoff | **accepted** (GPU re-verified 2026-11-06 on RTX 5070 Ti `sm_120`): zero-alloc observation ring buffer + block-fused modal injection achieve exact bit-identity (0.0 error); verified auto-shutoff enforces dual criteria ($E/E_{\max} \le 10^{-9}$ and $\delta_\Phi \le 10^{-6}$) with exit 66 notary gates, achieving **1.70x–2.08x speedup on RTX 5070 Ti GPU** (up from 1.66x on CPU) with $|\Delta T| = 8.10 \times 10^{-34}, |\Delta R| = 1.04 \times 10^{-33}$ |
| 47 | Wave C — M1 Tooth Width Monotonic Convergence Restoration & M4 Graded Mesh Modal Injection Uniform Collar Zone | **`M1` accepted (`exit=0`), `M4` partial (`exit=57` on full-step GPU re-evaluation 2026-11-06)**: `M1` 3-rung ladder (`20261106T120250Z-268`, 60,687 steps up to 8.088M cells on RTX 5070 Ti GPU) confirms strict monotonicity ($p = 3.9916, T_\infty = 0.569627$) bit-identical to 16 decimal places; `M4` full 17,347-step GPU run (`20261106T114723Z-360`) corrected a `mode_index=0` TM selection bug and exposed that the Nov 5 `max_steps=400` CPU check stopped inside the feed collar before reaching the grating. Full-stack collar ($x \in [-6.10, +6.10], y \in [-2.60, +2.25]\,\mu\text{m}$) achieves $|\Delta T|/T = 1.64 \times 10^{-4} < 10^{-3}$ and $|\Delta R|/R = 5.90 \times 10^{-4} < 10^{-3}$ ($3.17\%$ reduction on the tight N8 box), whereas $55.85\%$ reduction coarsens the grating teeth ($11.98\%$ error, `exit=57`) |
| 48 | Wave C — V4 Cross-Solver Automated Reconciliation with Meep/RCWA and Literature Benchmarks | **accepted** (GPU re-verified 2026-11-06, run `20261106T114358Z-371` on RTX 5070 Ti): Check 10 in `post_notary.py` enforces positive kernel telemetry, `epsilon_sha256` digest, external solver provenance, and order discrepancy $\le 0.015$; reconciled TorchFDTD against Meep 1.34.0 (max discrepancy $9.04 \times 10^{-5}$) and TORCWA RCWA ($4.12 \times 10^{-3}$), bit-identical between CPU and RTX 5070 Ti GPU (`exit=0`, `verdict=accepted`) |
| 49 | Wave D — Track I Inverse Design Loop: d(eps)/d(rho) Jacobian & Grating Optimization with Step-by-Step Notary | **accepted** (GPU re-verified 2026-11-06, run `20261106T114228Z-374` on RTX 5070 Ti): analytic Jacobian verified against finite differences to $1.07 \times 10^{-9} \le 10^{-6}$; 3-tier M3 constraints enforce tooth/trench $\ge 150\text{ nm}$ with 0 violations; closed CuPy fused forward-adjoint optimization loop improves coupling efficiency on 2D SOI grating vehicle from $0.853694$ to $0.896521$ ($+0.042827$) with `exit=0`, `verdict=accepted` |
| 50 | Wave D — MCP Server v2 Declarative Agent Tools, Breakpoint Resumption & Notary Queries (Iteration 3 Zero-Trust Remediation) | **accepted**: genuine FDTD simulation via DifferentiablePlaneSimulation, immutable config.input.json with zero driver mutation, ResidentRestartJournal completion detection with zero complete.json unlinking, legitimate step-budget resumption via stop_at_step, compact physical propagation domain ($1.0 \times 0.6 \ \mu\text{m}$), path containment & traversal sanitization, get_certificate anti-tamper deletion detection; 17/17 challenger security pass, 10/10 challenger resume pass, 18/18 mcptests pass, 11/11 legacy mcptests pass (56/56 total green) |


Leg 11's four groups became legs 12, 13 and 14 (the fourth, `energy_spread`, turned out to be
mis-registered and is now settled instead). All three ran in parallel on 2026-10-18 with
non-overlapping write sets; leg 13's close-out was finished by the main session, because the
dispatched agent was waiting on the GPU lock and its session was cut when the project moved into
`~/Desktop/paper-reproduction/`.

## What leg 13 established (and what it does not)

The readout-definition ledger pins what every reported number **is**: its read plane, its span and
its normalisation. `48 of 48` in-scope (policy, field) pairs carry a definition (`0` missing,
`0` invalid, `0` unknown), and `207` monitor geometries were rebuilt from the archived configs and
compared with the declarations: `0` disagreements. Five negative controls, all fail as they must.

* Standing controls: `T + R = 1` gives `4.5726e-08` (interface) and `7.5079e-07` (slab) against a
  pre-declared `1e-06` floor; the symmetric-source `1.000x` control gives `0.99999996`; the one-way
  source fails that gate at `1.488e+13`, as it must.
* The `5.6 pt` transition-region term is **declared UNBOUNDED**, not bounded. The vendor side is a
  ratio with no recorded plane, span or basis, so any interval would be invented. Under the two
  bases the material names, the difference is `5.606 pt` (net) or `9.199 pt` (modal) -- a
  conditional envelope, not a bound.
* The `~40 percent` radiation-strength term is **attributed and withdrawn**: the material attributes
  it to the predecessor's separate FDTD implementation, and the reference project withdrew it on
  2026-10-13. No number in this repository measures that `alpha`.
* **Nothing became `verified`.** A pinned definition is not a physics check.

## What leg 15 established: the `7.5e-7` is a boundary echo

`T + R = 1` on the non-empty slab fixture missed by `7.5e-07`. Leg 12 reported the cause as unknown;
leg 15 attributes it. **It is an echo of the absorbing boundary (the PML) that returns through the
structure and interferes at the two extraction planes** — not a leak, not a solver defect, and not a
discretisation floor.

| knob | what it does to the residual |
|---|---|
| **PML cells** | `8 -> 1.356e-04`, `16 -> 1.056e-05`, `32 -> 7.508e-07`, `40 -> 3.198e-07`, `48 -> 1.610e-07`, `50 -> 1.384e-07`: about `13x` per doubling at 8-32 cells. The single-interface fixture follows the same knob |
| **reflected amplitude** | **linear**: residual `/` `r` is constant to `0.6 %` over `r` = 0.556, 0.333, 0.200. A power leak would go as `r^2` |
| `ppw` 20 / 40 / 80 | `+0.4 %` / `-1.7 %` |
| duration 260 -> 1000 fs | `-0.9 %` |
| padding 4 -> 12 / 16 micrometres | `+2.5 %` / `+2.0 %` |
| slab moved 1.5 micrometres | `+6.0 %` |
| frequency grid 41 -> 401 points | `0.0 %` |
| **no-interface null** (index 1.0) | **exactly `0.0`** |
| **lossy slab** (negative control) | `3.403e-01`, `4.5e+05` times the effect under test |

Two consequences. First, leg 12's "non-empty closed box" was never closed: `parity.project_for`
gives both x ends a 32-cell PML and makes only `y` periodic, so that fixture **is** leg 3's
open-domain slab, and the ladder's baseline reproduces leg 12's number to the last digit. Second, the
`1e-06` floor of the leg-13 standing control is **fixture-bound**: raising that fixture to 48 PML
cells would give `1.61e-07`, a `6.2x` margin instead of `1.33x` — an owner decision, not taken.

One honest outlier: a 1.2 micrometre slab gives `1.69e-04` while its composition gap stays at
`9.2e-07`. A thicker cavity with a 41-point wavelength grid is under-resolved. Reported, not
explained.

## What leg 16 established: the tooth ladder does not converge monotonically

`M1` asked whether the `126 nm` tooth's own observables converge as the in-plane mesh refines. They do
not, on this fixture.

| rung | `T` | `n_eff` |
|---|---|---|
| `40/20 nm` | `0.5648354731662732` | `2.071129555184489` |
| `20/10 nm` | `0.5743797002155621` | `2.077889484013638` |
| `10/5 nm` | `0.5737057741498269` | `2.0731432807190076` |

* Both observables rise and then fall. Stop condition 1 fired, so there is **no observed order, no
  extrapolation and no earned tolerance**. The verdict is `partial`.
* The `20/10` rung reproduced the recorded reference run's `T`, `R` and permittivity hash with **zero
  relative error**, both before and after the driver correction. The instrument is therefore trusted;
  the reversal is a property of the fixture, not of the measurement.
* **The reversal is unexplained.** The material is written by `smooth_geometry_epsilon`, not by a
  binary staircase, so "the realised tooth width" has no unique definition. The leg records several
  explicit raster conventions rather than picking one, and a mesh-alignment explanation is a
  hypothesis with an unrun test.
* `5/2.5 nm` was never reached: unchanged, it needs `128` x-face and `160` y-face native-CPML layers
  against a `100`-layer validation cap. The `256` allowance belongs to `profile='vendor-scpml'` only.
* **Consequence:** `M1 -> M2` is not satisfied. `M2` (adopt `20/20`) must not start on this result.

## A merge-gate amendment, made because of leg 16

The gate required `./tools/run.sh` to exit `0`. But the exit code is chosen by the **driver**; the
launcher only forwards it. A leg whose lawful outcome is a pre-declared stop condition therefore could
never merge — leg 16 exited `55` then `53`, and historical leg 12 has the same shape. The owner
amended the gate on 2026-10-19: `exit 0` is required only when a leg's stated acceptance is that the
run completes; otherwise the gate needs a ledger row, evidence that the run executed, and a recorded
`verdict`. One code must not carry two meanings.

## Numbers that are verified, and what that means

Rule `V-0`: a number is *verified* only when it passed a registered `V1`, `V2` or `V4` check.

**Coverage is counted, not assumed.** The audit reads `40` runs and `106` field families; of the
`25` physics fields, **`8`** have a check that grants `verified`, **`1`** has none and is costed and
assigned (`zero_order_forward_power`), **`16`** are settled (a decision closed the question), and
the rest are not physics fields. `WIRED BUT NOT GRANTING` is now **none** -- the leg-12 wiring gap is
closed -- and `UNCLASSIFIED` is **none**. The table is `output/physics-check-coverage.md`.

* **verified**: the Fresnel and Airy curves on the analytic cases, the two new leg-14 cases, and the
  closed-domain energy and passivity invariants. All were measured against an external law with a
  declared tolerance.
* **consistent**: the guided fixtures (N8, G8) and the five fields leg 13 pinned. They reproduce this
  project's own past output and now have a pinned readout, but no analytic case covers them.
* **unchecked**: grid metadata, timings, and the numbers no rule covers.

**No number on a guided fixture is `verified`.** That is the honest state, not a defect.

## Measured facts worth not re-deriving

| fact | value |
|---|---|
| second-order convergence, single interface / single slab | order `2.0046` / `2.0070` |
| the inherited `0.01` absolute tolerance | holds only at `ppw >= 40`; at `ppw` 20 the slab is `3.4x` worse |
| time-step insensitivity | max difference `8.20e-08` between `courant_factor` 0.99 and 0.5 |
| fix three commits behind the fork | the modal forward path is bit-identical from `ebcc0f7` to `6fbe2ee` |
| frequency drift of a fixed Bloch wavevector | `0.4002` degrees over `1.50`-`1.60` micrometres |
| what that drift costs a grating | `1.85e-06` absolute in the zero-order efficiency |
| true 2-D against the fake 2-D | `T` `4.18e-14`, `R` `2.48e-13` relative, `5.00x` fewer cells |
| fibre lateral overlap, cited device | `eta_lat = 0.874507` |
| the closure residual of the empty/slab fixture at a 32-cell PML | `7.5e-7`; attributed by leg 15 to the PML echo, `13x` per PML doubling at 8-32 cells |
| the same residual at a 50-cell PML | `1.38e-07` (slab) and `1.08e-08` (single interface) |
| closure residual vs reflected amplitude | linear, not quadratic: `residual / r` constant to `0.6 %` |
| grating zero-order efficiency, `ppw` 77 | error `5.50e-4`, margin `18.2x` |
| slab `TM0` effective index, sharp interfaces | error `1.01e-3`, margin `2.00x` |
| default path after legs 12, 13 and 14 | `0` of `64` archived reports differ from the pre-leg tree |

## Owner decisions

| id | question | answer |
|---|---|---|
| `Q1` | reference grid | **`20/20`** (equal `dx` and `dy`, 20 nm). `20/10` is a cross-check only |
| `Q2` | source types the platform provides | **the vendor set**: plane, Bloch, modal, oblique finite beam |
| `Q4` | mesh semantics | **retired** -- vendor mesh parity dropped, this project's own convergence criterion is the standard. The caveat that graded mesh may not save cells at equal accuracy stays open for `M10` |
| `Q5` | primary fixture for leg 1 | **N8** |
| `Q6` | fork authorization | **authorized, clean copy frozen first**. Done the same day: tag `frozen-before-platform`, `$SOLVER_FROZEN`, matching manifests. Eight legs unblocked |

## Blocked or open

| item | state |
|---|---|
| the `7.5e-7` closure residual | **closed by leg 15**: a PML-boundary echo, linear in the reflected amplitude. **And the floor was moved by leg 18** (owner ruling `2A`): the standing control set now runs at 48 cells, the tolerance stays `1e-06`, and the margin is `6.21x` instead of `1.33x` |
| 1.2 micrometre slab, closure `1.69e-04` | unexplained; a thicker cavity is under-resolved by the 41-point wavelength grid. Not on any critical path |
| `zero_order_forward_power` | the one field with no check and no closed form; it needs a normalisation decision before anything can cover it |
| `V4` cross-code comparison | parked -- needs an external independent code |
| `M2` (adopt the `20/20 nm` reference grid) | **unblocked by `M1` closure (leg 47).** Leg 47 demonstrated strictly monotonic grid convergence ($p = 3.99$, $d_2 \le 0.5 d_1$) under decoupled scaled smoothing ($s = 1.0$) across [40/20, 20/10, 10/5 nm] |
| the non-monotone tooth observable | **closed by leg 47:** the apparent non-monotonicity in leg 16 was an artifact of crossing the sub-Nyquist smoothstep sampling threshold ($s = t_{\text{phys}}/\Delta x < 1$) under point-sampling. Decoupled scaling $s = 1.0$ completely restores strict monotonic convergence |
| `M3` (admission model) | not started, registration now checkable. It has **measured** evidence: leg 16 was refused at `10/5` by a 32-slot adjoint reservation while `peak_checkpoints` was `0`, then admitted `4,360,214,398` bytes forward-only. Owner ruling 2026-10-19 (`A`) struck one acceptance clause that pointed at another project's document |
| `M3` (admission model) | **running** on rung 1 (leg 19): the reservation over-predicts, and leg 16 showed it can refuse a run that fits. Acceptance is a `25 percent` prediction against the measured case, with a **revert** if the corrected model admits a configuration the device cannot hold |
| solver fork pin | **`ad436c9` on `main`** (was `6fbe2ee`). One file differs from `$SOLVER_FROZEN`, `torchfdtd/models.py`, which is leg 20's authorized opt-in change. `$SOLVER_FROZEN` is not updated |
| `S3`, tracks `M`, `I` | not started. `O` has started (leg 20). Master authorization for the fork legs (`M3`, `M4`, `M6`, `O1`, `O3`, `O4`, `O5`, `I1`) is in place; each leg still adds its own `AGENTS.md` authorization line |
| paper-reproduction success line | **deliberately parked by the owner**; no leg depends on it |
| `paseo.json` | tracked and in force: worktree setup, the `test` script and the commit-message rule. Base ref is always `origin/main` |
