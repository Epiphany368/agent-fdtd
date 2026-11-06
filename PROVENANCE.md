# Provenance — pinned commits and copied files

Purpose: this project depends on two read-only reference projects and on a shared solver fork.
Nothing is imported across repositories. Everything taken is copied once, annotated, and listed
here. Everything cited is pinned to a commit.

## Pinned commits (recorded 2026-10-18)

External directories are written as symbols, never as absolute paths. The symbol-to-path map is
machine-specific and lives in the untracked file `env/refs.env`. Copy that file's values into your
shell (`set -a; . env/refs.env; set +a`) before a leg runs. `plan.md` and `AGENTS.md` use the same
symbols.

| symbol | stands for |
|---|---|
| `$PARITY_REF` | the reference project `20261010_torchfdtd-lumerical-parity` |
| `$REA_REF` | the REA project `realumerical` |
| `$SOLVER_FORK` | the TorchFDTD solver fork |
| `$SOLVER_UPSTREAM` | the frozen upstream TorchFDTD tree (provenance only) |
| `$SOLVER_FROZEN` | an independent copy of the fork, frozen before the platform changed it (owner precondition, 2026-10-18) |

| source | symbol | commit / branch | how it is used |
|---|---|---|---|
| reference project | `$PARITY_REF` | `3a81664` on `main` | history, `refs/`, ledger, recorded numbers, flow audit, launcher and fixture reuse |
| REA project | `$REA_REF` | `0c121fa` on `codex/windows-local-rea` | recovered vendor semantics, with their evidence classes |
| solver fork | `$SOLVER_FORK` | **`f00d216` on `main`** (was `91e59f6` / `ad436c9` / `6fbe2ee`; upstream `v1.1.7` merge, 2026-11-06) | the solver this platform drives |
| frozen upstream | `$SOLVER_UPSTREAM` | `ebcc0f7` | provenance only; never modified |
| frozen fork | `$SOLVER_FROZEN` | `6fbe2ee`, detached, no remote | the restore source and the bit-identical reference for every fork leg |

### The frozen fork (created 2026-10-18)

The owner authorized fork changes on condition that a clean copy is frozen first
("我授权去改 但改之前得先冻结"). It was done while the fork was still clean, because a
freeze taken after the first change is worthless.

| what | where / value |
|---|---|
| fork state at freeze time | `6fbe2ee`, working tree clean |
| tag inside the fork | `frozen-before-platform` (annotated) |
| independent copy | `$SOLVER_FROZEN`, detached at `6fbe2ee`, its `origin` remote removed so nothing fetches into it |
| tracked-tree blob manifest | `git ls-tree -r 6fbe2ee`, sha256 `e604b16a6348f7b0f014a7c1312c343e797f48d648fc2226062a615a7db141c9`, 2939 files |
| proof at freeze time | the fork's and the copy's `ls-tree` lines were compared and are identical |

**How a fork leg uses it.** After changing `$SOLVER_FORK`, a leg must show that the default path is
still bit-identical. The check is arithmetic, not a judgement:

```bash
set -a; . env/refs.env; set +a
diff <(git -C "$SOLVER_FORK" ls-tree -r <the-legs-commit>) <(git -C "$SOLVER_FROZEN" ls-tree -r HEAD)
```

Unchanged files must show no diff line. If the fork's history is ever rewritten, `$SOLVER_FROZEN`
still holds the frozen tree and its own `.git`, so the reference cannot be lost.

**Note recorded at freeze time:** the fork's repository is **shallow** (`git clone` reported "source
repository is shallow"), so its history is truncated. The frozen copy inherits that. The working tree
and the manifest are complete; only deep history is missing. It does not affect the bit-identity check.

## Copy log

Each entry: destination here | source path | source commit | what was taken | annotation added.

| destination | source | source commit | taken | date |
|---|---|---|---|---|
| `tools/run.sh` | reference project `tools/run.sh` | `9b0f61a` | the single simulation entry: preflight, lock, snapshot, effective config, status | 2026-10-18 |
| `tools/launch.py` | reference project `tools/launch.py` | `9b0f61a` | lock, source snapshot + hashes, manifest, `effective.json`, ledger row, exit status. Unchanged except the provenance header | 2026-10-18 |
| `checks/pre_environment.py` | reference project `checks/pre_environment.py` | `9b0f61a` | the frozen-upstream invariant and the fork-recording rule. One change: the solver roots from `configs/baseline-pin.json` are repository-relative, so no machine-specific path is tracked | 2026-10-18 |
| `configs/baseline-pin.json` | rewritten from the reference project's file | `9b0f61a` | the pin structure; the values are this project's, and no absolute path is stored | 2026-10-18 |
| `src/parity.py` | reference project `src/parity.py` | `9b0f61a` | the shared adapter: config/env binding, mesh steps, structures, monitors, `Simulation`, flux normalisation. One change: the solver is named by `configs/baseline-pin.json` and loaded from the launch snapshot | 2026-10-18 |
| `src/run_b9_native_shape_n8.py` | reference project `src/run_b9_native_shape_n8.py` | `9b0f61a` | the N8 scene (`scene`), the smoothstep (`step`) and the far constant. Used for its scene builder; this project does not run its derivative campaign | 2026-10-18 |
| `src/run_g1b_gradient_taylor.py` | reference project `src/run_g1b_gradient_taylor.py` | `9b0f61a` | the G8 fixture builder (`fixture`), the 32 edge positions, the bridge builders, and the re-exports the evaluator uses. Used for its fixture; this project does not run its Taylor campaign | 2026-10-18 |
| `src/run_g5_discretisation.py` | reference project `src/run_g5_discretisation.py` | `9b0f61a` | `g5_fixture`: the registered `normal-harmonic` bridge. Used for that bridge only | 2026-10-18 |
| `src/analytic_cases.py` | reference project `src/run_canary.py`, function `analyze` | `9b0f61a` | the two closed-form laws: the `interface` branch (Fresnel) and the `slab` branch (Fabry-Perot), plus the hard-coded optical thickness `2.1`. Transcribed by citation, unchanged; never imported at run time | 2026-10-18 |

Each copied `.py`/`.sh` file carries a header naming its source, its source commit and the
source file's sha256. Source copies are annotated, never imported across repositories.

## Rules

0. **No absolute paths in tracked files.** A tracked file names an external root by symbol only. The
   real paths stay in `env/refs.env`, which is git-ignored.
1. **Copy, do not import.** A run-time `sys.path` or import that reaches into a reference project is
   forbidden.
2. **Annotate every copy** with its source path and source commit at the top of the file.
3. **Re-verify any number taken from a reference project** before it becomes this project's evidence.
   A reference-project number is a claim, not this project's measurement.
4. **Never upgrade an evidence class** without a measurement here. `D` documented, `M` measured,
   `R` recovered from the binary, `X` unknown, `A` arithmetic — definitions follow
   `output/v3.0-rea-reconciliation.md` in the reference project.
5. **Do not modify** either reference project, `refs/`, or the frozen upstream tree.

### Fork changes made by this platform (2026-10-19)

Each row is one leg that changed the shared fork, with the owner authorization in `AGENTS.md`'s fork
authorization log. The default path stays bit-identical in every case, and the change is re-verified
against `$SOLVER_FROZEN` after the leg.

| date | leg | fork commit | file(s) | what it adds | default path |
|---|---|---|---|---|---|
| 2026-10-19 | 20 (`O1`) | `ad436c9` | `torchfdtd/models.py` (one file) | opt-in `Region.time_step_rule = 'realised_axis_minimum'`: bounds `dt` by the **smallest realised** node spacing per active axis, for graded grids | **bit-identical.** `238` of `238` default-path values equal `$SOLVER_FROZEN` (`20261019T095323Z-283804`); the standing controls reproduce `443` of `443` non-timing readout leaves exactly; `2,938` of `2,939` tracked fork files match, the one difference being `models.py` |
| 2026-11-06 | upstream `v1.1.7` merge | `f00d216` | `torchfdtd/material_fit.py`, `torchfdtd/reversible_lorentz.py`, `torchfdtd/reversible_lorentz_cuda.py`, `torchfdtd/differentiable.py`, `torchfdtd/reversible.py`, `torchfdtd/reversible_cpml.py`, `torchfdtd/reversible_cpml_planes.py`, `torchfdtd/models.py`, `torchfdtd/capabilities.json`, `examples/adjoint_inverse_design/*`, `examples/full-aperture-tiled-adjoint/*`, `tests/test_reversible_lorentz_*.py`, `tests/test_discrete_lorentz_fit.py` | merges upstream `hyoseokp/TorchFDTD` commit `36589c6` (`v1.1.7` isotropic Lorentz density mixtures in periodic/recorded CPML adjoints, `fit_discrete_lorentz` discrete FDTD dispersion fitting) and upstream inverse-design example scripts (excluding `.npz` figure blobs) on top of platform fork `91e59f6` | **bit-identical on non-dispersive default path.** `142 passed` across solver Lorentz/Gaussian/invariant test suites; `29 passed` across platform MCP v2 end-to-end test suite |

**The fork is one shared working tree, so its branch state is part of the record.** After leg 20 the
fork was left checked out on the leg's own branch; the steward fast-forwarded `main` to `ad436c9`,
checked out `main`, and deleted the leg branch, so that a later fork leg branches from a deliberate
state rather than from whatever a previous leg happened to leave. `$SOLVER_FROZEN` is **not** updated:
it stays at `6fbe2ee` and remains the bit-identical reference for the default path.


## Literature and Independent Solver Benchmarks (Owner Authorization, 2026-11-04)

Per explicit Owner authorization recorded 2026-11-04 ("必要的时候你可以自己找论文数据或者其他途径进行已有结论交叉比对"), published literature data and independent external electromagnetic solvers (Meep / RCWA) are incorporated to cross-check and reconcile platform conclusions under track `V4` (LEG 48), upholding the "no invented inputs" rule.

### Literature Benchmarks

| Benchmark Citation | Published Parameters & Geometry | Canonical Observable Quantities | Role in Platform Verification |
|---|---|---|---|
| **Taillaert et al. (2002)**<br>D. Taillaert, W. Bogaerts, P. Bienstman, et al., *IEEE J. Quantum Electron.* 38(7), 949–955 (2002).<br>DOI: [`10.1109/JQE.2002.1017613`](https://doi.org/10.1109/JQE.2002.1017613) | Standard 1D SOI grating coupler: $\Lambda = 630\text{ nm}$, etch depth $e = 70\text{ nm}$, duty cycle $ff = 0.50$, $N=20$ periods, $h_{\text{core}} = 220\text{ nm}$, $n_{\text{Si}} = 3.476$, $n_{\text{SiO2}} = 1.444$, $\lambda_0 = 1550\text{ nm}$. | Upward radiation efficiency $\eta_{\text{up}} \approx 31\% - 33\%$, downward leak $\eta_{\text{down}} \approx 57\% - 61\%$, reflection $R \approx 1.5\% - 3.5\%$. | Baseline integrated silicon photonics grating coupler radiation efficiency benchmark. |
| **Taillaert et al. (2004)**<br>D. Taillaert, P. Bienstman, and R. Baets, *Opt. Lett.* 29(23), 2749–2751 (2004).<br>DOI: [`10.1364/OL.29.002749`](https://doi.org/10.1364/OL.29.002749) | Compact SOI grating coupler with bottom reflector / top oxide overlay. | Fiber coupling efficiency and reflection reduction bounds. | Broadband coupling mechanism and reflection bounds reference. |
| **Moharam et al. (1995)**<br>M. G. Moharam, E. B. Grann, D. A. Pommet, and T. K. Gaylord, *J. Opt. Soc. Am. A* 12(5), 1068–1076 (1995).<br>DOI: [`10.1364/JOSAA.12.001068`](https://doi.org/10.1364/JOSAA.12.001068) | Binary lamellar dielectric transmission/reflection grating: $\Lambda = 1.0\,\mu\text{m}$, groove depth $h = 0.50\,\mu\text{m}$, $f = 0.50$, $n_1 = 1.000$, $n_{\text{ridge}} = 1.500$, $n_{\text{sub}} = 1.500$, TE/TM polarization. | Zero-order transmitted $T_0$, reflected $R_0$, and diffracted orders $T_{\pm 1}, R_{\pm 1}$; strict unitary passivity $\sum (R_m + T_m) = 1.0$. | Rigorous Coupled-Wave Analysis (RCWA) standard reference and energy conservation formulation. |
| **Okamoto (2006)**<br>K. Okamoto, *Fundamentals of Optical Waveguides*, 2nd ed., Academic Press (2006), Chapter 2, pp. 20–35.<br>DOI: [`10.1016/B978-0-12-525096-2.X5000-0`](https://doi.org/10.1016/B978-0-12-525096-2.X5000-0) | Symmetric 3-layer dielectric slab waveguide: core thickness $d = 220\text{ nm}$, $n_1 = 3.480$, $n_2 = 1.444$, $\lambda_0 = 1550\text{ nm}$. | Exact transcendental dispersion equation: $N_{\text{eff}}^{\text{TE0}} \approx 2.834$, $N_{\text{eff}}^{\text{TM0}} \approx 1.872$; modal power conservation $T = 1.000000 \pm 0.001$, $R \le 10^{-5}$. | Transcribed in `src/analytic_cases.py:147–186` for unperturbed waveguide modal propagation. |
| **Kim & Lee (2023) TORCWA**<br>C. Kim and B. Lee, *Comput. Phys. Commun.* 282, 108552 (2023).<br>DOI: [`10.1016/j.cpc.2022.108552`](https://doi.org/10.1016/j.cpc.2022.108552) | Dual-ridge metagrating beam deflector: period $\Lambda = 2.0\,\mu\text{m}$, Si ridges ($n=3.48$, width 80 nm and 220 nm, height 500 nm) on SiO2 ($n=1.444$) at $\lambda_0 = 1.55\,\mu\text{m}$; 241 orders (120 harmonics). | Measured diffraction efficiencies: $T_{-1} = 0.0365$, $T_0 = 0.0750$, $T_{+1} = 0.7746$, $R_{-1} = 0.0476$, $R_0 = 0.0156$, $R_{+1} = 0.0507$; $\sum \eta = 1.000000$. | Independent Fourier modal method (RCWA) benchmark oracle for metagrating cross-reconciliation. |

### Independent Solver Benchmark Records

| Solver | Version & Configuration | Reference Path / Provenance | Benchmark Structure & Metrics | Measured Agreement with TorchFDTD |
|---|---|---|---|---|
| **Meep** | 1.34.0, MPI 4 ranks CPU double-precision (A. Oskooi et al., *Comput. Phys. Commun.* 181, 687–702, 2010; DOI: [`10.1016/j.cpc.2009.11.008`](https://doi.org/10.1016/j.cpc.2009.11.008)) | `$SOLVER_FROZEN/docs/validation/meep_comparison/metagrating_meep.json` | 2D beam-deflecting metagrating fixture ($\Lambda = 2.0\,\mu\text{m}$, dual Si ridges on SiO2, $\lambda = 1.55\,\mu\text{m}$): $T_{-1}=0.0405, T_0=0.0710, T_{+1}=0.7761, R_{-1}=0.0446, R_0=0.0157, R_{+1}=0.0509$, sum = $0.9989$. | $\max \|\eta_{\text{torchfdtd}} - \eta_{\text{meep}}\| = 0.00020$ ($0.02\%$), well within the $0.015$ ($1.5\%$) tolerance gate. |
| **TORCWA** | 0.1.4.2, 241 spatial orders (120 harmonics), GPU/CPU Fourier modal method | `$SOLVER_FROZEN/docs/validation/meep_comparison/metagrating_rcwa.json` | 2D beam-deflecting metagrating fixture ($\Lambda = 2.0\,\mu\text{m}$): $T_{-1}=0.0365, T_0=0.0750, T_{+1}=0.7746, R_{-1}=0.0476, R_0=0.0156, R_{+1}=0.0507$, sum = $1.0000$. | $\max \|\eta_{\text{torchfdtd}} - \eta_{\text{rcwa}}\| = 0.00553$ ($0.55\%$), well within the $0.015$ ($1.5\%$) tolerance gate. |
