# Metalens: TorchFDTD against Meep

Two focusing devices, each set up once in `geometry.json` / `geometry_3d.json` and run by both solvers on the
same grid, time step, step count, source, monitors and material sampling. The records live in
`docs/validation/meep_comparison/`, the figure in `docs/figures/meep_comparison/metalens.png`, and
`compare.py` produces the tables below from the records alone.

## Part A: 2D cylindrical lens of silicon ridges

- Silicon ridges (n = 3.48) in air on a 0.6 um pitch, Ez (out-of-plane) polarisation, plane-wave pulse from below (+y), 1.55 um.
- Aperture 20 um (33 ridges at x = -9.6, -9.0, ..., 9.6 um), design focal length 15 um (NA 0.55), ridge height 1.0 um.
- Cell 24 x 21.5 um, mesh 0.025 um: 960 x 860 = 825,600 cells. PML 40 cells (1.0 um) on all four sides.
- Courant number 0.99/sqrt(2) = 0.70004, dt = 5.83767e-17 s, 5200 steps = 303.6 fs. The pulse (3-cycle Gaussian, sigma 15.5 fs,
  envelope centre 62 fs) ends near 155 fs; the 18 um path from an aperture edge to the focus takes 60 fs; the rest is the exit of the
  pulse through the top PML.
- Source: Ez current sheet at y = -9.5 um (0.25 um above the lower PML) over |x| <= 10 um. Meep gives the two end nodes of a volume
  source half weight (linear interpolation of the volume onto the grid, measured on a test cell); TorchFDTD reproduces this with a
  799-node sheet at full amplitude plus two half-amplitude point sources at x = +-10 um.
- Ridges from y = -9.0125 to -8.0125 um: the edges lie on half cells in x and y, so no Ez node sits on a ridge surface and the
  staircase is identical in both solvers (16,360 silicon Ez nodes in both records).
- Monitors, all at DFT frequencies c/1.6, c/1.55 and c/1.5 um over the whole run without a window: the aperture-wide line at
  y = -7.9 um (0.1125 um above the ridge tops; the bare-cell run gives the incident power and intensity there), the focal line
  at y = 5.6 um over the 22 um interior, and the on-axis line x = 0 from y = -7.9 to 9.75 um (the top PML starts at 9.75 um).
- Design step (TorchFDTD only, `design_2d.py`, stored in `design_2d.json`): a periodic unit-cell sweep of the transmission phase versus
  ridge width, widths 0.075 to 0.575 um in 0.05 um steps (every width the 0.025 um staircase can realise; 11 entries). At 1.0 um height
  the unwrapped phase spans 6.526 rad = 1.039 x 2 pi; at 0.9 um it spanned 5.375 rad, which is why the height is 1.0 um. Each ridge takes
  the library width whose phase is closest to the hyperbolic target modulo 2 pi (8 distinct widths are used). The same design run located the
  on-axis intensity peak at y = 5.5999 um, 13.61 um above the ridge tops (the geometric focus is 15 um; a 20 um aperture at f = 15 um has
  Fresnel number 4.3, for which the axial peak lies before the geometric focus), and the focal-plane monitor was declared at the nearest node,
  y = 5.6 um, before either comparison run.

## Part B: 3D lens of 112 silicon pillars

- The pillar list of `benchmarks/angular_spectrum.py` (`extract_3d.py` calls its `metalens()` builder with the pillar library recorded in
  `docs/validation/angular-spectrum-3060.json`): 6 um aperture, hyperbolic phase for f = 6 um, 0.5 um square grid, 0.6 um tall cylinders,
  radii 0.0871 to 0.1900 um, n = 3.48. No Yee sample of any component lies exactly on a pillar surface (the pillar faces sit a quarter
  cell off the z grid), so the staircase is identical in both solvers (35,808 / 35,808 / 36,288 silicon Ex / Ey / Ez samples in both records).
- x-polarised plane-wave pulse from below (+z), the same 3-cycle pulse as Part A.
- Cell 8 x 8 x 6.7 um, mesh 0.05 um: 160 x 160 x 134 = 3,430,400 cells. PML 10 cells (0.5 um) on all six sides; 0.5 um of air between
  the aperture edge and the lateral PML.
- Courant number 0.99/sqrt(3) = 0.57158, dt = 9.53287e-17 s, 2500 steps = 238.3 fs.
- Source: Ex current sheet at z = -2.5 um over |x| <= 2.975 um (Ex samples sit on half cells in x) and |y| <= 3 um (nodes), i.e. a Meep
  volume source of 5.95 x 6 um whose boundaries lie on Ex sample positions, so Meep weights the boundary samples 1/2 and the corners 1/4
  (measured on a test cell) and injects nothing outside. TorchFDTD reproduces this with a 5.85 x 5.9 um sheet at full amplitude, four
  half-amplitude boundary lines and four quarter-amplitude corner points: 120 x 121 samples with the same weights in both records.
  Pillars from z = -2.1125 to -1.5125 um.
- Monitors (same three frequencies): the 6 x 6 um plane at z = -1.4 um (0.1125 um above the pillar tops; incident power and intensity from
  the bare run), the 7 x 7 um focal plane at z = 1.65 um (3.1625 um above the pillar tops; the node nearest the on-axis peak of the
  TorchFDTD run in `extract_3d.py`, declared before the comparison runs), and the xz (y = 0) and yz (x = 0) sections from z = -1.4 to 2.85 um.
- The peak lies 3.12 um above the pillar tops (both solvers, 1.55 um), not at the design focal length of 6 um: the lens is 12 pillars wide with an 11-entry
  library, and the record of `benchmarks/angular_spectrum.py` places the same focus at 3.275 um in a cell with a 1 um lateral margin.
  This example compares the two solvers on that device as it is.

## Fairness

- One geometry file per part, loaded by both scripts; the records carry its SHA-256 and `compare.py` checks that it is the same.
- Same cell counts, mesh, Courant number (`dt` equal to the last bit), step count and run time; the Meep script steps the fields
  itself so that the step count is exact.
- Same PML thickness in cells. The formulations differ: TorchFDTD uses a stretched-coordinate CPML with a cubic sigma profile
  (kappa = 1, alpha = 1e-8); Meep uses its own stretched-coordinate PML with the default quadratic profile.
- Same source: one Gaussian pulse function of time (`metalens_common.waveform`, asserted equal to TorchFDTD's native waveform to 1e-15)
  injected as an additive current over the same nodes. Amplitude conventions differ, so every reported quantity is a ratio to the
  bare-cell run of the same solver at the same frequency. The boundary-sample weights are matched in both parts (2D: 799 nodes at 1
  and two end nodes at 1/2; 3D: 118 x 119 samples at 1, the boundary lines at 1/2, the corners at 1/4); both records list the source
  cell slices with their weights and `compare.py` checks that they are equal.
- Same monitor points: both solvers report the fields at the cell-centred points of each monitor plane. TorchFDTD interpolates its
  Yee samples there (trilinear); the Meep script reads the DFT fields on the Yee grid (`yee_grid=True`) and applies the same neighbour
  averaging. The reductions (peak, FWHM, windowed Poynting power) are one shared function set in `metalens_common.py`.
- Same material sampling: Meep `eps_averaging=False`, TorchFDTD `material_sampling="yee"` with staircase interfaces, both sampling the
  geometry at each Yee component position. The silicon sample counts per component are in both records and are compared.
- No symmetry planes, no subpixel smoothing, no coarse-then-fine. TorchFDTD runs float32 on the GPU with the fused CUDA kernels;
  Meep runs float64 on 4 MPI ranks with `OMP_NUM_THREADS=1`.

## Running

From the worktree root inside the WSL2 distribution `torchfdtd-bench` (`wsl.exe -d torchfdtd-bench`):

```
PYTHONPATH=$PWD python examples/meep_comparison/metalens/torchfdtd_metalens.py --part 2d
OMP_NUM_THREADS=1 mpirun -np 4 python examples/meep_comparison/metalens/meep_metalens.py --part 2d --ranks 4
python examples/meep_comparison/metalens/compare.py
```

`--part 3d` runs Part B with the same two solver commands (4 to 6 s per solve on the shared RTX 3060; 129 to 167 s per solve for Meep with 4 ranks on the shared CPU, two solves per script).
Both solver scripts take `--out <path>` and `--timing` (one warm-up, then three timed lens solves; the medians go into the record).
`compare.py` reads the four records and `criteria.json`, writes `docs/validation/meep_comparison/metalens_comparison.json`, renders
`docs/figures/meep_comparison/metalens.png` from the records, and prints the tables below. `design_2d.py` and `extract_3d.py` regenerate
the geometry files (TorchFDTD only) and are not part of the comparison.

## Criteria

Declared in `criteria.json` before the first comparison run, evaluated at 1.55 um: focal position within 0.1 um; FWHM within 3 percent
(3D: along x and along y); focal-plane profile RMS difference at most 3 percent of the peak; focusing efficiency (power within 1.5 FWHM
of the focal-plane peak, over the bare-cell power through the aperture plane) within 1 percentage point. The 1.6 and 1.5 um samples
are reported for information.

## Results

The tables are the output of `compare.py` (a test regenerates them from the records and compares).

### 2D ridge lens (Part A): agreement at 1.55 um

| Metric | TorchFDTD | Meep | Difference | Criterion | Pass |
|---|---|---|---|---|---|
| Focal position (um) | 5.6000 | 5.5992 | 7.87e-04 um | <= 0.1 um | yes |
| Focal-plane FWHM (um) | 1.1629 | 1.1629 | 1.38e-05 (relative) | <= 0.03 (relative) | yes |
| Focal-plane profile RMS difference / peak | 9.5296 | 9.5303 | 1.67e-05 of the peak | <= 0.03 of the peak | yes |
| Focusing efficiency | 0.5571 | 0.5571 | 3.96e-06 | <= 0.01 | yes |
| Transmission through the aperture plane | 0.6740 | 0.6741 | 1.18e-04 | (information) | - |
| On-axis peak intensity / incident | 9.5137 | 9.5143 | 6.67e-04 | (information) | - |

2D ridge lens (Part A): extra wavelengths (information, from the records)

| Wavelength (um) | Focal position TorchFDTD / Meep (um) | FWHM (um) TorchFDTD / Meep | Efficiency TorchFDTD / Meep | Transmission TorchFDTD / Meep |
|---|---|---|---|---|
| 1.60 | 5.5949 / 5.5967 | 1.2159 / 1.2159 | 0.5111 / 0.5112 | 0.6599 / 0.6600 |
| 1.50 | 5.7176 / 5.7178 | 1.0671 / 1.0671 | 0.4850 / 0.4850 | 0.6796 / 0.6796 |

2D ridge lens (Part A): wall time of the lens run

| Solver | Hardware | Precision | Cells | Steps | Setup (s) | Stepping (s) | Full (s) | Timing mode |
|---|---|---|---|---|---|---|---|---|
| TorchFDTD | NVIDIA GeForce RTX 3060 | float32 | 825600 | 5200 | 0.07 | 1.57 | 1.66 | development |
| Meep | 4 MPI ranks, 12th Gen Intel(R) Core(TM) i7-12700 | float64 | 825600 | 5200 | 0.10 | 13.29 | 13.39 | development |

Stepping ratio Meep / TorchFDTD: 8.5; full-run ratio: 8.0. Load condition: development run on a shared host: other agents used the CPU and the GPU at the same time; the numbers bound the solver time from above

### 3D pillar lens (Part B): agreement at 1.55 um

| Metric | TorchFDTD | Meep | Difference | Criterion | Pass |
|---|---|---|---|---|---|
| Focal position (um) | 1.6061 | 1.6088 | 0.0027 um | <= 0.1 um | yes |
| Focal-plane FWHM along x (um) | 1.3229 | 1.3228 | 1.65e-05 (relative) | <= 0.03 (relative) | yes |
| Focal-plane FWHM along y (um) | 1.1239 | 1.1239 | 1.12e-05 (relative) | <= 0.03 (relative) | yes |
| Focal-plane profile RMS difference / peak | 12.9387 | 12.9390 | 5.65e-06 of the peak | <= 0.03 of the peak | yes |
| Focusing efficiency | 0.6232 | 0.6232 | 2.13e-06 | <= 0.01 | yes |
| Transmission through the aperture plane | 0.8408 | 0.8407 | 1.72e-05 | (information) | - |
| On-axis peak intensity / incident | 12.8747 | 12.8747 | 7.55e-05 | (information) | - |

3D pillar lens (Part B): extra wavelengths (information, from the records)

| Wavelength (um) | Focal position TorchFDTD / Meep (um) | FWHM x, y (um) TorchFDTD / Meep | Efficiency TorchFDTD / Meep | Transmission TorchFDTD / Meep |
|---|---|---|---|---|
| 1.60 | 1.7174 / 1.7177 | 1.3748, 1.1612 / 1.3747, 1.1612 | 0.6159 / 0.6159 | 0.8391 / 0.8390 |
| 1.50 | 1.3987 / 1.3991 | 1.2548, 1.0960 / 1.2548, 1.0960 | 0.6263 / 0.6263 | 0.8767 / 0.8767 |

3D pillar lens (Part B): wall time of the lens run

| Solver | Hardware | Precision | Cells | Steps | Setup (s) | Stepping (s) | Full (s) | Timing mode |
|---|---|---|---|---|---|---|---|---|
| TorchFDTD | NVIDIA GeForce RTX 3060 | float32 | 3430400 | 2500 | 0.31 | 6.06 | 6.59 | development |
| Meep | 4 MPI ranks, 12th Gen Intel(R) Core(TM) i7-12700 | float64 | 3430400 | 2500 | 0.47 | 128.60 | 129.07 | development |

Stepping ratio Meep / TorchFDTD: 21.2; full-run ratio: 19.6. Load condition: development run on a shared host: other agents used the CPU and the GPU at the same time; the numbers bound the solver time from above

Timing: both records are development runs on the shared workstation (other agents used the CPU and the GPU during the runs), one
sample each; the maintainer's `--timing` runs with 12 Meep ranks on a quiet host replace them. Setup is object construction plus grid
initialisation (Meep: `init_sim` and the DFT chunks), stepping is the time loop, full is both. The bare-cell run is timed separately
in the records.

## Deviations from the brief

- Part B uses 3,430,400 cells instead of about 2 million: the focus of this pillar list lies 3.2 um above the pillars, and the cell
  must hold the source, the pillars, the focal region and 0.5 um PML on each side; the lateral margin was reduced to 0.5 um to keep
  the Meep run at 167 s per solve with 4 ranks.
- The focal-plane monitor of each part is declared at the on-axis peak found by a TorchFDTD run (design step), not at the design focal
  length; the Meep run uses the same declared plane.
- The 3D records store the focal-plane maps and the sections at 1.55 um only (the on-axis rows at all three wavelengths); the extra
  wavelength rows come from the per-solver summaries computed by the scripts with the shared functions.
