# Full-aperture tiled-adjoint design of three metasurfaces

Code, settings, final layouts and source data of the three freeform designs E1-E3 of the accompanying preprint
(H. Park and Y. Park, 2026). Each design is a density-based topology optimization over the whole aperture: every step solves
all tiles of the aperture forward and adjoint with TorchFDTD (3D FDTD, 20 nm Yee lattice), assembles the exit-plane Jones maps,
propagates them with a vector angular spectrum to the objective, and updates a filtered, tanh-projected density with Adam.

| | E1 nine-wavelength lens | E2 color hologram | E3 polarization-switched hologram |
|---|---|---|---|
| aperture | 200 um disk, D4 symmetric | 150 x 150 um square | 100 x 100 um square |
| wavelengths, input polarization | 420-670 nm (9), x and y | 450, 540, 635 nm, x | 540 nm, x and y |
| target | common focus at f = 333.3 um | color image at z = 300 um | x: "CNU", y: "PHY" at z = 200 um |
| driver | `src/zonec_multi.py` (polar octant) | `src/ns_run.py` (10 nm Cartesian) | `src/ns_run.py` (10 nm Cartesian) |
| tile core, solves per step | 19 um, 16 octant tiles x 2 pol | 19 um, 64 tiles | 25 um, 16 tiles x 2 pol |
| hardware of the reported run | 8 x H200 | 6 x H200 (steps 0-9), 8 x H100 80GB | 6 x H200 |
| steps, median time per step | 250, 76 s | 159, 218 s | 190, 225 s |
| binary objective in the run (margin 0.9 um) | 0.1205 (step 248) | 0.6761 (step 158) | 0.8479 (step 189) |
| score with 3.6 um margins (`evaluate.sh`) | 0.1196 | 0.6730 | 0.8439 |
| fill fraction of the final layout | 0.442 | 0.398 | 0.400 |

Common physics: SiN pillars 700 nm tall (n = 2.0) on SiO2 (n = 1.444), 2.5 um tall tile domain with CPML along z and periodic
lateral faces, 1200 time steps, float32 fields, complex128 propagation. Objectives are defined in `src/fa_objectives.py`.

## Files

| path | content |
|---|---|
| `designs/e{1,2,3}_freeform_step*.npz` | final binary layouts of E1-E3 on the 10 nm grid (`np.packbits`, x-major; E1 is the D4 quadrant x, y >= 0) |
| `designs/e{1,2,3}_metaatom.npz` | meta-atom baseline layouts, same encoding |
| `libraries/` | meta-atom libraries of the baselines (TORCWA, double precision): square posts with 13 and 11 Fourier orders (`library_o13.npz`, `library_o11.npz`), posts, holes and crosses (`library_disp_o13.npz`, `library_disp_o11.npz`) and the rectangular posts of E3 at 540 nm (`library_rect540_o13.npz`) |
| `reference/e{1,2,3}_history.csv` | optimization history of each run: beta, learning rate, gray and binary objective, gray fraction, fill, time per step |
| `reference/provenance.json` | TorchFDTD commit, software versions, source hashes and final scores |
| `src/` | drivers, tile solver set-up, objectives, filter/projection/Adam loop, solver speed patches |
| `configs/`, `schedules/`, `targets/` | settings of each case, beta of every step of the reported runs, design targets with SHA-256 |
| `run.sh`, `evaluate.sh`, `smoke.sh` | optimization, re-scoring with 3.6 um margins, small wiring check |
| `tools/` | E1 polar lookup builder, layout conversion, history comparison, CPU objective check, Windows launcher for `smoke.sh` |
| `figures/source_data/` | source data of every figure and table of the preprint (see below) |

Load a layout:

```python
import numpy as np
z = np.load("designs/e3_freeform_step0189.npz")
nx, ny = z["d10_shape"]
layout = np.unpackbits(z["d10_bits"])[:nx * ny].reshape(nx, ny)   # 1 = SiN, 10 nm pixels
```

The key names differ per file (`d10_quad`, `d10_bits` or `d10_full`, with the matching `*_shape`); the `note` field of each
file says which.

## Requirements

- Linux, NVIDIA GPUs (tested on H100 80GB and H200). One E1/E2 tile solve peaks at 37-40 GiB; an E3 tile (25 um core) needs
  about 64 GiB and therefore an H200-class GPU at these settings.
- Host RAM: with `FA_KEEP_TRACE=1` (the setting of the runs) each rank keeps forward boundary traces in pinned host memory, capped
  by `FF_HOST_GIB`. On smaller nodes set `FA_KEEP_TRACE=0` (same gradient to float32 round-off, about 28 % slower).
- Python 3.12, PyTorch 2.8.0 (CUDA 12.8), NumPy 2.1, SciPy, CuPy 13 (`cupy-cuda12x`), and TorchFDTD at commit `9d97df2`, the
  version of the reported runs:

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install "torchfdtd[cuda-kernels] @ git+https://github.com/hyoseokp/TorchFDTD@9d97df22384d2b788ee8ae9338f0d2c8456246ad"
```

## Evaluating the published layouts

```bash
bash evaluate.sh e1 designs/e1_freeform_step0248.npz     # or e2/e3 and any file in designs/
```

One forward solve per tile on all GPUs of the node (tile cores 15 um, margins 3.6 um), the homogeneous-substrate reference
solved first; the objective is printed and written to `runs/eval_<layout>/eval_*/metrics.json`. `EVAL_OVER` and `EVAL_TILE`
change the margin and the core. On 8 x H100 80GB the three freeform scores took 353 s (E1), 132 s (E2) and 113 s (E3).

## Running the optimizations

```bash
bash run.sh e1                         # or e2, e3; output in runs/<case>/, all GPUs of the node (NGPU to change)
bash evaluate.sh e1 runs/e1            # re-score the binary layout of the reference step with 3.6 um margins
python tools/compare_history.py e1 runs/e1
```

`run.sh` sources `configs/common.env` and `configs/<case>.env` and applies the beta of every step from `schedules/<case>.json`;
the learning rate follows from beta as lr(k) = min(0.05, 0.4 / beta(k)) * min(1, (k + 1) / 3). The gray start is 0.5 plus
uniform noise of amplitude 0.15 (seed 20260926). Running the same command again resumes after the last completed step, and
`touch runs/<case>/STOP` stops the run before its next step. E1 first builds `cache/d200_lookup.npz` (polar lookup of the octant,
about 250 MB and 30 s) and checks it against the SHA-256 of the lookup used in the reported run. Any `FF_*`, `EZ_*`, `EZC_*` or
`FA_*` variable set in the shell overrides the env files.

Repeated runs are not bitwise identical (float32 atomics and the NCCL reduction order differ at the 1e-7 relative level per
step), and E2's reported run changed from 6 to 8 GPUs at step 10, so a new run follows `reference/<case>_history.csv` closely
rather than to the last digit.

Checks without the full problem:

```bash
python tools/check_objectives.py   # CPU: the three objectives, target hashes, value and gradient on random pupil maps
bash smoke.sh e2                   # one GPU: the case's code path on a 6 um test lens, 2 steps
```

## Source data of the figures and tables

`figures/source_data/` holds, for each item of the preprint, the arrays it plots (`<item>.npz`, named `<record>__<key>`), the
JSON and CSV records it reads (`<item>.json`), and for each table every value behind it as (source, key, value) rows. The keys
name the record files of the original runs; those files themselves are not part of this folder.

| item | source data |
|---|---|
| Figs. 1-6 | `fig1.npz/.json` ... `fig6.json` (Fig. 6 includes the per-step histories of the three runs) |
| Extended Data Table 1 | `extended_data_table1.csv` |
| Supplementary Figs. 1-9 | `supp_fig1.json` ... `supp_fig9.json`, with `supp_fig2.npz`, `supp_fig6.npz`, `supp_fig7.npz`, `supp_fig8.npz` |
| Supplementary Figs. 10-13 | the layouts in `designs/` (`supp_fig10_13.json`) |
| Supplementary Tables 1-13 | `supp_table<k>_*.csv` |

Supplementary Fig. 8 and Supplementary Table 12 (layer thickness and edge bias) give, for the three freeform layouts, the
objective and its components at layer heights of 620, 640 and 700 nm and at edge biases of -20, -10, 0, +10 and +20 nm
(`supp_fig8.json`, `supp_fig8.npz`, `supp_table12_robustness.csv`).

Full three-dimensional fields are not stored. The transmitted exit-plane fields (`pupil_jones.npz`) and all scores can be
regenerated from the layouts above with `evaluate.sh`.
