# A 2D wavelength splitter

One input waveguide connects to a **3 x 3 um square design region**, which
connects to two output waveguides. The design routes **1.31 um to the upper
guide and 1.55 um to the lower guide**. Only the dielectric pattern inside the
square changes. The three guides remain fixed.

The public `DifferentiablePlaneSimulation` API computes the discrete Yee/CPML
adjoint of the output fluxes at both wavelengths in one run. PyTorch carries
that material gradient through a density filter and a tanh projection and takes
an Adam step.

## Run

Install TorchFDTD from PyPI, download `optimize.py` or enter this directory in a
repository checkout, and run:

```sh
pip install torchfdtd matplotlib
python optimize.py --iterations 60 --check-gradient --output results/wavelength-splitter
```

The default uses eight CPU threads, a 200 x 160 grid with 40 nm cells, 1200
timesteps, FP64 fields and four checkpoints. No CUDA device or Node.js is
required. The seed is fixed. The optional gradient check compares one adjoint
derivative with a central finite difference before the optimization starts.

[Conditions and numerical record](../../docs/assets/adjoint-inverse-design.json)
and [`reference.npz`](reference.npz) retain the actual installed-package run
(TorchFDTD 1.1.7 from PyPI, 35 min on eight threads of an i7-12700).

After 60 Adam updates the design is thresholded at 0.5 and simulated again.
The two-level design sends **97.5%** of the 1.31 um light into the upper guide
(1.0% into the lower) and **93.2%** of the 1.55 um light into the lower guide
(0.2% into the upper). Both start near 30%. One initial adjoint derivative agrees
with a central finite difference to 1.1e-9 relative error.

![Measured CPU wavelength splitter design](../../docs/assets/adjoint-inverse-design.png)

## Objective and design

The waveguides have refractive index 2 in an index-1 background and width
0.5 um. The two output centers are y = +0.8 and -0.8 um. A soft electric sheet
excites the input guide with a short Gaussian pulse centered at 1.43 um, which
covers both wavelengths. Two equal x-normal apertures at x = 2.6 um measure the
spectral Poynting flux at 1.31 and 1.55 um, one around each output guide.

Each transmission T is the aperture flux divided by the flux of a straight
0.5 um guide at the same position and wavelength, computed once before the
optimization. Scattering and reflection losses therefore lower T. The objective is

`J = log T_up(1.31) + log T_low(1.55) + log(T_up(1.31) / T(1.31)) + log(T_low(1.55) / T(1.55))`,

where T(wavelength) is the sum over both apertures. The first two terms reward
transmission into the right guide; the last two penalize light in the wrong one.

There are 5625 density logits (75 x 75, one per cell). A sigmoid, a 60 nm
Gaussian filter and a tanh projection give the relative permittivity between
1 and 4. The projection strength beta stays at 1 for 20 updates and then doubles
every 8 updates up to 32, which drives the design toward two levels.

This is a 2D TM (Ez) tutorial. The transmissions are aperture fluxes normalized
by a straight guide, not modal S-parameters, and the coarse tutorial does not
establish mesh convergence or a minimum feature size.

Outputs:

- `summary.png`: initial and thresholded design, |Ez|^2 at both wavelengths, transmission and beta history.
- `design.npz`: density/permittivity arrays, complex Ez maps of the thresholded design and the transmission history.
- `result.json`: versions, driver hash, geometry, transmissions, objective and gradient check.
- `project.json`: fixed guide/source/monitor setup with the initial uniform design placeholder.
- `optimized_project.json`: the thresholded design as 280 merged rectangles plus the three guides,
  ready to open in the browser workbench with `torchfdtd serve`.

The [recorded splitter project](splitter.json) can be opened without rerunning
the optimization. The numerical design remains in `reference.npz`.

The plots can be regenerated from saved numerical results without repeating
the optimization:

```sh
python optimize.py --render-only --output results/wavelength-splitter
```

For the full-aperture nine-wavelength lens, color hologram and polarization-switched
hologram, see [the paper examples](../full-aperture-tiled-adjoint/).
