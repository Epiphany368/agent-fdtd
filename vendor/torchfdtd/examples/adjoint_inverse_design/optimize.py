"""CPU 2D wavelength splitter: input waveguide, square design region, two output waveguides.

1.31 um is routed to the upper output and 1.55 um to the lower output. Transmissions are the
spectral Poynting flux in each output aperture divided by the flux of a straight reference guide.
The density is filtered and projected toward a binary design during the run.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchfdtd import (AdjointOptions, DifferentiablePlaneSimulation, FieldMonitor,
                       Material, Project, Region, Source, SpectrumSettings, Structure)

WAVELENGTHS_UM = (1.31, 1.55)       # routed to the upper and the lower output
FREQUENCIES_HZ = [299792458.0 / (w * 1e-6) for w in WAVELENGTHS_UM]
MESH_UM = .04
DESIGN_SIDE_UM = 3.0
GUIDE_WIDTH_UM = .5
OUTPUT_Y_UM = .8
APERTURE_X_UM = 2.6
EPS_CORE = 4.0                      # refractive index 2 in an index-1 background
FILTER_SIGMA_CELLS = 1.5            # 60 nm Gaussian density filter


def beta_at(update):
    """Projection strength: grey for 20 updates, then doubled every 8 updates up to 32."""
    return 1.0 if update < 20 else float(min(32, 2 ** (1 + (update - 20) // 8)))


def make_project(with_outputs=True, reference=False):
    guides = [Structure(id="input-guide", name="Input waveguide", center=(-2.75, 0, 0), size=(2.5, GUIDE_WIDTH_UM, .04), material="guide")]
    if reference:
        guides = [Structure(id="reference-guide", name="Straight reference guide", center=(0, 0, 0), size=(8, GUIDE_WIDTH_UM, .04), material="guide")]
    elif with_outputs:
        guides += [Structure(id="upper-guide", name="Upper output (1.31 um)", center=(2.75, OUTPUT_Y_UM, 0), size=(2.5, GUIDE_WIDTH_UM, .04), material="guide"),
                   Structure(id="lower-guide", name="Lower output (1.55 um)", center=(2.75, -OUTPUT_Y_UM, 0), size=(2.5, GUIDE_WIDTH_UM, .04), material="guide"),
                   Structure(id="design-box", name="Square design region", center=(0, 0, 0), size=(DESIGN_SIDE_UM, DESIGN_SIDE_UM, .04), material="initial design")]
    apertures = [("reference", 0.0)] if reference else [("upper", OUTPUT_Y_UM), ("lower", -OUTPUT_Y_UM)]
    return Project(name="Adjoint 2D wavelength splitter" + (" reference" if reference else ""),
        region=Region(dimension="2d", size=(8, 6.4, .04), mesh=MESH_UM, pml_cells=12,
                      steps=1200, precision="float64", backend="cpu", material_sampling="yee"),
        materials=[Material(name="guide", index=2.0, color="#2c8e9c"),
                   Material(name="initial design", index=math.sqrt(2.5), color="#70b8bd")],
        structures=guides,
        sources=[Source(name="Input excitation", kind="plane", normal="x", center=(-3.0, 0, 0),
                        size=(0, GUIDE_WIDTH_UM, .04), component="Ez", wavelength=1.43, pulse="gaussian",
                        time_definition="standard", pulse_length=6e-15, pulse_offset=15e-15)],
        monitors=[FieldMonitor(id=name, name=name.title() + " output flux", normal="x",
                               center=(APERTURE_X_UM, y, 0), size=(0, 1.2, .04), inherit_apodization=False,
                               spectrum=SpectrumSettings(sampling="frequency", wavelength_start=1.31,
                                   wavelength_stop=1.55, frequency_points=2, apodization="none"))
                  for name, y in apertures])


def node_grid(project):
    x = torch.as_tensor(np.array(project.region.mesh_nodes[0][:project.region.shape[0]]), dtype=torch.float64)
    y = torch.as_tensor(np.array(project.region.mesh_nodes[1][:project.region.shape[1]]), dtype=torch.float64)
    return torch.meshgrid(x, y, indexing="ij")


def design_slices(project):
    x, y = node_grid(project)
    half = DESIGN_SIDE_UM / 2 - 1e-9
    xs = torch.nonzero(x[:, 0].abs() <= half).flatten()
    ys = torch.nonzero(y[0].abs() <= half).flatten()
    return slice(int(xs[0]), int(xs[-1]) + 1), slice(int(ys[0]), int(ys[-1]) + 1)


def fixed_guides(project, reference=False):
    """Relative permittivity of the fixed guides on the Ez nodes (outside the design region)."""
    x, y = node_grid(project)
    w = GUIDE_WIDTH_UM / 2 + 1e-9
    if reference:
        core = y.abs() <= w
    else:
        core = ((x < 0) & (y.abs() <= w)) | ((x > 0) & ((y - OUTPUT_Y_UM).abs() <= w)) | ((x > 0) & ((y + OUTPUT_Y_UM).abs() <= w))
    return torch.where(core, torch.tensor(EPS_CORE, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64))[..., None]


_t = torch.arange(-4, 5, dtype=torch.float64)
KERNEL = torch.exp(-_t ** 2 / (2 * FILTER_SIGMA_CELLS ** 2))
KERNEL = (KERNEL[:, None] * KERNEL[None, :] / KERNEL.sum() ** 2)[None, None]


def project_density(rho, beta, eta=.5):
    return (math.tanh(beta * eta) + torch.tanh(beta * (rho - eta))) / (math.tanh(beta * eta) + math.tanh(beta * (1 - eta)))


def material(theta, project, beta, background):
    rho = F.conv2d(F.pad(torch.sigmoid(theta)[None, None], (4, 4, 4, 4), mode="replicate"), KERNEL)[0, 0]
    density = project_density(rho, beta)
    epsilon = background.clone()
    xs, ys = design_slices(project)
    epsilon[xs, ys, 0] = 1 + (EPS_CORE - 1) * density
    return epsilon, density


def transmissions(model, epsilon, project, incident):
    planes = model(epsilon, FREQUENCIES_HZ)
    # powers[port, wavelength] with port 0 upper and 1 lower, divided by the straight-guide flux
    powers = torch.stack([planes[name].flux()[:2] for name in ("upper", "lower")]) / incident
    if not bool(torch.isfinite(powers).all()) or bool((powers <= 0).any()):
        raise RuntimeError("Both apertures must carry finite positive forward flux.")
    return powers


def objective(t):
    """log T_up(1.31) + log T_low(1.55) + log of each wavelength's share in its own guide."""
    routed = torch.stack([t[0, 0], t[1, 1]])
    return torch.log(routed).sum() + torch.log(routed / t.sum(0)).sum()


def reference_flux():
    project = make_project(reference=True)
    model = DifferentiablePlaneSimulation(project, AdjointOptions(checkpoints=4))
    with torch.no_grad():
        planes = model(fixed_guides(project, reference=True), FREQUENCIES_HZ)
    return planes["reference"].flux()[:2].detach()


def export_project(project, density, path):
    """Export the thresholded design as merged row rectangles for the browser workbench."""
    binary = density > .5
    xs, ys = design_slices(project)
    nodes_x, nodes_y = project.region.mesh_nodes[0], project.region.mesh_nodes[1]
    structures = [s for s in project.structures if s.id != "design-box"]
    count = 0
    for i in range(binary.shape[0]):
        j = 0
        while j < binary.shape[1]:
            if not binary[i, j]:
                j += 1
                continue
            k = j
            while k < binary.shape[1] and binary[i, k]:
                k += 1
            y0, y1 = float(nodes_y[ys.start + j]) - MESH_UM / 2, float(nodes_y[ys.start + k - 1]) + MESH_UM / 2
            structures.append(Structure(id=f"design-{i}-{j}", name=f"design-{i}-{j}", material="guide",
                center=(float(nodes_x[xs.start + i]), (y0 + y1) / 2, 0), size=(MESH_UM, y1 - y0, .04)))
            count += 1
            j = k
    exported = Project.model_validate(project.model_copy(update={"materials": [project.materials[0]], "structures": structures}).model_dump())
    exported.save(path)
    return {"binary_threshold": .5, "design_rectangles": count, "structures": len(structures)}


def field_image(project, epsilon):
    view = project.model_copy(deep=True)
    # The 2D plane API uses x/y-normal observation lines. Collect Ez on every x node line
    # inside the absorber to assemble a full-resolution 2D image.
    xs = [float(x) for x in project.region.mesh_nodes[0][14:-14:1]]
    view.monitors = [FieldMonitor(id=f"view-{i}", normal="x", center=(x, 0, 0), size=(0, 5.2, .04), inherit_apodization=False)
                     for i, x in enumerate(xs)]
    view = Project.model_validate(view.model_dump())
    model = DifferentiablePlaneSimulation(view, AdjointOptions(checkpoints=4))
    with torch.no_grad():
        planes = model(epsilon, FREQUENCIES_HZ)
    first = planes[view.monitors[0].id]
    return {"field_ez": np.stack([np.stack([planes[m.id].fields[k, :, 2].numpy() for m in view.monitors]) for k in range(2)]),
            "field_x_um": np.array(xs), "field_y_um": first.points_um[:, 1].numpy()}


def render(output, arrays, history, record):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from matplotlib.patches import Rectangle
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    half = DESIGN_SIDE_UM / 2
    extent = (-4 - MESH_UM / 2, 4 - MESH_UM / 2, -3.2 - MESH_UM / 2, 3.2 - MESH_UM / 2)
    for axis, key, title in zip(axes[0, :2], ("initial_epsilon", "binary_epsilon"), ("Initial design", "Optimized design (thresholded)")):
        image = axis.imshow(arrays[key][..., 0].T, origin="lower", extent=extent, cmap="viridis", vmin=1, vmax=4, interpolation="nearest")
        axis.add_patch(Rectangle((-half, -half), 2 * half, 2 * half, fill=False, edgecolor="white", linestyle="--"))
        axis.plot([-3.0, -3.0], [-.25, .25], color="#ffad32", linewidth=3)
        axis.plot([APERTURE_X_UM] * 2, [OUTPUT_Y_UM - .6, OUTPUT_Y_UM + .6], color="#5aa9ff", linewidth=2)
        axis.plot([APERTURE_X_UM] * 2, [-OUTPUT_Y_UM - .6, -OUTPUT_Y_UM + .6], color="#ee6666", linewidth=2)
        axis.set(title=title, xlabel="x (um)", ylabel="y (um)", xlim=(-3.5, 3.5), ylim=(-2.6, 2.6))
    axes[0, 0].text(-3.4, .4, "input", color="white")
    axes[0, 0].text(1.6, 1.7, "upper: 1.31 um", color="white")
    axes[0, 0].text(1.6, -2.0, "lower: 1.55 um", color="white")
    figure.colorbar(image, ax=axes[0, :2].tolist(), label="Relative permittivity", shrink=.85)
    binary = record["binary"]["transmission"]
    for k, axis in zip(range(2), (axes[1, 0], axes[1, 1])):
        intensity = np.abs(arrays["field_ez"][k]) ** 2
        intensity /= intensity.max()
        field = axis.pcolormesh(arrays["field_x_um"], arrays["field_y_um"], intensity.T,
                                shading="nearest", cmap="inferno", norm=PowerNorm(.5, vmin=0, vmax=1), rasterized=True)
        axis.add_patch(Rectangle((-half, -half), 2 * half, 2 * half, fill=False, edgecolor="white", linestyle="--", linewidth=.8))
        port = "upper" if k == 0 else "lower"
        axis.set(title=f"|Ez|^2 at {WAVELENGTHS_UM[k]} um: T_{port} = {100 * binary[k][k]:.0f}%", xlabel="x (um)", ylabel="y (um)",
                 aspect="equal", xlim=(-3.5, 3.5), ylim=(-2.6, 2.6))
    figure.colorbar(field, ax=axes[1, :2].tolist(), label="Normalized |Ez| squared", shrink=.85)
    t = np.asarray([entry["transmission"] for entry in history])
    for (port, k), style, label in (((0, 0), "o-", "upper, 1.31 um"), ((1, 0), "o:", "lower, 1.31 um"),
                                     ((1, 1), "^-", "lower, 1.55 um"), ((0, 1), "^:", "upper, 1.55 um")):
        axes[0, 2].plot(100 * t[:, port, k], style, markersize=3, label=label, color="#3a7fd0" if k == 0 else "#d0503a")
    axes[0, 2].set(title="Transmission during optimization", xlabel="Adam update", ylabel="% of straight-guide flux", ylim=(0, 100))
    axes[0, 2].legend(fontsize=8, loc="center right")
    axes[1, 2].semilogy([entry["beta"] for entry in history], color="#2c3440")
    axes[1, 2].set(title="Projection strength beta", xlabel="Adam update", ylabel="beta")
    axes[1, 2].text(.04, .9, f"Thresholded design:\n1.31 um upper {100 * binary[0][0]:.1f}%, lower {100 * binary[1][0]:.1f}%\n"
                             f"1.55 um lower {100 * binary[1][1]:.1f}%, upper {100 * binary[0][1]:.1f}%",
                    transform=axes[1, 2].transAxes, va="top", fontsize=9, bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
    figure.savefig(output / "summary.png", dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("results/wavelength-splitter"))
    parser.add_argument("--check-gradient", action="store_true")
    parser.add_argument("--render-only", action="store_true", help="Regenerate plots from saved numerical results.")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    torch.set_num_threads(args.threads)
    if args.render_only:
        record = json.loads((args.output / "result.json").read_text())
        with np.load(args.output / "design.npz") as saved:
            arrays = {key: saved[key] for key in saved.files}
        render(args.output, arrays, record["history"], record)
        return
    started = time.perf_counter()
    incident = reference_flux()
    project = make_project()
    model = DifferentiablePlaneSimulation(project, AdjointOptions(checkpoints=4))
    background = fixed_guides(project)
    xs, ys = design_slices(project)
    n = (xs.stop - xs.start, ys.stop - ys.start)
    theta = torch.nn.Parameter(.02 * torch.randn(n, generator=torch.Generator().manual_seed(args.seed), dtype=torch.float64))
    optimizer = torch.optim.Adam([theta], lr=.1)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    history, arrays, derivative = [], {}, None
    for update in range(args.iterations + 1):
        beta = beta_at(update)
        optimizer.zero_grad(set_to_none=True)
        epsilon, density = material(theta, project, beta, background)
        t = transmissions(model, epsilon, project, incident)
        score = objective(t)
        row = {"update": update, "beta": beta, "score": float(score.detach()), "transmission": t.detach().tolist(),
               "grey_fraction": float(((density > .05) & (density < .95)).double().mean())}
        history.append(row)
        if update == 0:
            arrays["initial_epsilon"] = epsilon.detach().numpy().copy()
        print(f"update {update:02d} beta {beta:4.0f}: 1.31 um upper {100 * t[0, 0]:.1f}% lower {100 * t[1, 0]:.1f}% | "
              f"1.55 um lower {100 * t[1, 1]:.1f}% upper {100 * t[0, 1]:.1f}% | grey {100 * row['grey_fraction']:.0f}%", flush=True)
        if update == args.iterations:
            break
        (-score).backward()
        if theta.grad is None or not bool(torch.isfinite(theta.grad).all()):
            raise RuntimeError("Nonfinite or missing adjoint gradient.")
        if update == 0 and args.check_gradient:
            index = int(theta.grad.abs().argmax())
            adjoint = -float(theta.grad.flatten()[index])
            values, step = [], 1e-4
            for sign in (-1, 1):
                trial = theta.detach().clone()
                trial.flatten()[index] += sign * step
                values.append(float(objective(transmissions(model, material(trial, project, beta, background)[0], project, incident))))
            finite_difference = (values[1] - values[0]) / (2 * step)
            error = abs(adjoint - finite_difference) / max(abs(adjoint), abs(finite_difference), 1e-12)
            derivative = {"parameter_flat_index": index, "step": step, "adjoint": adjoint, "finite_difference": finite_difference, "relative_error": error}
            print(f"gradient check: relative error {error:.2e}", flush=True)
            if error > 1e-5:
                raise RuntimeError(f"Gradient check failed: {derivative}")
        optimizer.step()
    # Score the thresholded design: what a fabricated two-level device would do.
    with torch.no_grad():
        final_density = density.detach()
        binary_epsilon = background.clone()
        binary_epsilon[xs, ys, 0] = torch.where(final_density > .5, torch.tensor(EPS_CORE, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64))
        binary_t = transmissions(model, binary_epsilon, project, incident)
    arrays.update(final_epsilon=epsilon.detach().numpy(), final_density=final_density.numpy(), binary_epsilon=binary_epsilon.numpy(),
                  logits=theta.detach().numpy(), transmission_history=np.asarray([r["transmission"] for r in history]),
                  incident_flux=incident.numpy())
    arrays.update(field_image(project, binary_epsilon))
    np.savez_compressed(output / "design.npz", **arrays)
    project.save(output / "project.json")
    record = {"torchfdtd_version": importlib.metadata.version("torchfdtd"), "torch_version": torch.__version__, "numpy_version": np.__version__,
              "backend": "cpu", "threads": args.threads, "precision": "float64", "shape": project.region.shape, "mesh_um": MESH_UM,
              "steps": project.region.steps, "checkpoints": 4, "design_side_um": DESIGN_SIDE_UM, "density_parameters": theta.numel(),
              "filter_sigma_nm": FILTER_SIGMA_CELLS * MESH_UM * 1000, "projection": "tanh, eta 0.5, beta 1 for 20 updates then x2 every 8 up to 32",
              "guide_width_um": GUIDE_WIDTH_UM, "output_centers_y_um": [OUTPUT_Y_UM, -OUTPUT_Y_UM], "core_index": 2.0,
              "wavelengths_um": list(WAVELENGTHS_UM), "routing": {"1.31": "upper", "1.55": "lower"}, "seed": args.seed,
              "iterations": args.iterations, "objective": "log T_up(1.31) + log T_low(1.55) + log(T_up(1.31)/T(1.31)) + log(T_low(1.55)/T(1.55))",
              "transmission_definition": "aperture flux / flux of a straight 0.5 um guide at the same aperture position, per wavelength; transmission[port][wavelength], port 0 upper",
              "history": history, "final": {"transmission": history[-1]["transmission"], "grey_fraction": history[-1]["grey_fraction"]},
              "binary": {"threshold": .5, "transmission": binary_t.tolist()}, "gradient_check": derivative,
              "seconds": time.perf_counter() - started,
              "scope": "2D TM (Ez) tutorial. Aperture flux normalized by a straight guide; not modal S-parameters.",
              "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    record["cad_export"] = export_project(project, final_density.numpy(), output / "optimized_project.json")
    (output / "result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    render(output, arrays, history, record)
    print(json.dumps({k: record[k] for k in ("binary", "final", "gradient_check", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
