"""G3-05r5: the Drude sphere of G3-05 with dispersive subpixel interfaces, on a held-out fixture.

The pre-declared case is docs/validation/cases/G3-05r5_drude_sphere_subpixel.json. It keeps the Drude
model, band, pulse, domain, PML, TFSF box, monitors, duration, mesh sequence,
judged mesh, observables and limits of G3-05_drude_sphere.json and changes the
radii and the sphere centres to ones that no development run of the method used
(the case lists those runs). Every limit asserted here is read from the case
file. The fast suite runs the coarse mesh only; TORCHFDTD_G3_FULL=1 adds the
judged mesh and TORCHFDTD_G3_RECORD=<dir> writes <dir>/G3-05r5.json. The fixture
helpers and the complex-index Mie series are those of tests/test_physics_g3_b.py.
"""
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

import test_physics_g3_b as g3
from torchfdtd import Project, Source, Structure, Material, FieldMonitor
from torchfdtd.materials import permittivity

ROOT = Path(__file__).resolve().parents[1]
CASE = json.loads((ROOT/'docs'/'validation'/'cases'/'G3-05r5_drude_sphere_subpixel.json').read_text(encoding='utf-8'))
NAME = 'G3-05r5'


def record(row_id, payload):
    if not g3.RECORD_DIR:
        return
    target = Path(g3.RECORD_DIR)/f'{NAME}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(target.read_text(encoding='utf-8')) if target.exists() else dict(
        task=CASE['task'], case_id=CASE['case_id'], fixture_path=f'docs/validation/cases/{CASE["case_id"]}.json',
        environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                         os=platform.platform(), gpu=torch.cuda.get_device_name() if g3.CUDA else None), rows={})
    data['generated'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    data['full_mode'] = g3.FULL
    data['rows'][row_id] = payload
    target.write_text(json.dumps(data, indent=1, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def project(mesh, sphere, *, backend='cpu', precision='float64'):
    """The fixture at one mesh; ``sphere`` is one entry of fixture.spheres, or None for the empty reference."""
    f = CASE['fixture']
    r = g3.region('3d', (f['domain_um'],)*3, mesh, f['pml_um'], backend, precision, f['duration_fs'], f['interface_method'])
    r.subpixel_quadrature = f['subpixel_quadrature']
    spectrum = g3.wavelength_spectrum(*f['band'])
    monitors = (g3.box_monitors(f['inner_monitor_half_um'], 'xyz', spectrum, 'in_')
                +g3.box_monitors(f['outer_monitor_half_um'], 'xyz', spectrum, 'out_'))
    monitors.append(FieldMonitor(id='incident', name='incident', center=(0., 0., 0.), size=(0., .06, .06), spectrum=spectrum))
    d = f['drude']
    return Project(name='G3 drude sphere r5', region=r,
                   materials=[Material(name='drude', model='drude', epsilon_inf=d['epsilon_inf'], plasma_rad_s=d['plasma_rad_s'],
                                       collision_rad_s=d['collision_rad_s'])],
                   structures=[Structure(kind='sphere', radius=sphere['radius_um'], center=tuple(sphere['center_um']), material='drude')]
                   if sphere else [],
                   sources=[Source(kind='tfsf', size=(f['tfsf_um'],)*3, component='Ez', normal='x', direction='+',
                                   wavelength=f['pulse_wavelength_um'], time_definition='standard', pulse_length=f['pulse_length_s'],
                                   pulse_offset=f['pulse_offset_s'])],
                   monitors=monitors)


def rows(mesh, backend, precision):
    f = CASE['fixture']
    d = f['drude']
    reference, wall_reference = g3.run(project(mesh, None, backend=backend, precision=precision),
                                       ('drude', NAME, mesh, backend, precision, 'empty'))
    out = {}
    for sphere in f['spheres']:
        radius = sphere['radius_um']
        sample_project = project(mesh, sphere, backend=backend, precision=precision)
        sample, wall = g3.run(sample_project, ('drude', NAME, mesh, backend, precision, radius))
        wavelengths, intensity = g3.incident_intensity(reference)
        inner = g3.scattered_power(sample, reference, 'xyz', 'in_')/intensity*1e12
        outer = g3.scattered_power(sample, reference, 'xyz', 'out_')/intensity*1e12
        absorption = g3.total_inward_power(sample, 'xyz', 'in_')/intensity*1e12
        eps = g3.drude_permittivity(wavelengths, d['epsilon_inf'], d['plasma_rad_s'], d['collision_rad_s'])
        mie = g3.sphere_cross_sections(wavelengths, radius, np.sqrt(eps))
        # The Mie series with the ADE-sampled permittivity at (2/dt) tan(omega dt/2) separates the time discretisation.
        eps_ade = permittivity(sample_project.materials[0], g3.C0/(wavelengths*1e-6), sample_project.region.time_step)
        mie_ade = g3.sphere_cross_sections(wavelengths, radius, np.sqrt(eps_ade))
        sca_error, abs_error = outer/mie['scattering']-1, absorption/mie['absorption']-1
        out[radius] = dict(
            mesh_um=mesh, grid=list(sample_project.region.shape), steps=sample_project.region.steps,
            time_step_s=sample_project.region.time_step, backend=backend, precision=precision, radius_um=radius,
            center_um=list(sphere['center_um']), center_in_steps=[c/mesh for c in sphere['center_um']], cells_per_radius=radius/mesh,
            interface_method='subpixel', subpixel=sample.summary['subpixel']['dispersive'],
            wavelengths_um=g3.listed(wavelengths), scattering_um2=g3.listed(outer), inner_scattering_um2=g3.listed(inner),
            absorption_um2=g3.listed(absorption), mie_scattering_um2=g3.listed(mie['scattering']),
            mie_absorption_um2=g3.listed(mie['absorption']), mie_ade_scattering_um2=g3.listed(mie_ade['scattering']),
            mie_ade_absorption_um2=g3.listed(mie_ade['absorption']), scattering_relative_error=g3.listed(sca_error),
            absorption_relative_error=g3.listed(abs_error), max_scattering_relative_error=float(abs(sca_error).max()),
            max_absorption_relative_error=float(abs(abs_error).max()),
            inner_outer_scattering_max_difference_over_band_maximum=float(abs(inner-outer).max()/abs(outer).max()),
            peak_scattering_um=g3.peak_and_width(wavelengths, outer)[0],
            mie_peak_scattering_um=g3.peak_and_width(wavelengths, mie['scattering'])[0],
            peak_absorption_um=g3.peak_and_width(wavelengths, absorption)[0],
            mie_peak_absorption_um=g3.peak_and_width(wavelengths, mie['absorption'])[0], wall_s=wall, wall_reference_s=wall_reference)
    return out


@pytest.mark.parametrize('radius', [s['radius_um'] for s in CASE['fixture']['spheres']])
def test_g3_05r5_drude_sphere_subpixel(radius):
    f = CASE['fixture']
    meshes = f['mesh_sequence_um'] if g3.FULL else f['mesh_sequence_um'][:1]
    consistency = CASE['acceptance']['self_consistency']['inner_outer_scattering_max_difference_over_band_maximum']
    judged = []
    for mesh in meshes:
        row = rows(mesh, 'cpu', 'float64')[radius]
        judged.append(row)
        record(f'r={radius}/h={mesh}/cpu-float64', row)
        assert row['inner_outer_scattering_max_difference_over_band_maximum'] < consistency, 'inner and outer scattered-power surfaces disagree'
    g3.full_only('judged mesh h/4 of the Drude fixture')
    final = judged[-1]
    assert final['mesh_um'] == f['judged_mesh_um']
    limits = CASE['acceptance']['per_radius'][str(radius)]
    failures = []
    if final['max_scattering_relative_error'] > limits['scattering_relative_error']:
        failures.append(f'scattering {final["max_scattering_relative_error"]:.3f} > {limits["scattering_relative_error"]}')
    if final['max_absorption_relative_error'] > limits['absorption_relative_error']:
        failures.append(f'absorption {final["max_absorption_relative_error"]:.3f} > {limits["absorption_relative_error"]}')
    assert not failures, f'r={radius} um at h={final["mesh_um"]} um: '+'; '.join(failures)


def test_g3_05r5_drude_cuda_layer_a():
    g3.needs_cuda()
    f = CASE['fixture']
    rtol = CASE['acceptance']['layer_a_cuda_fp32_vs_cpu_fp64']['rtol']
    meshes = f['mesh_sequence_um'][:2] if g3.FULL else f['mesh_sequence_um'][:1]
    for mesh in meshes:
        cpu, cuda = rows(mesh, 'cpu', 'float64'), rows(mesh, 'cuda', 'float32')
        for sphere in f['spheres']:
            radius = sphere['radius_um']
            differences = {name: np.asarray(cuda[radius][name])/np.asarray(cpu[radius][name])-1 for name in ('scattering_um2', 'absorption_um2')}
            worst = max(float(abs(v).max()) for v in differences.values())
            row = dict(cuda[radius], layer_a_relative_difference={k: g3.listed(v) for k, v in differences.items()},
                       layer_a_max_relative_difference=worst, layer_a_rtol=rtol)
            record(f'r={radius}/h={mesh}/cuda-float32', row)
            assert worst <= rtol, f'layer A Drude r={radius} h={mesh}: {worst:.2e}'
