# ---------------------------------------------------------------------------
# COPIED, not invented. Source: the reference project 20261010_torchfdtd-lumerical-parity
#   src/parity.py
#   at commit 9b0f61a41b026342420422b7d208a046d7b6e45b
#   sha256(source) = 50b06d655d256d574b479bb6743f0ec55928def41b85697cdb7d658a31e98cca
#   copied 2026-10-18 by leg 1 (F0+F8). This project owns this copy now.
# ---------------------------------------------------------------------------
"""Parity adapter over the proven, unmodified TorchFDTD chain; no solver fork."""
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# CHANGED from the copy. The reference version inserted `src/torchfdtd_upstream`.
# Here the solver is named once, by `configs/baseline-pin.json`, and the launch
# snapshot's copy of it is used when running under tools/run.sh -- so the solver
# that runs is the solver the manifest records. `src/torchfdtd` is a machine-local
# link to the shared fork and is not tracked.
_solver_root = (Path(os.environ['PARITY_SNAPSHOT']) / 'src/solver'
                if os.environ.get('PARITY_SNAPSHOT') else ROOT / 'src/torchfdtd')
sys.path.insert(0, str(_solver_root))
from torchfdtd import (Project, Region, Material, Structure, Source, FieldMonitor,
                       SpectrumSettings, Boundaries, BoundaryFace, Simulation, normalize_flux)
from torchfdtd.injection import oneway_tables, oneway_plan

C0 = 299792458.

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + '\n')

def load_run():
    if 'RUN_DIR_ABS' not in os.environ:
        raise RuntimeError('Use tools/run.sh to execute simulations')
    config = json.loads(Path(os.environ['PARITY_CONFIG']).read_text())
    run = Path(os.environ['RUN_DIR_ABS'])
    write_json(os.environ['EFFECTIVE_JSON'], dict(config=config, solver_pin=json.loads((ROOT/'configs/baseline-pin.json').read_text()),
               conventions=dict(fourier='exp(+i omega t)', normal='+x', R='-signed reflected flux',
               T='signed transmitted flux', denominator='matched empty-domain flux; incident-line cross-check',
               geometry_units='um', time_units='s'), resolved_cases=[]))
    return config, run

def ppw_step(wavelength_min_um, index_max, ppw):
    return wavelength_min_um / (index_max * ppw)

def mesh_accuracy_step(wavelength_min_um, index_max, accuracy):
    if not isinstance(accuracy, int) or not 1 <= accuracy <= 8:
        raise ValueError('Mesh accuracy must be 1..8')
    return ppw_step(wavelength_min_um, index_max, 4 * accuracy + 2)

# CHANGED from the copy (leg 18): the absorber depth is an explicit input. The default stays 32
# because legs 3, 4 and 12 archived their numbers at 32; a caller that wants another depth says so.
DEFAULT_PML_CELLS = 32

def project_for(kind, index, ppw, duration_fs, transverse_cells=6, frequency_points=241,
                precision='float64', monitor_kernel='fused', graph_steps=32,
                pml_cells=DEFAULT_PML_CELLS):
    if isinstance(pml_cells, bool) or not isinstance(pml_cells, int) or pml_cells < 1:
        raise ValueError(f'pml_cells must be a positive integer, got {pml_cells!r}')
    dx = ppw_step(1.2, index, ppw)
    # Fixed physical interior, exterior PML cells explicit. Half-cell faces keep
    # the nominal slab thickness on the Ez grid; no staircase boundary ties.
    nx = 2 * math.ceil(4.0/dx) + 2 * pml_cells
    span = nx * dx
    width = transverse_cells * dx
    r = Region(size=(span,width,1), mesh=dx, mesh_steps=(dx,dx,1),
               material_sampling='yee', interface_method='staircase', pml_cells=pml_cells,
               backend='cuda', precision=precision, cuda_kernel='fused',
               cuda_monitor_kernel=monitor_kernel, snapshot_interval=10000,
               boundaries=Boundaries(y_min=BoundaryFace(kind='periodic'), y_max=BoundaryFace(kind='periodic')))
    r.steps = math.ceil(duration_fs * 1e-15 / r.time_step)
    snap = lambda x: round(x/dx) * dx
    thickness = 2.1/index
    center = 0.001  # fixed physical position; avoid exact node/face ties at every ladder level
    structures = []
    if kind == 'slab':
        structures = [Structure(center=(center,0,0), size=(thickness,width,1), material='dielectric')]
    elif kind == 'interface':
        left = dx/2
        right = span/2 + dx
        structures = [Structure(center=((left+right)/2,0,0),size=(right-left,width,1),material='dielectric')]
    elif kind != 'empty':
        raise ValueError('Unknown geometry')
    spec = SpectrumSettings(sampling='wavelength',wavelength_start=1.2,wavelength_stop=1.8,
                            frequency_points=frequency_points,apodization='none')
    # The default name is the archived name; a non-default depth is visible in it.
    name = f'{kind}-n{index}-ppw{ppw}' + ('' if pml_cells == DEFAULT_PML_CELLS else f'-pml{pml_cells}')
    p = Project(name=name, region=r,
                materials=[Material(name='dielectric',index=index)], structures=structures,
                sources=[Source(id='source',kind='plane',injection='oneway',normal='x',component='Ez',
                                center=(snap(-2.5),0,0),size=(0,width,0),wavelength=1.5,pulse_cycles=1)],
                monitors=[FieldMonitor(id=name,name=name,center=(snap(x),0,0),size=(0,width,1),normal='x',
                                       spectrum=spec.model_copy(deep=True))
                          for name,x in [('reflection',-2.),('transmission',2.),('scattered',-3.)]])
    return Project.model_validate(p.model_dump())

def geometry_assertions(p, expected_interface='staircase'):
    r = p.region
    assert r.material_sampling == 'yee' and r.interface_method == expected_interface
    assert not r.run_control.auto_shutoff
    assert p.sources[0].injection == 'oneway'
    assert np.isfinite(np.concatenate(r.mesh_nodes)).all()
    assert all(np.all(np.diff(a)>0) for a in r.mesh_nodes)
    oneway_plan(p.sources[0], r)
    lo, hi = r.interior_bounds(0)
    assert all(lo < m.center[0] < hi for m in p.monitors)
    if 'interface' in p.name:
        obj = p.structures[0]
        assert obj.center[0] + obj.size[0]/2 >= r.actual_size[0]/2
    if p.name.startswith('slab-n') and r.mesh_type == 'uniform':
        from torchfdtd.solver import voxelize
        epsilon, _ = voxelize(p)
        realized = np.count_nonzero(epsilon[:,0,0,2] > 1) * r.axis_steps[0]
        assert abs(realized - p.structures[0].size[0]) < 1e-10, 'Staircase slab thickness differs from physical thickness'
    return dict(shape=list(r.shape), dt_s=r.time_step, steps=r.steps,
                actual_duration_s=r.steps*r.time_step, interior_x_um=[lo,hi],
                mesh_nodes_um=[a.tolist() for a in r.mesh_nodes], project=p.model_dump())

def simulate(p, graph_steps=32):
    import torch
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = Simulation(p).run(cuda_graph=True,cuda_graph_steps=graph_steps)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    for monitor in result.frequency_fields:
        for key in ('fields','flux'):
            assert np.isfinite(monitor[key]).all(), f'Non-finite monitor {key}'
    return result, elapsed

def incident_power(p, frequencies):
    """Discrete incident-line power at an E node with interpolated Yee H.

    Correction tables encode C*h(t-dt/2)/n^2 and C*e(t), not field
    amplitudes. Recover the fields, align timestamps and average H on both
    sides using the discrete Yee dispersion relation. Reduced units cancel.
    """
    source, r = p.resolved_source(p.sources[0]), p.region
    e_correction, h_correction = oneway_tables(source,r)
    from torchfdtd.mesh import local_uniform_step
    axis,k,_=oneway_plan(source,r)
    dx=local_uniform_step(r,axis,k-1,k+2)
    courant = r.time_step*C0/(dx*1e-6)
    e = h_correction/courant
    h = e_correction*r.background_index**2/courant
    times = np.arange(1,r.steps+1)*r.time_step
    phase = np.exp(2j*np.pi*np.asarray(frequencies)[:,None]*times[None,:])
    E = phase @ e * r.time_step
    H = phase @ h * r.time_step * np.exp(-1j*np.pi*np.asarray(frequencies)*r.time_step)
    kdx = 2*np.arcsin(r.background_index*np.sin(np.pi*np.asarray(frequencies)*r.time_step)/courant)
    H *= (1 + np.exp(1j*kdx))/2
    area=r.actual_size[1]*1e-6
    if r.dimension=='3d':area*=r.actual_size[2]*1e-6
    return .5*np.real(E*H.conj())*area

def spectra(sample, reference):
    reflected=normalize_flux(sample.field_monitor('reflection'),reference.field_monitor('reflection'),subtract_incident=True)
    transmitted=normalize_flux(sample.field_monitor('transmission'),reference.field_monitor('transmission'))
    assert reflected['valid'].all() and transmitted['valid'].all(), 'Entire preregistered band must be supported'
    R,T=-reflected['ratio'],transmitted['ratio']
    assert np.isfinite(R).all() and np.isfinite(T).all()
    f=transmitted['frequency_hz']
    sourcepower=incident_power(reference.project,f)
    denominator=abs(reference.field_monitor('reflection')['flux'])
    assert np.isfinite(sourcepower).all() and np.all(sourcepower>0)
    return dict(wavelength_um=(C0/f*1e6).tolist(),frequency_hz=f.tolist(),R=R.tolist(),T=T.tolist(),
                sourcepower_reduced=sourcepower.tolist(),reference_power_reduced=denominator.tolist(),
                sourcepower_max_relative_error=float(np.max(abs(sourcepower/denominator-1))),
                empty_reflection_floor=float(np.max(abs(reference.field_monitor('scattered')['flux'])/denominator)),
                energy_residual=float(np.max(abs(R+T-1))),finite=True)
