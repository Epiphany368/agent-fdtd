import math

import fdtd
import numpy as np
import pytest

from torchfdtd import Project, Region, Source, Monitor, Simulation
from torchfdtd.models import Boundaries, BoundaryFace
from torchfdtd.boundaries import YeeGrid


def paired(kind='periodic', dimensions=3):
    return Boundaries(**{axis+'_'+side: BoundaryFace(kind=kind)
                         for axis in 'xyz'[:dimensions] for side in ('min', 'max')})


def test_boundary_pair_validation_and_source_extent():
    with pytest.raises(ValueError, match='paired'):
        Region(boundaries=Boundaries(x_min=BoundaryFace(kind='periodic')))
    with pytest.raises(ValueError, match='Bloch phase'):
        Region(bloch_phase=(.3, 0, 0))
    r = Region(size=(4, 1, 1), mesh=.1, pml_cells=5,
               boundaries=Boundaries(y_min=BoundaryFace(kind='periodic'), y_max=BoundaryFace(kind='periodic')))
    Project(region=r, sources=[Source(kind='plane', size=(0, 1, 0))])
    with pytest.raises(ValueError, match='non-PML'):
        Project(region=r, sources=[Source(kind='plane', size=(0, 1.2, 0))])
    asym = Region(size=(5, 4, 1), mesh=.1, pml_cells=5,
                  boundaries=Boundaries(x_min=BoundaryFace(layers=12), x_max=BoundaryFace(layers=4)))
    assert asym.interior_bounds(0) == pytest.approx((-1.3, 2.1))


@pytest.mark.parametrize('dimension', ['2d', '3d'])
@pytest.mark.parametrize('kind', ['periodic', 'bloch'])
def test_discrete_plane_wave_eigenmode(dimension, kind):
    """Analytic Yee dispersion checks all curls and the exact N*dx Bloch period."""
    dims = 2 if dimension == '2d' else 3
    phase = (.37, -.61, .22 if dims == 3 else 0) if kind == 'bloch' else (0, 0, 0)
    r = Region(dimension=dimension, size=(2.4, 2, 1.6), mesh=.1,
               boundaries=paired(kind, dims), bloch_phase=phase, precision='float64')
    fdtd.set_backend('numpy')
    g = YeeGrid(r)
    q = np.array([(2*np.pi*m+ph)/n if n > 1 else 0 for m, ph, n in zip((1, -2, 1), phase, r.shape)])
    k = np.sin(q/2)
    pol = np.array([0., 0., 1.]) if dims == 2 else np.cross(k, [0, 0, 1.])
    pol /= np.linalg.norm(pol)
    omega = 2*np.arcsin(g.courant_number*np.linalg.norm(k))
    coords = np.indices(r.shape)
    base = np.exp(1j*np.einsum('i,i...->...', q, coords))
    electric = base[..., None]*pol*np.exp(.5j*q)
    magnetic = base[..., None]*(np.cross(k, pol)/np.linalg.norm(k))*np.exp(.5j*(q.sum()-q-omega))
    g.E[:] = electric if kind == 'bloch' else electric.real
    g.H[:] = magnetic if kind == 'bloch' else magnetic.real
    steps = 67
    for _ in range(steps):
        g.update_E()
        g.update_H()
    expected_e, expected_h = electric*np.exp(-1j*steps*omega), magnetic*np.exp(-1j*steps*omega)
    if kind == 'periodic':
        expected_e, expected_h = expected_e.real, expected_h.real
    np.testing.assert_allclose(g.E, expected_e, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(g.H, expected_h, atol=2e-13, rtol=2e-13)


def pulse_project(backend='cpu', bloch=False):
    return Project(region=Region(size=(8, 1, 1), mesh=.05, steps=900, pml_cells=14,
                                 backend=backend, precision='float64', snapshot_interval=20,
                                 boundaries=Boundaries(
                                     x_min=BoundaryFace(layers=16, kappa=3, alpha=.02, alpha_polynomial=1),
                                     x_max=BoundaryFace(layers=20, kappa=3, alpha=.02, alpha_polynomial=1),
                                     y_min=BoundaryFace(kind='bloch' if bloch else 'periodic'),
                                     y_max=BoundaryFace(kind='bloch' if bloch else 'periodic')),
                                 bloch_phase=(0, .4 if bloch else 0, 0)),
                   sources=[Source(kind='plane', center=(-1, 0, 0), size=(0, 1, 0), wavelength=1, pulse_cycles=2)],
                   monitors=[Monitor(center=(0, 0, 0))])


def test_custom_cpml_absorbs_periodic_sheet_pulse():
    p = pulse_project()
    r = Simulation(p).run()
    signal = r.signals[:, 0]
    # The incident packet has passed before 40 fs; roundtrip reflections arrive later.
    incident = np.max(abs(signal[r.times < 40e-15]))
    residual = np.max(abs(signal[r.times > 55e-15]))
    assert residual / incident < .005, residual / incident
    # A periodic uniform source must remain uniform along the repeated direction.
    assert np.max(np.ptp(r.electric[..., 2], axis=1)) < 1e-12


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_bloch_cuda_complex_graph_and_result_roundtrip(precision, tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA GPU not available')
    p = pulse_project(bloch=True)
    p.region.steps = 250
    p.region.precision = precision
    cpu = Simulation(p).run()
    p.region.backend = 'cuda'
    eager = Simulation(p).run(cuda_graph=False)
    graph = Simulation(p).run()
    tolerance = 2e-5 if precision == 'float32' else 2e-12
    assert graph.summary['complex_fields'] and graph.summary['cuda_graph']
    assert np.iscomplexobj(graph.electric) and np.max(abs(graph.electric.imag)) > 1e-5
    np.testing.assert_allclose(graph.electric, cpu.electric, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(graph.signals, eager.signals, atol=tolerance, rtol=tolerance)
    graph.save(tmp_path / 'bloch.npz')
    data = np.load(tmp_path / 'bloch.npz')
    np.testing.assert_array_equal(data['E'], graph.electric)
    np.testing.assert_array_equal(data['signals'], graph.signals)
    monitor = graph.monitor_data()[0]
    assert monitor['complex'] and max(abs(np.asarray(monitor['signal_imag']))) > 1e-5


def test_zero_bloch_phase_equals_periodic():
    p = pulse_project()
    p.region.steps = 100
    real = Simulation(p).run()
    for face in p.region.boundaries.pair(1):
        face.kind = 'bloch'
    complex_ = Simulation(p).run()
    np.testing.assert_allclose(complex_.electric.real, real.electric, atol=1e-13, rtol=1e-13)
    assert not np.any(complex_.electric.imag)


def test_bloch_sheet_matches_oblique_discrete_vacuum_propagation():
    p = pulse_project(bloch=True)
    p.monitors = [Monitor(center=(0, 0, 0)), Monitor(center=(.5, 0, 0)), Monitor(center=(0, .25, 0))]
    run = Simulation(p).run()
    frequency = 299792458/1e-6
    phasor = np.exp(2j*np.pi*frequency*run.times)@run.signals
    courant = .99/math.sqrt(2)
    omega = 2*np.pi*frequency*(run.times[1]-run.times[0])
    qy = p.region.bloch_phase[1]/p.region.shape[1]
    qx = 2*np.arcsin(np.sqrt((np.sin(omega/2)/courant)**2-np.sin(qy/2)**2))
    assert abs(phasor[1]/phasor[0]-np.exp(1j*qx*10)) < .004
    assert abs(phasor[2]/phasor[0]-np.exp(1j*qy*5)) < 1e-10
