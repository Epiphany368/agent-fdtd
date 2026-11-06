"""PEC cavity resonance measured from a point-monitor spectrum.

The existing cavity checks seed the exact discrete eigenvector and assert its
phase advance. This test excites a closed PEC cavity with a point source, as
a user would, and locates the TM110 resonance in the monitor spectrum. The
precise oracle is the discrete Yee eigenfrequency of the cavity; the
continuum eigenfrequency is reported against the program's 1 percent cavity
threshold. Both are closed forms that share nothing with the update code.
"""
import math
import numpy as np
import pytest
import torch

from torchfdtd import Project, Region, Source, Monitor, AdjointOptions, DifferentiableSimulation
from torchfdtd.models import Boundaries, BoundaryFace
from torchfdtd.boundaries import material_shape
from torchfdtd.solver import field_axes

C0 = 299792458.0
CELLS, MESH, STEPS = 8, .1, 4000
SOURCE_END_STEPS = 600          # past 8 sigma of the 2-cycle Gaussian for both indices (477 steps at n = 1.5)
ESTIMATOR_FLOOR = 1e-3          # declared budget of the DFT peak location
CONTINUUM_LIMIT = 1e-2          # completion_gates.json cavity_resonant_frequency_relative_error


def cavity_project(wavelength_um):
    faces = {a+'_'+s: BoundaryFace(kind='pec') for a in 'xyz' for s in ('min', 'max')}
    region = Region(dimension='3d', size=(CELLS*MESH,)*3, mesh=MESH, steps=STEPS, boundaries=Boundaries(**faces),
                    material_sampling='yee', precision='float32', backend='cpu', cuda_kernel='torch')
    at = lambda component, index: tuple(float(axis[i]) for axis, i in zip(field_axes(region, component), index))
    half = CELLS//2
    p = Project(region=region,
                sources=[Source(center=at('Ez', (half, half, half)), component='Ez', pulse='gaussian',
                                wavelength=wavelength_um, pulse_cycles=2)],
                monitors=[Monitor(center=at('Ez', (3, 2, 3)), component='Ez'),
                          Monitor(center=at('Ez', (5, 3, 1)), component='Ez')])
    return Project.model_validate(p.model_dump())


def eigenfrequencies(region, index):
    """Continuum and discrete Yee TM110 angular frequencies of the cubic cavity, in Hz."""
    length = CELLS*MESH*1e-6
    dt = region.time_step
    courant = C0*dt/(MESH*1e-6)
    continuum = C0/index*math.pi*math.sqrt(2)/length/(2*math.pi)
    q = math.pi/CELLS
    discrete = 2*math.asin(courant/index*math.sqrt(2*math.sin(q/2)**2))/dt/(2*math.pi)
    return continuum, discrete


def peak_frequency(result, column, centre_hz, window):
    """Two-stage DFT peak search with a parabolic refinement on the fine grid."""
    coarse = centre_hz*torch.linspace(.85, 1.15, 601, dtype=torch.float64)
    magnitude = result.spectrum(coarse, window=window)[:, column].abs()
    i = int(magnitude.argmax())
    assert 0 < i < len(coarse)-1, 'the resonance must lie inside the search window'
    step = float(coarse[1]-coarse[0])
    fine = torch.linspace(float(coarse[i])-2*step, float(coarse[i])+2*step, 401, dtype=torch.float64)
    magnitude = result.spectrum(fine, window=window)[:, column].abs()
    j = int(magnitude.argmax())
    assert 0 < j < len(fine)-1
    a, b, c = (float(magnitude[j+k]) for k in (-1, 0, 1))
    shift = .5*(a-c)/(a-2*b+c)
    return float(fine[j])+shift*float(fine[1]-fine[0])


@pytest.mark.parametrize('index', [1., 1.5])
def test_pec_cavity_tm110_resonance_matches_discrete_then_continuum_eigenfrequency(index):
    probe = cavity_project(1.)
    continuum, discrete = eigenfrequencies(probe.region, index)
    p = cavity_project(C0/continuum*1e6)
    epsilon = index**2*torch.ones(material_shape(p.region))
    with torch.no_grad():
        result = DifferentiableSimulation(p, AdjointOptions(checkpoints=2))(epsilon)
    assert result.signals.shape == (STEPS, 2)
    window = torch.ones(STEPS, dtype=torch.float64)
    window[:SOURCE_END_STEPS] = 0
    ringing = result.signals[SOURCE_END_STEPS:].abs().max()
    assert ringing > 1e-3, 'the cavity must ring after the source has ended'
    measured = [peak_frequency(result, column, continuum, window) for column in range(2)]
    for f in measured:
        discrete_error = abs(f/discrete-1)
        continuum_error = abs(f/continuum-1)
        assert discrete_error < ESTIMATOR_FLOOR
        assert continuum_error < CONTINUUM_LIMIT
        # The measurement resolves numerical dispersion: it sits on the discrete
        # eigenfrequency, not on the continuum one.
        assert abs(f-discrete) < abs(f-continuum)
    print({'index': index, 'continuum_hz': continuum, 'discrete_hz': discrete, 'measured_hz': measured,
           'discrete_relative_error': [abs(f/discrete-1) for f in measured],
           'continuum_relative_error': [abs(f/continuum-1) for f in measured],
           'numerical_dispersion': discrete/continuum-1, 'time_step_s': result.time_step,
           'ringing_amplitude': float(ringing)})
