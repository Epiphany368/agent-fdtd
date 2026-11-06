"""G6-02: the source preview reports the realized injection, not a drawing of an unsupported source.

Spatial amplitude and Bloch phase on the source cells, the polarization vector,
the effective bandwidth of the mesh-time samples (declared as the band above
1 percent of the peak amplitude) and the incidence definition: every oblique
source is fixed k_parallel (the Bloch phase), the angle varies across the band,
and a fixed-angle request is refused with the registry message.
"""
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from torchfdtd import Project, Region, Source, BoundaryFace
from torchfdtd.capabilities import INCIDENCE
from torchfdtd.models import demo_project
from torchfdtd.server import create_app
from torchfdtd.solver import C0
from torchfdtd.source_preview import BANDWIDTH_FRACTION, effective_bandwidth, preview_source
from torchfdtd.waveforms import pulse_parameters


def bloch_sheet(angle_deg=20., wavelength=1.02, period=1.2, mesh=.03, steps=1500, **source):
    """The G3-08 grating cell without its ridge: a Bloch sheet along x, propagation along +y."""
    kx = 2*math.pi/wavelength*math.sin(math.radians(angle_deg))
    r = Region(dimension='2d', size=(period, 2.2, 1.), mesh=mesh, pml_cells=10, steps=steps, backend='cpu', precision='float64',
               material_sampling='yee', boundaries={'x_min': BoundaryFace(kind='bloch'), 'x_max': BoundaryFace(kind='bloch')},
               bloch_phase=(kx*period, 0., 0.))
    return Project(region=r, sources=[Source(id='sheet', name='sheet', kind='plane', normal='y', center=(0., -.6, 0.), size=(period, 0., 1.),
                                             component='Ez', wavelength=wavelength, pulse_cycles=3, **source)])


def test_bloch_sheet_is_fixed_k_parallel_with_the_angle_varying_across_the_band():
    p = bloch_sheet()
    r = p.region
    out = preview_source(p, 'sheet')
    incidence = out['incidence']
    assert incidence['kind'] == 'fixed_k_parallel' and incidence['fixed_angle_supported'] is False
    assert incidence['statement'] == INCIDENCE['fixed_k_parallel']['description']
    assert incidence['fixed_angle_message'] == INCIDENCE['fixed_angle']['message']
    kx = r.bloch_phase[0]/r.actual_size[0]
    assert incidence['k_parallel_rad_per_um'] == {'x': pytest.approx(kx)}
    angles = incidence['angle_deg']
    assert angles['carrier']['angle_deg'] == pytest.approx(20., abs=1e-9)
    # Longer wavelength, larger angle: sin(theta) = k_parallel / k0.
    assert angles['band_low']['angle_deg'] > angles['band_peak']['angle_deg'] > angles['band_high']['angle_deg']
    for sample in angles.values():
        assert sample['angle_deg'] == pytest.approx(math.degrees(math.asin(kx*sample['wavelength_um']/(2*math.pi))))
        assert sample['reason'] is None
    # Realized spatial phase on the sheet cells: exp(+i phi (x - x0) / L), reported relative to the first cell.
    term, = out['spatial']
    assert term['component'] == 'Ez' and term['kind'] == 'soft sheet' and term['amplitude'] == 1.
    x = np.asarray(term['positions_um']['x'])
    assert len(x) == r.shape[0] and term['cells'][0] == [0, r.shape[0]]
    expected = r.bloch_phase[0]*(x-x[0])/r.actual_size[0]
    np.testing.assert_allclose(term['phase_rad_by_axis']['x'], expected, atol=1e-12)
    assert 'Bloch spatial phase' in term['profile'] and 'first cell' in term['phase_reference']
    assert out['polarization']['vector'] == [0., 0., 1.] and out['polarization']['family'] == 'electric'
    with pytest.raises(ValueError, match='Fixed-angle broadband injection is not implemented'):
        preview_source(p, 'sheet', incidence='fixed_angle')
    assert preview_source(p, 'sheet', incidence='fixed_k_parallel')['incidence'] == incidence
    with pytest.raises(ValueError, match='incidence must be'):
        preview_source(p, 'sheet', incidence='oblique')


def test_evanescent_band_edge_is_named_not_clipped():
    # 80 degrees at the carrier: the long-wavelength edge of the band has k_parallel > k0.
    out = preview_source(bloch_sheet(angle_deg=80.), 'sheet')
    angles = out['incidence']['angle_deg']
    assert angles['carrier']['angle_deg'] == pytest.approx(80.)
    assert angles['band_low']['angle_deg'] is None and 'evanescent' in angles['band_low']['reason']
    assert angles['band_high']['angle_deg'] is not None


def test_normal_sheet_point_oneway_and_tfsf_spatial_records():
    p = demo_project('scatterer')
    p.sources = [Source(id='source', kind='plane', center=(-2.5, 0, 0), size=(0, 4, 0))]  # the demo itself now carries a one-way sheet
    p = Project.model_validate(p.model_dump())
    out = preview_source(p, p.sources[0].id)
    assert out['incidence']['kind'] == 'normal' and out['incidence']['k_parallel_rad_per_um'] == {}
    term, = out['spatial']
    assert term['kind'] == 'soft sheet' and term['phase_rad_by_axis'] == {} and 'no spatial phase' in term['profile']
    assert term['cells'][0][1]-term['cells'][0][0] == 1 and len(term['positions_um']['y']) == term['cells'][1][1]-term['cells'][1][0]
    p = demo_project('3d')
    out = preview_source(p, p.sources[0].id)
    assert out['incidence']['kind'] == 'not applicable' and out['spatial'][0]['kind'] == 'point'
    assert [b-a for a, b in out['spatial'][0]['cells']] == [1, 1, 1]
    oneway = Project(region=Region(dimension='2d', size=(4., 2., 1.), mesh=.1, pml_cells=5, steps=400, backend='cpu',
                                   boundaries={'y_min': BoundaryFace(kind='periodic'), 'y_max': BoundaryFace(kind='periodic')}),
                     sources=[Source(id='ow', kind='plane', injection='oneway', normal='x', center=(-1., 0., 0.), size=(0., 2., 0.), component='Ez')])
    out = preview_source(oneway, 'ow')
    term, = out['spatial']
    assert term['kind'] == 'one-way plane' and term['cells'][0] == [out['plane']['electric_index'], out['plane']['electric_index']+1]
    assert term['cells'][1] == [0, 20] and 'normal incidence' in term['profile'] and out['incidence']['kind'] == 'normal'
    tfsf = Project(region=Region(dimension='2d', size=(4., 4., 1.), mesh=.1, pml_cells=5, steps=400, backend='cpu'),
                   sources=[Source(id='box', kind='tfsf', normal='x', center=(0., 0., 0.), size=(2., 2., 1.), component='Ez')])
    out = preview_source(tfsf, 'box')
    term, = out['spatial']
    assert term['kind'] == 'tfsf box' and term['cells'] == [[lo, hi] for lo, hi in zip(out['tfsf']['lower_node_indices'], out['tfsf']['upper_node_indices'])]
    assert term['phase_rad_by_axis'] == {}


def test_vector_polarization_and_amplitude_weights():
    p = demo_project('3d')
    p.sources[0] = Source(id='v', kind='point', center=(0., 0., 0.), component='Ex', theta=60., phi=30., amplitude=.5)
    out = preview_source(p, 'v')
    st, ct = math.sin(math.radians(60)), math.cos(math.radians(60))
    sp, cp = math.sin(math.radians(30)), math.cos(math.radians(30))
    np.testing.assert_allclose(out['polarization']['vector'], [st*cp, st*sp, ct], atol=1e-12)
    assert out['polarization']['theta_deg'] == 60 and out['polarization']['phi_deg'] == 30
    assert {t['component'] for t in out['spatial']} == {'Ex', 'Ey', 'Ez'}
    for term in out['spatial']:
        assert term['amplitude'] == pytest.approx(.5*abs(out['polarization']['components'][term['component']]))


def test_effective_bandwidth_matches_the_gaussian_spectrum_and_names_a_zero_spectrum():
    p = bloch_sheet(steps=4000)
    out = preview_source(p, 'sheet')
    band = out['bandwidth']
    assert band['threshold_fraction'] == BANDWIDTH_FRACTION == .01 and '0.01 of its peak' in band['definition'] and '-40 dB' in band['definition']
    pulse = pulse_parameters(p.sources[0])
    # |X(f)| of a Gaussian envelope exp(-t^2 / (2 sigma^2)) is exp(-2 pi^2 sigma^2 (f - f0)^2): 1 percent at this half width.
    half = math.sqrt(math.log(100)/2)/(math.pi*pulse.sigma_s)
    resolution = 1/(p.region.steps*p.region.time_step)
    assert band['frequency_resolution_hz'] == pytest.approx(resolution, rel=1e-9)
    assert band['peak_frequency_hz'] == pytest.approx(pulse.frequency_hz, abs=resolution)
    assert band['frequency_hz'][0] == pytest.approx(pulse.frequency_hz-half, abs=1.5*resolution)
    assert band['frequency_hz'][1] == pytest.approx(pulse.frequency_hz+half, abs=1.5*resolution)
    assert band['wavelength_um'] == [pytest.approx(C0/band['frequency_hz'][1]*1e6), pytest.approx(C0/band['frequency_hz'][0]*1e6)]
    assert band['declared_wavelength_um'] is None and band['reason'] is None
    p.sources[0].enabled = False
    disabled = preview_source(p, 'sheet')
    assert disabled['bandwidth']['frequency_hz'] is None and 'zero' in disabled['bandwidth']['reason']
    assert disabled['spatial'] == [] and disabled['incidence']['kind'] == 'fixed_k_parallel'
    assert 'band_low' not in disabled['incidence']['angle_deg'] and 'carrier' in disabled['incidence']['angle_deg']
    broadband = bloch_sheet(pulse='broadband', time_definition='wavelength', wavelength_start=.9, wavelength_stop=1.2)
    assert preview_source(broadband, 'sheet')['bandwidth']['declared_wavelength_um'] == [.9, 1.2]
    direct = effective_bandwidth([1., 2., 3., 4.], [0., 1., .5, .001])
    assert direct['frequency_hz'] == [2., 3.] and direct['peak_frequency_hz'] == 2.


def test_registry_and_server_refuse_the_fixed_angle_definition():
    assert INCIDENCE['fixed_angle']['status'] == 'rejected' and INCIDENCE['fixed_angle']['feature_inventory'] == ('source.angle', 'boundary.bfast')
    assert INCIDENCE['fixed_angle']['code_path'] == 'torchfdtd/source_preview.py::preview_source'
    p = bloch_sheet()
    with TestClient(create_app()) as client:
        registry = client.get('/api/capabilities').json()['combinations']['incidence']
        assert registry['fixed_angle']['message'] == INCIDENCE['fixed_angle']['message']
        features = {f['id']: f for f in client.get('/api/capabilities').json()['features']}
        assert features['source.angle']['native'] == 'missing' and features['boundary.bfast']['native'] == 'missing'
        ok = client.post('/api/sources/sheet/preview', json=p.model_dump(mode='json'))
        assert ok.status_code == 200 and ok.json()['incidence']['kind'] == 'fixed_k_parallel'
        same = client.post('/api/sources/sheet/preview', params={'incidence': 'fixed_k_parallel'}, json=p.model_dump(mode='json'))
        assert same.status_code == 200
        refused = client.post('/api/sources/sheet/preview', params={'incidence': 'fixed_angle'}, json=p.model_dump(mode='json'))
        assert refused.status_code == 422 and refused.json()['detail'] == INCIDENCE['fixed_angle']['message']
        assert client.post('/api/sources/missing/preview', json=p.model_dump(mode='json')).status_code == 404
