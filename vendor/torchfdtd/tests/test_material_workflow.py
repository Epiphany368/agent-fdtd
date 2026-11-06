"""G6-01: one material workflow from a raw n/k or permittivity table to a fitted material with provenance.

``import_material_table`` hashes the raw file or text, runs the existing passive
fitter over the declared band, reports the fit residual, the time-discretization
n/k error of the trapezoidal ADE at a given timestep and the extrapolation warning
when a simulation band leaves the fitted band. The public table is the fused
silica Sellmeier formula of Malitson (1965), written to tests/fixtures/materials by
``sellmeier_sio2`` below with the formula stated in the file; the synthetic table
is an authored Lorentz permittivity table.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from torchfdtd import Material, Project, Region, Source, Structure
from torchfdtd.material_fit import (FitOptions, MaterialBandWarning, MaterialImportResult, discretization_report,
                                    fit_band_extrapolation, fit_material, import_material_table)
from torchfdtd.materials import permittivity
from torchfdtd.models import MaterialProvenance
from torchfdtd.server import create_app
from torchfdtd.solver import estimate

ROOT = Path(__file__).resolve().parent
SIO2 = ROOT / 'fixtures' / 'materials' / 'sio2_sellmeier_malitson1965.csv'
C0 = 299792458.0


def sellmeier_sio2(wavelength_um):
    """Malitson, J. Opt. Soc. Am. 55, 1205 (1965): fused silica at 20 C, stated valid from 0.21 to 3.71 um."""
    w2 = np.asarray(wavelength_um, dtype=float)**2
    b = (0.6961663, 0.4079426, 0.8974794)
    c = (0.0684043**2, 0.1162414**2, 9.896161**2)
    return np.sqrt(1+sum(bi*w2/(w2-ci) for bi, ci in zip(b, c)))


def lorentz_epsilon_text(unit='nm'):
    """Authored two-pole Lorentz permittivity table (the poles of tests/test_material_fit.py 'mixed' without Drude)."""
    wavelength = np.linspace(.8, 2.0, 61)
    omega = 2*np.pi*C0/(wavelength*1e-6)
    epsilon = np.full(wavelength.shape, 2.3+0j)
    for w0, a, g in ((1.7e15, 4e30, 2e14), (3e15, 2e30, 5e14)):
        epsilon += a/(w0*w0-omega*omega-1j*g*omega)
    scale = {'nm': 1000., 'um': 1.}[unit]
    rows = [f'wavelength_{unit},epsilon_real,epsilon_imag']
    rows += [f'{w*scale:.6f},{e.real:.10f},{e.imag:.10f}' for w, e in zip(wavelength, epsilon)]
    return '\n'.join(rows)+'\n', epsilon, wavelength


def test_fixture_table_is_the_stated_sellmeier_formula():
    rows = [line for line in SIO2.read_text(encoding='utf-8').splitlines() if line and not line.startswith('#')]
    assert rows[0] == 'wavelength_um,n,k'
    table = np.array([[float(v) for v in line.split(',')] for line in rows[1:]])
    assert table.shape == (81, 3) and table[0, 0] == .4 and table[-1, 0] == 2.
    np.testing.assert_allclose(table[:, 1], sellmeier_sio2(table[:, 0]), rtol=0, atol=6e-9)
    assert not table[:, 2].any()
    assert 'doi:10.1364/JOSA.55.001205' in SIO2.read_text(encoding='utf-8')


def test_public_sio2_table_records_provenance_and_fits_within_the_band():
    result = import_material_table(SIO2, source='Malitson 1965 Sellmeier (fused silica)',
                                   licence='Formula-generated test table; the formula is stated in the file header.',
                                   name='SiO2 (Malitson)', options=FitOptions(max_poles=3, tolerance=1e-4), imported='2026-09-22')
    assert isinstance(result, MaterialImportResult)
    p = result.provenance
    assert p.raw_sha256 == hashlib.sha256(SIO2.read_bytes()).hexdigest()
    assert p.source == 'Malitson 1965 Sellmeier (fused silica)' and p.licence.startswith('Formula-generated')
    assert p.file_name == SIO2.name and p.columns == 'nk' and p.wavelength_unit == 'um' and p.imported == '2026-09-22'
    assert result.material.provenance == p and result.material.name == 'SiO2 (Malitson)'
    assert result.converged and result.report['pole_count'] <= 3
    assert result.report['analytic']['normalized_rms'] <= 1e-4
    assert result.material.fit_band_um == (.4, 2.) and result.material.fit_dt_s is None
    assert result.discretization is None and result.warnings == []
    assert result.require_tolerance() is result.material
    # Held-out wavelengths against the formula itself, not the sampled table.
    held = np.linspace(.41, 1.99, 97)
    fitted = np.sqrt(permittivity(result.material, C0/(held*1e-6)))
    assert np.max(abs(fitted.real-sellmeier_sio2(held))) < 2e-4
    assert np.max(abs(fitted.imag)) < 1e-6
    assert result.as_dict()['provenance']['raw_sha256'] == p.raw_sha256


def test_discretization_error_is_the_bilinear_ade_response_and_second_order_in_dt():
    material = import_material_table(SIO2, source='Malitson 1965', options=FitOptions(max_poles=3, tolerance=1e-4)).material
    dt = 5e-17
    coarse, fine = discretization_report(material, dt), discretization_report(material, dt/2)
    assert coarse['band_um'] == [.4, 2.] and coarse['sample_count'] == 201 and coarse['dt_s'] == dt
    # Independent evaluation of the same poles with omega -> (2/dt) tan(omega dt / 2).
    wavelength = np.linspace(.4, 2., 201)
    omega = 2*np.pi*C0/(wavelength*1e-6)
    bilinear = 2/dt*np.tan(omega*dt/2)
    def epsilon(w):
        return material.epsilon_inf+sum(a/(w0*w0-w*w-1j*g*w) for w0, a, g in material.oscillators)
    delta = np.sqrt(epsilon(bilinear))-np.sqrt(epsilon(omega))
    assert coarse['max_abs_n_error'] == pytest.approx(np.max(abs(delta.real)), rel=1e-12)
    assert coarse['max_abs_n_error_at_um'] == pytest.approx(wavelength[np.argmax(abs(delta.real))])
    assert coarse['max_abs_k_error'] < 1e-8
    assert 3.8 < coarse['max_abs_n_error']/fine['max_abs_n_error'] < 4.2
    assert 'tan(omega dt/2)' in coarse['definition']
    with pytest.raises(ValueError, match='positive timestep'):
        discretization_report(material, 0)
    with pytest.raises(ValueError, match='Nyquist'):
        discretization_report(material, 2e-15)
    with pytest.raises(ValueError, match='Declare a wavelength band'):
        discretization_report(Material(name='constant', index=1.5), dt)
    explicit = discretization_report(Material(name='constant', index=1.5), dt, band_um=(1., 2.))
    assert explicit['max_abs_n_error'] == 0 and explicit['band_um'] == [1., 2.]


def test_extrapolation_warning_is_raised_kept_and_shared_with_the_estimate():
    inside = import_material_table(SIO2, source='Malitson 1965', options=FitOptions(max_poles=3, tolerance=1e-4),
                                   simulation_band_um=(.5, 1.6))
    assert inside.warnings == []
    with pytest.warns(MaterialBandWarning, match='0.3.*1 um lies outside the SiO2 fit band'):
        outside = import_material_table(SIO2, source='Malitson 1965', name='SiO2',
                                        options=FitOptions(max_poles=3, tolerance=1e-4), simulation_band_um=(.3, 1.))
    assert len(outside.warnings) == 1 and 'not validated' in outside.warnings[0]
    material = outside.material
    assert fit_band_extrapolation(material, (.4, 2.)) is None
    assert fit_band_extrapolation(material, (2.5, 2.5), label='probe').startswith('probe: source wavelength band 2.5')
    assert fit_band_extrapolation(Material(name='analytic', index=2.), (0.1, 9.)) is None
    with pytest.raises(ValueError, match='finite and positive'):
        fit_band_extrapolation(material, (0., 1.))
    # The solver's estimate raises the same message for a source whose band leaves the fit band.
    p = Project(region=Region(size=(4, 4, 4), mesh=.1, pml_cells=5, backend='cpu', steps=100),
                materials=[material], structures=[Structure(kind='sphere', radius=.5, material='SiO2')],
                sources=[Source(name='probe', center=(-.8, 0, 0), wavelength=1., pulse_cycles=1)])
    assert not any('fit band' in w for w in estimate(p)['warnings'])
    p.sources[0].wavelength = 2.5
    assert fit_band_extrapolation(material, (2.5, 2.5), label='probe') in estimate(p)['warnings']
    p.sources[0] = Source(name='wide', center=(-.8, 0, 0), pulse='broadband', time_definition='wavelength',
                          wavelength_start=.35, wavelength_stop=.8)
    assert any(w.startswith('wide: source wavelength band 0.35') and 'not validated' in w for w in estimate(p)['warnings'])


def test_synthetic_epsilon_text_table_hashes_the_text_and_fits():
    text, epsilon, wavelength = lorentz_epsilon_text('nm')
    result = import_material_table(text=text, kind='epsilon', unit='nm', source='authored two-pole Lorentz table',
                                   licence='synthetic', file_name='lorentz.csv', name='Lorentz pair',
                                   options=FitOptions(max_poles=3, tolerance=1e-6, include_drude=False), dt_s=4e-17)
    assert result.provenance.raw_sha256 == hashlib.sha256(text.encode('utf-8')).hexdigest()
    assert result.provenance.columns == 'epsilon' and result.provenance.wavelength_unit == 'nm'
    assert result.provenance.file_name == 'lorentz.csv'
    assert result.converged and result.report['pole_count'] == 2
    assert result.report['ade'] is not None and result.report['dt_s'] == 4e-17
    assert result.discretization['dt_s'] == 4e-17 and result.discretization['max_abs_n_error'] > 0
    assert result.material.fit_band_um == (.8, 2.) and result.material.fit_dt_s is None
    fitted = permittivity(result.material, C0/(wavelength*1e-6))
    assert np.max(abs(fitted-epsilon)/np.maximum(abs(epsilon), 1)) < 1e-5
    failed = import_material_table(text=text, kind='epsilon', unit='nm', source='authored',
                                   options=FitOptions(max_poles=1, tolerance=1e-9, include_drude=False))
    assert not failed.converged and failed.warnings and 'did not reach tolerance' in failed.warnings[0]
    with pytest.raises(ValueError, match='did not reach'):
        failed.require_tolerance()
    with pytest.raises(ValueError, match='exactly one of path or text'):
        import_material_table(SIO2, text=text, source='x')
    with pytest.raises(ValueError, match='exactly one of path or text'):
        import_material_table(source='x')
    with pytest.raises(ValueError, match='Name the source'):
        import_material_table(text=text, kind='epsilon', unit='nm', source='  ')
    with pytest.raises(ValueError, match='Header wavelength unit'):
        import_material_table(text=text, kind='epsilon', unit='um', source='x')


def test_project_json_keeps_provenance_fit_band_and_timestep(tmp_path):
    text, _, _ = lorentz_epsilon_text('um')
    result = import_material_table(text=text, kind='epsilon', unit='um', source='authored Lorentz', licence='CC0',
                                   name='Lorentz pair', options=FitOptions(max_poles=2, tolerance=1e-4, include_drude=False,
                                                                           target='ade', dt_s=4e-17, wavelength_range_um=(.9, 1.9)))
    # The table is the continuum response, so the ADE-target fit keeps a discretization residual (6e-5 here).
    material = result.require_tolerance()
    assert material.fit_band_um == (.9, 1.9) and material.fit_dt_s == 4e-17
    p = Project(region=Region(size=(4, 4, 4), mesh=.1, pml_cells=5, backend='cpu', steps=100), materials=[material],
                structures=[Structure(kind='sphere', radius=.5, material='Lorentz pair')], sources=[Source(center=(-.8, 0, 0))])
    path = tmp_path/'project.json'
    p.save(path)
    loaded = Project.load(path)
    saved = loaded.materials[0]
    assert saved.provenance == material.provenance and saved.provenance.licence == 'CC0'
    assert saved.fit_band_um == (.9, 1.9) and saved.fit_dt_s == 4e-17 and saved.samples == material.samples
    assert saved.poles == material.poles
    raw = json.loads(path.read_text(encoding='utf-8'))
    assert raw['materials'][0]['provenance']['raw_sha256'] == material.provenance.raw_sha256
    assert material.provenance.raw_sha256 in p.python_script()
    # Provenance is validated like every other field: a bad hash is refused, not defaulted.
    with pytest.raises(ValueError):
        MaterialProvenance(source='x', raw_sha256='abc')
    with pytest.raises(ValueError):
        Material(name='m', provenance={'source': 'x', 'raw_sha256': 'f'*64, 'columns': 'csv'})


def test_server_routes_surface_provenance_discretization_and_extrapolation():
    text, _, _ = lorentz_epsilon_text('um')
    with TestClient(create_app()) as client:
        request = dict(text=text, kind='epsilon', unit='um', reference='authored')
        data = client.post('/api/materials/data', json=request).json()
        assert 'provenance' not in data
        provenance = client.post('/api/materials/provenance', json=dict(request, source='authored Lorentz', licence='synthetic', file_name='lorentz.csv')).json()
        assert provenance['raw_sha256'] == hashlib.sha256(text.encode('utf-8')).hexdigest()
        assert provenance['source'] == 'authored Lorentz' and provenance['licence'] == 'synthetic' and provenance['file_name'] == 'lorentz.csv'
        assert client.post('/api/materials/provenance', json=dict(text=text, kind='epsilon', unit='um')).json() is None
        # The reference field names the source when no source is given (the browser fills it with the file name).
        named = client.post('/api/materials/provenance', json=dict(text=text, kind='epsilon', unit='um', reference='lorentz.csv')).json()
        assert named['source'] == 'lorentz.csv'
        assert client.post('/api/materials/provenance', json=dict(text='1,2', kind='epsilon', unit='um', source='x')).status_code == 422
        fit = client.post('/api/materials/fit', json=dict(data=data, provenance=provenance, name='Lorentz pair',
                          options=dict(max_poles=2, tolerance=1e-6, include_drude=False, dt_s=4e-17))).json()
        assert fit['material']['provenance'] == provenance
        assert fit['report']['converged'] and fit['discretization']['dt_s'] == 4e-17
        assert fit['discretization']['max_abs_n_error'] > 0 and 'tan(omega dt/2)' in fit['discretization']['definition']
        material = fit['material']
        preview = client.post('/api/materials/preview', params=dict(wavelength_start=1., wavelength_stop=1.5, dt_fs=.04), json=material).json()
        assert preview['provenance'] == provenance and preview['fit_band_um'] == [.8, 2.]
        assert preview['discretization']['band_um'] == [.8, 2.] and preview['discretization']['dt_s'] == pytest.approx(4e-17)
        project = Project(region=Region(size=(4, 4, 4), mesh=.1, pml_cells=5, backend='cpu', steps=100),
                          materials=[Material.model_validate(material)],
                          structures=[Structure(kind='sphere', radius=.5, material='Lorentz pair')],
                          sources=[Source(name='probe', center=(-.8, 0, 0), wavelength=2.5, pulse_cycles=1)])
        valid = client.post('/api/validate', json=project.model_dump(mode='json')).json()
        assert any(w.startswith('probe: source wavelength band 2.5') and 'Lorentz pair fit band' in w for w in valid['warnings'])
        assert valid['project']['materials'][0]['provenance'] == provenance
        project.sources[0].wavelength = 1.2
        valid = client.post('/api/validate', json=project.model_dump(mode='json')).json()
        assert not any('fit band' in w for w in valid['warnings'])


def test_fit_material_accepts_and_keeps_provenance_without_changing_the_fit():
    text, _, _ = lorentz_epsilon_text('um')
    from torchfdtd import OpticalData
    data = OpticalData.from_text(text, kind='epsilon', unit='um')
    provenance = MaterialProvenance(source='authored', raw_sha256='0'*64)
    options = FitOptions(max_poles=2, tolerance=1e-6, include_drude=False)
    plain, kept = fit_material(data, options=options), fit_material(data, options=options, provenance=provenance)
    assert plain.material.provenance is None and kept.material.provenance == provenance
    assert kept.material.poles == plain.material.poles and kept.material.epsilon_inf == plain.material.epsilon_inf
