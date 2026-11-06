import numpy as np
import pytest

from torchfdtd import Material, LorentzPole, fit_discrete_lorentz
from torchfdtd.materials import permittivity


SELLMEIER = ((0.6377579417631474, 0.09932777016261009),
             (3.5250338280915408, 0.034835477625238975))


def test_transparent_sellmeier_fit_tracks_the_discrete_visible_band():
    fit = fit_discrete_lorentz(sellmeier_coefficients=SELLMEIER, dt_s=2e-17,
                              wavelength_range_um=(.42, .67))
    material = fit.require_tolerance()
    wavelengths = np.array([.42, .45, .47, .51, .54, .57, .6, .635, .67])
    target = 1 + sum(b*wavelengths**2/(wavelengths**2-c) for b, c in SELLMEIER)
    measured = permittivity(material, 299792458/(wavelengths*1e-6), 2e-17)
    np.testing.assert_allclose(np.sqrt(measured.real), np.sqrt(target), atol=1e-3, rtol=0)
    assert len(material.oscillators) == 2 and all(g == 0 for _, _, g in material.oscillators)
    assert material.epsilon_inf >= 1 and material.fit_dt_s == 2e-17
    assert fit.report['fp32_coefficients'] and fit.report['max_abs_n_error'] <= 1e-3


def test_single_lorentz_declaration_is_fitted_without_changing_the_input():
    original = Material(name='single', model='multipole', epsilon_inf=1.4,
        poles=[LorentzPole(resonance_rad_s=9e15, strength_rad_s_squared=7e31, damping_rad_s=0)])
    before = original.model_dump()
    fit = fit_discrete_lorentz(original, dt_s=1e-17, wavelength_range_um=(.4, .8))
    assert fit.report['converged'] and len(fit.material.oscillators) == 1
    assert original.model_dump() == before
    wavelength = np.linspace(.4, .8, 19)
    frequency = 299792458/(wavelength*1e-6)
    np.testing.assert_allclose(np.sqrt(permittivity(fit.material, frequency, 1e-17).real),
                              np.sqrt(permittivity(original, frequency).real), atol=1e-3, rtol=0)


def test_optional_normal_incidence_phase_compensation_reports_its_scope():
    fit = fit_discrete_lorentz(sellmeier_coefficients=SELLMEIER, dt_s=2e-17,
        wavelength_range_um=(.42, .67), grid_spacing_m=20e-9)
    assert fit.report['converged']
    assert fit.report['spatial_compensation'] == 'normal-incidence 1D Yee phase'
    assert fit.report['max_abs_n_error'] <= 1e-3


def test_lossy_lorentz_input_uses_the_passive_complex_ade_fitter():
    original = Material(name='weakly lossy',model='multipole',epsilon_inf=1.5,
        poles=[LorentzPole(resonance_rad_s=8e15,strength_rad_s_squared=4e31,damping_rad_s=1e12)])
    fit = fit_discrete_lorentz(original,dt_s=1e-17,wavelength_range_um=(.4,.8))
    result = fit.require_tolerance()
    assert result.fit_dt_s == 1e-17
    assert all(s>0 and g>=0 for _,s,g in result.oscillators)
    assert fit.report['input_kind'] == 'Lorentz declaration'


def test_transparent_zero_resonance_input_uses_the_complex_ade_fitter():
    original = Material(name='transparent Drude',model='multipole',epsilon_inf=2.,
        poles=[LorentzPole(resonance_rad_s=0,strength_rad_s_squared=4e29,damping_rad_s=0)])
    fit = fit_discrete_lorentz(original,dt_s=1e-17,wavelength_range_um=(.4,.8))
    result = fit.require_tolerance()
    assert result.fit_dt_s == 1e-17
    assert any(rate==0 for rate,_,_ in result.oscillators)


@pytest.mark.parametrize('kwargs', [
    {'dt_s': 0}, {'dt_s': True}, {'dt_s': 2e-15}, {'sample_count': True},
    {'sample_count': 2}, {'wavelength_range_um': (.67, .42)}, {'tolerance': 0},
    {'sellmeier_coefficients': [(1, -1)]}, {'grid_spacing_m': 2e-6},
])
def test_invalid_fit_or_nyquist_input_is_rejected(kwargs):
    options = dict(sellmeier_coefficients=SELLMEIER, dt_s=2e-17,
                   wavelength_range_um=(.42, .67))
    options.update(kwargs)
    with pytest.raises(ValueError):
        fit_discrete_lorentz(**options)
