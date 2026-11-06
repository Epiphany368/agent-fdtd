import json

import numpy as np
import pytest

from torchfdtd import Project, Monitor, Region, Source, Simulation, SpectrumSettings, FDTD
from torchfdtd.spectra import apodization_window, direct_transform, frequency_samples, point_spectrum


def test_dft_gaussian_has_analytic_complex_amplitude_and_phase():
    t = np.arange(0, 1, 1e-4)
    sigma, center, carrier = .035, .43, 37.2
    e = np.exp(-.5*((t-center)/sigma)**2)*np.exp(-2j*np.pi*carrier*t)
    f = np.linspace(25, 50, 61)
    exact = np.sqrt(2*np.pi)*sigma*np.exp(-2*np.pi**2*sigma**2*(f-carrier)**2)*np.exp(2j*np.pi*(f-carrier)*center)
    np.testing.assert_allclose(direct_transform(t, e, f), exact, atol=1e-14)


def test_full_apodization_against_gaussian_product_integral():
    t = np.arange(0, 1, 1e-4)
    sigma, center, carrier = .045, .48, 34
    settings = SpectrumSettings(apodization='full', apodization_center=.43, apodization_time_width=.08)
    e = np.exp(-.5*((t-center)/sigma)**2)*np.exp(-2j*np.pi*carrier*t)
    sigma_w = .08/(2*np.sqrt(np.log(2)))
    variance = 1/(1/sigma**2+1/sigma_w**2)
    mean = variance*(center/sigma**2+.43/sigma_w**2)
    factor = np.exp(-.5*(center-.43)**2/(sigma**2+sigma_w**2))
    f = np.array([28, 34, 40])
    exact = factor*np.sqrt(2*np.pi*variance)*np.exp(-2*np.pi**2*variance*(f-carrier)**2)*np.exp(2j*np.pi*(f-carrier)*mean)
    np.testing.assert_allclose(direct_transform(t, e, f, apodization_window(t, settings)), exact, atol=1e-14)
    # Power FWHM and one-sided pass regions are physical definitions.
    times = np.array([.39, .43, .47])
    full = apodization_window(times, settings)
    np.testing.assert_allclose(full**2, [.5, 1, .5])
    settings.apodization = 'start'
    np.testing.assert_allclose(apodization_window(times, settings)[1:], 1)
    settings.apodization = 'end'
    np.testing.assert_allclose(apodization_window(times, settings)[:2], 1)


def test_sampling_nyquist_and_legacy_fft():
    s = SpectrumSettings(sampling='wavelength', frequency_points=51)
    np.testing.assert_allclose(299792458/frequency_samples(s)*1e6, np.linspace(1.3, 1.8, 51))
    s.sampling = 'frequency'
    np.testing.assert_allclose(np.diff(frequency_samples(s)), np.diff(frequency_samples(s))[0])
    with pytest.raises(ValueError, match='Maximum wavelength'):
        SpectrumSettings(wavelength_start=2, wavelength_stop=1)
    with pytest.raises(ValueError, match='Nyquist'):
        Project(region=Region(size=(40,40,1),mesh=1,pml_cells=3),
                monitors=[Monitor(spectrum=SpectrumSettings(sampling='wavelength', wavelength_start=.1))])
    with pytest.raises(ValueError, match='uniform'):
        direct_transform(np.array([0,1,3]), np.ones(3), np.array([.1]))
    t = np.arange(64)*.001
    y = np.cos(2*np.pi*80*t)
    data = point_spectrum(t, y, SpectrumSettings())
    np.testing.assert_allclose(abs(data['value']), abs(np.fft.rfft(y*np.hanning(64)))[1:]/32)


def test_native_dft_windows_exports_and_familiar_si_properties(tmp_path):
    f = FDTD()
    f.adddft(name='out', minimum_wavelength=1.2e-6, maximum_wavelength=1.9e-6,
             frequency_points=41, apodization='Full', apodization_center=16e-15, apodization_time_width=9e-15)
    assert f.project.monitors[0].spectrum.wavelength_start == pytest.approx(1.2)
    p = Project(region=Region(size=(4,4,1),mesh=.1,pml_cells=5,steps=260,backend='cpu'),
                sources=[Source(wavelength=1.55,pulse_cycles=1,center=(-.5,0,0))],
                monitors=f.project.monitors)
    result = Simulation(p).run()
    data = result.monitor_data()[0]
    assert len(data['spectrum']) == 41
    assert data['spectrum_units'] == 'reduced field * s'
    assert max(data['spectrum']) > 0
    result.save(tmp_path/'data.npz')
    stored = np.load(tmp_path/'data.npz')
    np.testing.assert_array_equal(stored['monitor_0_spectrum'], result.spectra[0]['value'])
    assert json.loads(str(stored['monitor_spectra']))[0]['settings']['apodization'] == 'full'
