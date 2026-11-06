"""Point-trace spectral processing with explicit sampling and normalization.

Custom samples use dt*sum(E(t)*window(t)*exp(+2j*pi*f*t)). They are field
integrals, not source-normalized fields or power. Work is chunked on the CPU
after the GPU solver has returned its traces.
"""
import numpy as np

from .models import SpectrumSettings

C0 = 299792458.0


def apodization_window(times, settings: SpectrumSettings):
    t = np.asarray(times, dtype=np.float64)
    mode = settings.apodization
    if mode == 'none':
        return np.ones_like(t)
    if mode == 'hann':
        return np.hanning(len(t))
    d = t-settings.apodization_center
    if mode == 'start':
        d = np.minimum(d, 0)
    elif mode == 'end':
        d = np.maximum(d, 0)
    # Width is the FWHM of |window|^2, validated against installed v241.
    return np.exp(-2*np.log(2)*(d/settings.apodization_time_width)**2)


def frequency_samples(settings: SpectrumSettings):
    if settings.sampling=='custom':return np.asarray(settings.custom_frequencies_hz,dtype=np.float64)
    a, b, n = settings.wavelength_start, settings.wavelength_stop, settings.frequency_points
    if settings.sampling == 'fft':
        raise ValueError('FFT samples depend on the recorded time grid.')
    if settings.sampling=='chebyshev':
        u=(np.array([.5]) if n==1 else (1-np.cos(np.pi*np.arange(n)/(n-1)))/2) if settings.chebyshev_nodes=='lobatto' else (1-np.cos(np.pi*(np.arange(n)+.5)/n))/2
        if settings.chebyshev_wavelength:return C0/((a+(b-a)*u)*1e-6)
        return C0/(b*1e-6)+(C0/(a*1e-6)-C0/(b*1e-6))*u
    if n == 1:
        return np.array([C0/(1e-6*(a+b)/2) if settings.sampling == 'wavelength'
                         else .5*C0*1e6*(1/a+1/b)])
    if settings.sampling == 'wavelength':
        return C0/(np.linspace(a, b, n)*1e-6)
    return np.linspace(C0/(b*1e-6), C0/(a*1e-6), n)


def frequency_count(settings: SpectrumSettings):
    """len(frequency_samples(settings)) without building the samples."""
    if settings.sampling == 'fft':
        raise ValueError('FFT samples depend on the recorded time grid.')
    return len(settings.custom_frequencies_hz) if settings.sampling == 'custom' else settings.frequency_points


def highest_frequency(settings: SpectrumSettings):
    """max(frequency_samples(settings)) from the extreme sample alone, evaluated as frequency_samples evaluates it."""
    if settings.sampling == 'custom':
        return max(settings.custom_frequencies_hz)
    a, b, n = settings.wavelength_start, settings.wavelength_stop, settings.frequency_points
    if settings.sampling == 'fft':
        raise ValueError('FFT samples depend on the recorded time grid.')
    if settings.sampling == 'chebyshev':
        # u rises with the node index; the highest frequency is at the largest u, or at the smallest for wavelength nodes.
        k = 0 if settings.chebyshev_wavelength else n-1
        index = np.array([k], dtype=np.int64)
        u = ((np.array([.5]) if n == 1 else (1-np.cos(np.pi*index/(n-1)))/2) if settings.chebyshev_nodes == 'lobatto'
             else (1-np.cos(np.pi*(index+.5)/n))/2)
        if settings.chebyshev_wavelength:
            return float((C0/((a+(b-a)*u)*1e-6))[0])
        return float((C0/(b*1e-6)+(C0/(a*1e-6)-C0/(b*1e-6))*u)[0])
    if n == 1:
        return float(C0/(1e-6*(a+b)/2) if settings.sampling == 'wavelength' else .5*C0*1e6*(1/a+1/b))
    # linspace keeps its end points exactly, so the extreme sample is the shortest wavelength.
    return float(C0/(np.float64(a)*1e-6))


def direct_transform(times, signal, frequencies, window=None):
    t, e, f = np.asarray(times), np.asarray(signal), np.asarray(frequencies)
    if t.ndim != 1 or e.shape != t.shape or f.ndim != 1 or len(t) < 2:
        raise ValueError('DFT requires a one-dimensional trace with at least two time samples.')
    dt = t[1]-t[0]
    if dt <= 0 or not np.allclose(np.diff(t), dt, rtol=1e-7, atol=0):
        raise ValueError('DFT requires a uniform increasing time grid.')
    if not np.all(np.isfinite(f)) or np.any(f <= 0) or np.any(f >= .5/dt):
        raise ValueError('DFT frequencies must be positive and below the temporal Nyquist limit.')
    weighted = e if window is None else e*np.asarray(window)
    values = np.empty(len(f), dtype=np.complex128)
    # At most one million phase entries (~16 MB), regardless of run duration.
    chunk = max(1, 1_000_000//len(t))
    for start in range(0, len(f), chunk):
        block = f[start:start+chunk]
        values[start:start+chunk] = (np.exp(2j*np.pi*block[:, None]*t)@weighted)*dt
    return values


def point_spectrum(times, signal, settings: SpectrumSettings):
    window = apodization_window(times, settings)
    if len(times) < 2:
        return {'frequency_hz': np.array([]), 'value': np.array([], dtype=complex),
                'units': 'unavailable', 'transform': 'insufficient time samples'}
    dt = times[1]-times[0]
    if settings.sampling != 'fft':
        f = frequency_samples(settings)
        return {'frequency_hz': f, 'value': direct_transform(times, signal, f, window),
                'units': 'reduced field * s', 'transform': 'dt * sum(field * window * exp(+2pi i f t))'}
    # Compatibility with v0.1/v0.2 FFT magnitudes. No window-gain correction.
    if np.iscomplexobj(signal):
        f = np.fft.fftfreq(len(signal), dt)
        values = np.fft.fft(signal*window)/len(signal)
    else:
        f = np.fft.rfftfreq(len(signal), dt)
        values = np.fft.rfft(signal*window)/(len(signal)/2)
    keep = f > 0
    # FFT indices start at zero; preserve magnitude and include the time origin in phase.
    values = values*np.exp(-2j*np.pi*f*times[0])
    return {'frequency_hz': f[keep], 'value': values[keep], 'units': 'reduced field',
            'transform': 'FFT exp(-2pi i f t); scale 2/N real or 1/N complex; no window-gain correction'}
