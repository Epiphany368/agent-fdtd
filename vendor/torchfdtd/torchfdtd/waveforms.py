"""Independent temporal pulse definitions inferred from controlled public outputs.

No vendor runtime, stored waveform cache or fitted per-project coefficients.
"""
import math
from dataclasses import asdict, dataclass

import numpy as np

C0=299792458.0
TAIL_INNER=math.sqrt(2*math.log(1e4))
TAIL_OUTER=1.1*TAIL_INNER


@dataclass(frozen=True)
class PulseParameters:
    frequency_hz: float
    sigma_s: float
    offset_s: float
    frequency_span_hz: float
    chirped: bool

    def as_dict(self):
        return {**asdict(self),'pulse_length_s':2*math.sqrt(math.log(2))*self.sigma_s,
                'center_wavelength_um':C0/self.frequency_hz*1e6}


def pulse_parameters(source):
    frequency=C0/(source.wavelength*1e-6)
    span=source.chirp_bandwidth_hz if source.pulse=='broadband' else 0.
    if source.time_definition in ('wavelength','frequency'):
        low=C0/(source.wavelength_stop*1e-6);high=C0/(source.wavelength_start*1e-6)
        frequency=(low+high)/2;span=high-low
        length=(2 if source.optimize_for_short_pulse else 8)/(high+.01*low)
        sigma=length/(2*math.sqrt(math.log(2)));offset=TAIL_OUTER*sigma
        chirped=span>math.sqrt(math.log(2))/(math.pi*sigma)
    elif source.time_definition=='standard':
        sigma=source.pulse_length/(2*math.sqrt(math.log(2)));offset=source.pulse_offset
        chirped=source.pulse=='broadband'
    else:
        sigma=source.pulse_cycles/frequency;offset=4*sigma;chirped=False
    return PulseParameters(frequency,sigma,offset,span,chirped)


def gaussian_envelope(u, smooth=False):
    envelope=np.exp(-.5*u*u)
    if not smooth:return envelope
    # C1 Hermite taper from Gaussian amplitude 1e-4 to zero over 10% extra time.
    h=TAIL_OUTER-TAIL_INNER;distance=np.maximum(TAIL_OUTER-np.abs(u),0)
    alpha=(3e-4-TAIL_INNER*1e-4*h)/(h*h)
    beta=(TAIL_INNER*1e-4*h-2e-4)/(h*h*h)
    return np.where(np.abs(u)<=TAIL_INNER,envelope,alpha*distance**2+beta*distance**3)


def source_time_signal(source,times):
    times=np.asarray(times,dtype=np.float64)
    if source.pulse=='sampled':
        signal=source.signal
        amplitude=np.interp(times,signal.time_s,signal.amplitude,left=0,right=0)
        phase=np.interp(times,signal.time_s,signal.phase_rad)+math.radians(source.phase)
        return source.amplitude*amplitude*np.sin(phase)
    envelope,carrier,_=_carrier(source,times)
    return source.amplitude*envelope*np.sin(carrier)


def source_quadrature_signal(source,times):
    """Quadrature partner q(t) of source_time_signal s(t) for Gaussian beams.

    s(t)+i*q(t) carries only the exp(-i*omega*t) (positive physical frequency)
    carrier, so with the exp(+i*k*s) spatial phasor the complex drive is one
    travelling wave exp(i*(k*s-omega*t)) instead of a real carrier whose
    exp(+i*omega*t) half would launch the mirrored tilt.
    """
    if source.pulse=='sampled':
        raise ValueError('Sampled waveforms have no declared carrier sign for a Gaussian beam.')
    times=np.asarray(times,dtype=np.float64)
    envelope,carrier,sign=_carrier(source,times)
    return sign*source.amplitude*envelope*np.cos(carrier)


def _carrier(source,times):
    """Envelope, total sine argument, and the sign of d(argument)/dt."""
    p=pulse_parameters(source);x=times-p.offset_s;u=x/p.sigma_s
    omega=2*math.pi*p.frequency_hz;phase=math.radians(source.phase)
    sign=1
    if source.pulse=='continuous':
        envelope=1-np.exp(-(times/p.sigma_s)**2);carrier=omega*times
    else:
        envelope=gaussian_envelope(u,source.eliminate_discontinuities)
        carrier=omega*times if source.time_definition=='cycles' else -omega*x
        sign=1 if source.time_definition=='cycles' else -1
        if p.chirped:
            # Frequency sweeps sinusoidally between its limits over x=±2*sigma.
            inside=p.frequency_span_hz*p.sigma_s*(4*np.cos(np.pi*u/4)-2*np.pi)
            outside=-np.pi*p.frequency_span_hz*np.abs(x)
            carrier=carrier+np.where(np.abs(u)<=2,inside,outside)
    return envelope,carrier+phase,sign
