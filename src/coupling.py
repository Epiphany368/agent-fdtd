"""S2: the fibre lateral-overlap readout, for cited inputs only.

What problem this solves. The device model in this project is `2d z-invariant`: it computes the
efficiency into a **line** focus, not into a fibre mode. Turning that into a fibre number needs one
more factor, the lateral overlap between the device aperture and the fibre's Gaussian mode. That
factor needs two numbers, and neither is in this repository's record.

**Owner instruction, 2026-10-18:** take them from a published device in the same direction and cite
it. So the inputs are a citation, not an estimate. Every number here carries its source, and a
number that cannot be sourced is not used.

Model, transcribed by citation from the reference project `tools/analyze_coupling_efficiency.py`,
function `lateral_overlap`:

    u = 1 for |z| < W/2 else 0        (a device that is uniform across its width)
    g = exp(-(z/w_lat)^2)             (the fibre mode, field amplitude, 1/e radius w_lat)
    eta_lat = |sum_z u g*|^2 / [(sum_z |u|^2)(sum_z |g|^2)]

**This model's own limitation, stated because it decides how the result may be read.** A laterally
*uniform* aperture delivers power across its whole width, including where the mode is not. So the
overlap peaks at a finite width and **falls** for wider devices. A wider device is therefore not
automatically a better one under this model, and this module never claims otherwise.
"""
import math

import numpy as np

# The cited inputs. `quoted` means the paper states it; `derived` means this project computes it
# from a stated convention, and says so.
CITED_INPUTS = {
    'source': ('R. Marchetti, C. Lacava, A. Khokhar, et al., "High-efficiency grating-couplers: '
               'demonstration of a new design strategy", Sci Rep 7, 16670 (2017), '
               'DOI 10.1038/s41598-017-16505-z'),
    'device_width_um': {
        'value': 12.0, 'kind': 'quoted',
        'quote': ('The grating was 12 um wide (along the x-hat direction of Fig. 1(left)), to '
                  'properly accommodate the Gaussian mode of the fiber, and 14.847 um long.')},
    'fibre_mfd_um': {
        'value': 10.4, 'kind': 'quoted',
        'quote': ('The fiber, which was assumed to have an outer diameter of 125 um and a mode '
                  'field diameter (MFD) of 10.4 um, was tilted by theta_air = 14.5 deg.')},
    'fibre_waist_um': {
        'value': 5.2, 'kind': 'derived',
        'derivation': ('MFD / 2. The overlap model uses g = exp(-(z/w_lat)^2), a field amplitude '
                       'with 1/e radius w_lat, so w_lat = MFD/2. The reference project uses the '
                       'same convention in its own default set of waists.'),
        'from': 'fibre_mfd_um'},
    'platform_note': ('The cited device is a linearly apodized grating coupler on SOI, simulated by '
                      'that paper with 2D-FDTD in a 33.4 x 5.8 um window. That is the same '
                      'modelling limitation this project has, which is why the width must be '
                      'stated separately in both.'),
}


def lateral_overlap(width_um, w_lat_um, n=200001):
    """Overlap of a laterally uniform device of width `width_um` with a Gaussian fibre mode.

    Returns the factor. Raises ValueError for inputs the model cannot describe, so a degenerate
    case becomes a failure rather than a plausible-looking number.
    """
    if not (isinstance(width_um, (int, float)) and isinstance(w_lat_um, (int, float))):
        raise ValueError('width and waist must be numbers')
    if not (math.isfinite(width_um) and math.isfinite(w_lat_um)):
        raise ValueError('width and waist must be finite')
    if width_um <= 0:
        raise ValueError('a zero-width or negative-width device has no aperture to overlap')
    if w_lat_um <= 0:
        raise ValueError('a non-positive fibre waist has no mode to overlap')
    half = max(3.0 * w_lat_um, width_um / 2.0 + 1e-9)
    z = np.linspace(-half, half, n)
    u = (np.abs(z) <= width_um / 2.0).astype(float)
    g = np.exp(-(z / w_lat_um) ** 2)
    numerator = abs(np.sum(u * np.conj(g))) ** 2
    denominator = float(np.sum(np.abs(u) ** 2) * np.sum(np.abs(g) ** 2))
    factor = float(numerator / denominator)
    if not math.isfinite(factor):
        raise ValueError('the overlap is not finite')
    if not -1e-12 <= factor <= 1.0 + 1e-12:
        raise ValueError(f'the overlap {factor!r} is outside [0, 1]: the model does not apply here')
    return min(max(factor, 0.0), 1.0)


def sweep(widths_um, waists_um):
    """Sensitivity table. Never the headline; always labelled as a sweep."""
    return [dict(width_um=float(width), **{f'w_lat_{waist:g}': lateral_overlap(float(width), float(waist))
                                           for waist in waists_um}) for width in widths_um]


def peak_width(w_lat_um, low=1.0, high=60.0, steps=600):
    """Where the overlap peaks in width, for a given waist. The peak is a property of the model."""
    best = (lateral_overlap(low, w_lat_um), low)
    for step in range(1, steps + 1):
        width = low + (high - low) * step / steps
        value = lateral_overlap(width, w_lat_um)
        if value > best[0]:
            best = (value, width)
    return dict(peak_factor=best[0], peak_width_um=best[1], w_lat_um=w_lat_um)
