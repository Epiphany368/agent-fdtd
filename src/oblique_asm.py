"""S1: oblique illumination via the **Array Scanning Method** (ASM).

The problem. A finite oblique beam cannot be expressed as a Gaussian inside a periodic cell. A
single Bloch source is one plane wave, and its angle drifts with frequency. The vendor's answer to
this is a kernel change (BFAST). The literature's answer, and the route this leg registers first, is
the Array Scanning Method:

    U_n(r, t) = (d / 2 pi) * integral over the Brillouin zone of
                U_inf(r, t, k) exp(i k n d) dk

evaluated by the midpoint rule over `M` Bloch wavevectors, where `U_inf(r, t, k)` is one ordinary
Bloch-cell run. The cost of the route is therefore **M runs**.

`ASM` here is the Array Scanning Method. It is **not** the angular spectrum method, which the fork
already provides as `torchfdtd/angular_spectrum.py` and which does a different job.

Two competing rules predict `M`:

* **window rule** — the reconstruction is exact for `|x| < M d / 2`, so the beam must fit inside it:
  `M > 2 w0 / d`;
* **spectrum rule** — each Bloch run supplies one plane wave, so the spectrum must be sampled:
  `M > pi w0 / d`.

The model below is transcribed by citation from the reference project `tools/asm_feasibility.py`,
function `reconstruct`, which settled this question there by arithmetic. This project re-derives it
rather than quoting the answer.
"""
import math

import numpy as np

# Orders to sweep. The first `M` that meets the tolerance is reported, so the sweep has to bracket
# both cost rules for every waist this project can state.
DEFAULT_ORDERS = (4, 6, 8, 12, 16, 24, 32, 48, 64, 96)


def brillouin_midpoints(period_um, m):
    """`m` midpoint samples of the Brillouin zone `[-pi/d, pi/d]`."""
    if not isinstance(m, int) or m < 1:
        raise ValueError('the number of Bloch wavevectors must be a positive integer')
    if not (isinstance(period_um, (int, float)) and math.isfinite(period_um) and period_um > 0):
        raise ValueError('the period must be a positive finite number')
    edges = -math.pi / period_um + np.arange(m + 1) * (2.0 * math.pi / (m * period_um))
    return 0.5 * (edges[:-1] + edges[1:])


def cost_rules(w0_um, period_um):
    """The two competing predictions for `M`, and which one is tighter."""
    if not (isinstance(w0_um, (int, float)) and math.isfinite(w0_um) and w0_um > 0):
        raise ValueError('the waist must be a positive finite number')
    if not (isinstance(period_um, (int, float)) and math.isfinite(period_um) and period_um > 0):
        raise ValueError('the period must be a positive finite number')
    window = 2.0 * w0_um / period_um
    spectrum = math.pi * w0_um / period_um
    tighter = 'spectrum' if spectrum >= window else 'window'
    return dict(window_prediction=window, spectrum_prediction=spectrum, tighter_rule=tighter,
                periods_covered_by_beam=window)


def reconstruct(x_um, w0_um, period_um, m, k0_per_um, *, correct_span=True):
    """The ASM midpoint sum, truncated to the propagating branch.

    Continuous form `E(x) = integral A(k) exp(i k x) dk` with `A(k) = exp(-k^2 w0^2 / 4)`, whose
    exact result is `exp(-x^2/w0^2)` up to scale. Evanescent samples carry no power in the
    continuous case, so they are dropped, exactly as a run that only propagates would drop them.
    """
    k = brillouin_midpoints(period_um, m)
    if not correct_span:
        # Negative control C5: sample HALF the Brillouin zone, so the propagating spectrum is
        # truncated. (Note: widening the span is NOT wrong here -- see `span_note`.)
        k = k * 0.5
    amplitude = np.exp(-(k ** 2) * w0_um ** 2 / 4.0)
    amplitude = np.where(np.abs(k) <= k0_per_um, amplitude, 0.0)
    field = (amplitude[None, :] * np.exp(1j * k[None, :] * np.asarray(x_um)[:, None])).sum(axis=1) / m
    return field


def reconstruction_error(x_um, w0_um, period_um, m, k0_per_um, *, correct_span=True):
    """Relative L2 error against the exact Gaussian, measured where the beam matters."""
    x = np.asarray(x_um, dtype=float)
    exact = np.exp(-(x ** 2) / w0_um ** 2)
    exact = exact / np.abs(exact).max()
    field = reconstruct(x, w0_um, period_um, m, k0_per_um, correct_span=correct_span)
    field = field / np.abs(field).max()
    numerator = float(np.sqrt(np.mean(np.abs(field - exact) ** 2)))
    denominator = float(np.sqrt(np.mean(np.abs(exact) ** 2)))
    return numerator / denominator


def first_m_within(w0_um, period_um, k0_per_um, tolerance, orders=DEFAULT_ORDERS, samples=801):
    """The first swept `M` whose reconstruction error is inside `tolerance`, or None."""
    if not (isinstance(tolerance, (int, float)) and tolerance > 0):
        raise ValueError('the tolerance must be a positive number')
    x = np.linspace(-w0_um, w0_um, samples)
    for m in orders:
        if reconstruction_error(x, w0_um, period_um, m, k0_per_um) <= tolerance:
            return m
    return None


def sweep(w0_um, period_um, k0_per_um, orders=DEFAULT_ORDERS, samples=801):
    """Per-order errors for one waist, so the convergence is visible and not summarised away."""
    x = np.linspace(-w0_um, w0_um, samples)
    return [dict(w0_um=w0_um, m=m,
                 rel_l2_error=reconstruction_error(x, w0_um, period_um, m, k0_per_um),
                 **cost_rules(w0_um, period_um)) for m in orders]


# Observed while building the control, and worth keeping: at this wavelength and period the
# propagating branch is LARGER than the first Brillouin zone (`k0 = 4.056 /um` against
# `pi/d = 3.491 /um`). So a sample span widened beyond the zone is not an error -- it captures more
# of the propagating spectrum and the reconstruction gets better. The wrong direction is a span that
# is too NARROW, which truncates the spectrum. The control uses that direction.
span_note = ('k0 > pi/d here, so widening the sampled span beyond the Brillouin zone improves the '
             'reconstruction and narrowing it degrades it. The negative control narrows the span.')


def cell_weights(period_um, m, w0_um, k0_per_um, *, centre=True):
    """The ASM weights in the CELL INDEX, and the position of each cell.

    This is the step the failed campaign got wrong, and it is worth stating plainly:

        a_n = (1/M) * sum_m A(k_m) exp(i k_m n d),  with A(k) = exp(-k^2 w0^2 / 4).

    `a_n` is the **M-point inverse transform** of the sampled spectrum, so `n` is a CIRCULAR index.
    Mapped naively as `x = n d` for `n = 0..M-1`, it covers only the right half of the beam and the
    reconstruction cannot match a centred Gaussian. The physical mapping is the centred one,
    `x = (n - M//2) d`, which puts the source at the middle of the valid window `|x| < M d / 2`.

    A single source cell cannot produce a beam four micrometres wide on its own. The weights do that:
    each cell carries `a_n`, so the sum spans `M` cells. The source is therefore **one cell wide**,
    and the beam width comes from the weights.
    """
    k = brillouin_midpoints(period_um, m)
    amplitude = np.exp(-(k ** 2) * w0_um ** 2 / 4.0)
    amplitude = np.where(np.abs(k) <= k0_per_um, amplitude, 0.0)
    weights = np.fft.ifft(amplitude)
    if centre:
        weights = np.fft.fftshift(weights)
        index = np.arange(m) - m // 2
    else:
        index = np.arange(m)
    return weights, index * period_um


def centred_window_error(period_um, m, w0_um, k0_per_um):
    """Relative error between the ASM cell weights and the Gaussian sampled at cell centres.

    This is the arithmetic that settles the method question: if the weights reproduce the sampled
    Gaussian, the method is right and any failure is in the driver. If they do not, the method needs
    a different formulation.
    """
    weights, x = cell_weights(period_um, m, w0_um, k0_per_um)
    magnitude = np.abs(weights)
    if magnitude.max() <= 0:
        return float('inf')
    measured = magnitude / magnitude.max()
    exact = np.exp(-(x ** 2) / w0_um ** 2)
    exact = exact / exact.max()
    return float(np.sqrt(np.mean((measured - exact) ** 2)) / np.sqrt(np.mean(exact ** 2)))


def window_um(period_um, m):
    """The reconstruction is exact only inside this half-width."""
    return m * period_um / 2.0
