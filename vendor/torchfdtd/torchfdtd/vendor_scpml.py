"""Opt-in SC-PML algebra, transcribed from the recovered U3 report.

No vendor code is included. Arrays are generated in float64, then the existing
YeeGrid coefficient conversion casts them to field precision. This is algebra
conformance, not a claim about realized vendor arrays or layer adaptation.
"""
from typing import NamedTuple
import numpy as np


class Preset(NamedTuple):
    K: float
    Sigma2: float
    Alpha: float
    m: float
    ma: float
    initial_layers: int
    min_layers: int
    max_layers: int


# Stored/editor ranks 0..3. The property adapter's 1..4 is not used here.
PRESETS = {
    'standard': Preset(2, 1, 0, 3, 1, 8, 8, 64),
    'stabilized': Preset(5, 1, .9, 3, 1, 42, 32, 256),
    'steep angle': Preset(3, 1.5, 0, 3, 1, 12, 12, 64),
    'custom': Preset(2, 1, 0, 3, 1, 8, 8, 64),
}
PROFILE_RANKS = tuple(PRESETS)
FALLBACKS = dict(K=2., Sigma2=1., Alpha=0., m=3., ma=1.)
EPS0 = 8.8541878128e-12  # F/m, vacuum permittivity; only SI diagnostics use it.


def face_factors(grid_vectors_m, dt, dimension, enabled=True):
    """Six faces in x-/x+/y-/y+/z-/z+ order, from actual SI coordinates.

    dimension='2d' adopts the reported d==0 selector interpretation. Missing
    (<=1-coordinate) axes and the inactive z axis retain one. No ratio tolerance
    or near-unity snapping is applied.
    """
    if dimension not in ('2d', '3d') or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Require dimension 2d/3d and a positive finite dt')
    factors = np.ones(6, dtype=np.float64)
    if not enabled:
        return factors
    C = 423970560.0007665 if dimension == '2d' else 519255768.9819587
    for axis, vector in enumerate(grid_vectors_m):
        if axis >= (2 if dimension == '2d' else 3):
            break
        g = np.asarray(vector, dtype=np.float64)
        if len(g) <= 1:
            continue
        if not np.isfinite(g).all() or np.any(np.diff(g) <= 0):
            raise ValueError('SC-PML requires finite ascending grid vectors')
        for side, delta in enumerate((g[1]-g[0], g[-1]-g[-2])):
            factors[2*axis+side] = min(1., dt/(.99*delta/C))
    return factors


def coefficients(N, h, *, preset='standard', parameters=None, factor=1., dt):
    """Return ka,b,c plus corrected normalized and documented SI profiles.

    j increases from the outer face toward the interior. Sampling h=1 and .5
    uses the recovered formula without an upper clamp on r. Explicit N overrides
    the preset initial count; min/max are metadata, not a recovered adaptive law.
    parameters are custom edits using K/Sigma2/Alpha/m/ma; omitted keys fall back
    to standard. Factor scales Sigma2 and Alpha only, before grading.
    """
    if preset not in PRESETS:
        raise ValueError('Unknown vendor SC-PML preset')
    if not isinstance(N, (int, np.integer)) or N < 2:
        raise ValueError('Recovered N=1 denominator is undefined; require N >= 2')
    if h not in (1, .5) or not 0 < factor <= 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Require h=1/.5, 0 < factor <= 1 and positive dt')
    values = dict(zip(FALLBACKS, PRESETS[preset][:5]))
    if parameters is not None:
        if preset != 'custom' or set(parameters)-FALLBACKS.keys():
            raise ValueError('Only custom accepts K/Sigma2/Alpha/m/ma edits')
        values = dict(FALLBACKS, **parameters)
    K, S, A, m, ma = (values[key] for key in FALLBACKS)
    if not np.isfinite([K,S,A,m,ma]).all() or K < 1 or min(S,A,m,ma) < 0:
        raise ValueError('Invalid custom SC-PML parameters')
    j = np.arange(N, dtype=np.float64)
    r = np.maximum(0., (N-j-h)/(N-1))
    k = 1 + (K-1)*r**m
    s = (S*factor)*r**m
    a = (A*factor)*np.maximum(0., 1-r)**ma
    D = 1e-50 + s + a*k
    b = np.exp(-2*D/k)
    c = (b-1)*s/D
    return dict(ka=1/k, b=b, c=c, kappa=k, sigma=s, alpha=a,
                sigma_si=(2*EPS0/dt)*s, alpha_si=(2*EPS0/dt)*a)
