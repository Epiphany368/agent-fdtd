"""Per-port diagnostics on the fixed periodic-supercell eigenmodes.

Everything here works on WaveguideMode values returned by
solve_waveguide_modes and on DifferentiablePlaneResult planes. The mode basis
is fixed: the eigenvalue and the mode profile carry no Torch graph, so every
gradient that passes through these functions is a gradient with respect to
the plane fields (and through them the interior material) at a fixed basis.
A port cross-section that carries a Torch graph is rejected with
FixedPortSectionError before any eigenproblem is assembled.
"""
from dataclasses import dataclass
import math
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment

from .mode_ports import (C0, WaveguideMode, mode_power_overlap, solve_waveguide_modes,
                         _amplitudes, _uniform_cell_quadrature)


class FixedPortSectionError(ValueError):
    """The port cross-section is a design variable, which no port solver supports.

    Port eigenmodes are solved once from a fixed real permittivity and are not
    differentiated. A section tensor with requires_grad, or a callable that
    returns one, is refused here rather than silently detached.
    """


class ModeTrackingWarning(UserWarning):
    """A tracked mode's overlap with its predecessor fell below the declared minimum."""


class WeakModeWarning(UserWarning):
    """A port mode's confinement factor fell below the declared threshold."""


def fixed_port_section(section):
    """Return the section with any Torch graph refused, as the port solvers see it."""
    import torch

    def checked(value):
        if isinstance(value, torch.Tensor):
            if value.requires_grad:
                raise FixedPortSectionError('Port cross-sections are fixed. Trainable profile tensors are unsupported.')
            return value.detach().cpu().numpy()
        return value
    if callable(section):
        return lambda u, v: checked(section(u, v))
    return checked(section)


NORMALIZATION = dict(
    power='unit signed reduced power: 0.5 Re(Eu Hv* - Ev Hu*) summed over the transverse Yee cells times the SI cell area equals +1 for a forward mode and -1 for mode.backward()',
    phasor='exp(+i beta w - i omega t) with reduced H; beta in inverse micrometres',
    phase='the largest transverse E component is real and positive at the reference plane',
    degenerate='equal-beta clusters are power orthogonalized and rotated onto the canonical Eu-diagonal basis',
    units='reduced fields, not watts; |amplitude|^2 from separate_directions is the modal reduced power in the plane field units',
)


def port_normalization():
    """The normalization convention of every port mode, as a plain dictionary."""
    return dict(NORMALIZATION)


@dataclass(frozen=True)
class TrackedPortModes:
    """Modes of one port section ordered by continuity across a wavelength band.

    modes[w][m] is track m at wavelengths_um[w]. overlaps[w-1, m] is the
    absolute power overlap between track m at wavelength w and the same track
    at wavelength w-1; minimum_overlap is its smallest entry. solver_index
    records which of the solver's candidates became each track.
    """
    wavelengths_um: tuple
    modes: tuple
    overlaps: np.ndarray
    minimum_overlap: float
    solver_index: np.ndarray
    declared_minimum: float

    @property
    def neff(self):
        return np.array([[mode.neff for mode in row] for row in self.modes])

    @property
    def beta_per_um(self):
        return np.array([[mode.beta_per_um for mode in row] for row in self.modes])

    def report(self):
        return dict(wavelengths_um=list(self.wavelengths_um), neff=self.neff.tolist(),
                    overlaps=self.overlaps.tolist(), minimum_overlap=self.minimum_overlap,
                    declared_minimum=self.declared_minimum, solver_index=self.solver_index.tolist(),
                    normalization=port_normalization())


def track_port_modes(section, wavelengths_um, *, shape, spacing_um, normal='x', num_modes=1,
                     candidates=None, minimum_overlap=.9, precision='float32', origin_um=None,
                     target_neff=None):
    """Solve the port modes at every wavelength and order them by overlap continuity.

    At the first wavelength the solver's first num_modes modes define the
    tracks. At every later wavelength `candidates` modes (default
    num_modes + 2) are solved and assigned to the tracks by the maximum-weight
    matching of the absolute power overlaps with the previous wavelength; the
    assignment is global, so two tracks can never claim one candidate. An
    overlap below minimum_overlap issues ModeTrackingWarning and is reported,
    never repaired. The section is fixed; see FixedPortSectionError.
    """
    section = fixed_port_section(section)
    wavelengths = tuple(float(w) for w in wavelengths_um)
    if len(wavelengths) < 1 or any(not math.isfinite(w) or w <= 0 for w in wavelengths):
        raise ValueError('wavelengths_um must be a nonempty sequence of positive finite values.')
    if isinstance(num_modes, bool) or not isinstance(num_modes, int) or num_modes < 1:
        raise ValueError('num_modes must be a positive integer.')
    candidates = num_modes+2 if candidates is None else candidates
    if isinstance(candidates, bool) or not isinstance(candidates, int) or candidates < num_modes:
        raise ValueError('candidates must be an integer of at least num_modes.')
    if not math.isfinite(minimum_overlap) or not 0 < minimum_overlap <= 1:
        raise ValueError('minimum_overlap must lie in (0, 1].')
    keywords = dict(shape=shape, spacing_um=spacing_um, normal=normal, precision=precision,
                    origin_um=origin_um, target_neff=target_neff)
    rows, overlaps, indices = [], [], []
    previous = None
    for wavelength in wavelengths:
        solved = solve_waveguide_modes(section, wavelength_um=wavelength,
                                       num_modes=num_modes if previous is None else candidates, **keywords)
        if previous is None:
            chosen = list(range(num_modes))
        else:
            weight = np.array([[abs(mode_power_overlap(old, new)) for new in solved] for old in previous])
            track, chosen_index = linear_sum_assignment(-weight)
            chosen = [int(chosen_index[list(track).index(m)]) for m in range(num_modes)]
            overlaps.append([float(weight[m, chosen[m]]) for m in range(num_modes)])
        row = tuple(solved[i] for i in chosen)
        rows.append(row)
        indices.append(chosen)
        previous = row
    overlaps = np.array(overlaps, dtype=float).reshape(max(len(wavelengths)-1, 0), num_modes)
    minimum = float(overlaps.min()) if overlaps.size else 1.
    if minimum < minimum_overlap:
        where = np.unravel_index(int(np.argmin(overlaps)), overlaps.shape)
        warnings.warn(f'Mode track {where[1]} has overlap {minimum:.4f} between wavelengths '
                      f'{wavelengths[where[0]]:.6g} and {wavelengths[where[0]+1]:.6g} um, below the declared '
                      f'minimum {minimum_overlap:.4f}.', ModeTrackingWarning, stacklevel=2)
    return TrackedPortModes(wavelengths, tuple(rows), overlaps, minimum, np.array(indices, dtype=np.int64),
                            float(minimum_overlap))


def degenerate_clusters(modes, *, tolerance=1e-6):
    """Index tuples of equal-beta modes: |beta_i - beta_j| <= tolerance * max(beta_i, beta_j)."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('tolerance must be a finite nonnegative number.')
    betas = [abs(float(mode.beta_per_um)) for mode in modes]
    clusters, start = [], 0
    while start < len(betas):
        end = start+1
        while end < len(betas) and abs(betas[end]-betas[start]) <= tolerance*max(betas[end], betas[start]):
            end += 1
        clusters.append(tuple(range(start, end)))
        start = end
    return tuple(clusters)


def overlap_matrix(modes):
    """Complex power Gram matrix of the modes; the identity for an orthonormal basis."""
    return np.array([[mode_power_overlap(a, b) for b in modes] for a in modes])


def confinement_factor(mode, core):
    """Fraction of the summed squared six-component amplitude inside the declared core.

    core is a boolean array on the transverse cells or a callable core(u_um,
    v_um) evaluated at the cell centres. This counts amplitude, not flux, and
    treats every component as belonging to the cell whose node it shares.
    """
    energy = (np.abs(mode.fields.astype(np.complex128))**2).sum(-1)
    if callable(core):
        axes = [mode.origin_um[a]+(np.arange(mode.fields.shape[a])+.5)*mode.spacing_um[a] for a in range(2)]
        mask = np.asarray(core(*np.meshgrid(*axes, indexing='ij')), dtype=bool)
    else:
        mask = np.asarray(core, dtype=bool)
    if mask.shape != energy.shape:
        raise ValueError('The core mask must match the transverse cell grid of the mode.')
    total = float(energy.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError('The mode carries no finite amplitude.')
    return float(energy[mask].sum()/total)


@dataclass(frozen=True)
class PortDiagnostics:
    neff: np.ndarray
    beta_per_um: np.ndarray
    clusters: tuple
    overlap: np.ndarray
    confinement: np.ndarray
    weak_modes: tuple
    confinement_threshold: float
    normalization: dict

    def report(self):
        return dict(neff=self.neff.tolist(), beta_per_um=self.beta_per_um.tolist(),
                    degenerate_clusters=[list(c) for c in self.clusters],
                    overlap_abs=np.abs(self.overlap).tolist(),
                    overlap_max_off_diagonal=float(np.abs(self.overlap-np.eye(len(self.overlap))).max()),
                    confinement=self.confinement.tolist(), weak_modes=list(self.weak_modes),
                    confinement_threshold=self.confinement_threshold, normalization=dict(self.normalization))


def port_diagnostics(modes, *, core=None, confinement_threshold=.5, degeneracy_tolerance=1e-6):
    """Degenerate clusters, the overlap matrix and the confinement of every mode.

    With a declared core, every mode whose confinement factor lies below
    confinement_threshold is listed under weak_modes and WeakModeWarning is
    issued. Without a core the confinement is reported as NaN and no mode is
    judged weak. Degenerate clusters keep the solver's canonical basis; the
    overlap matrix shows how far it is from power orthonormal.
    """
    modes = tuple(modes)
    if not modes or not all(isinstance(mode, WaveguideMode) for mode in modes):
        raise ValueError('Provide at least one WaveguideMode.')
    if not math.isfinite(confinement_threshold) or not 0 <= confinement_threshold <= 1:
        raise ValueError('confinement_threshold must lie in [0, 1].')
    clusters = degenerate_clusters(modes, tolerance=degeneracy_tolerance)
    gram = overlap_matrix(modes)
    if core is None:
        confinement = np.full(len(modes), np.nan)
        weak = ()
    else:
        confinement = np.array([confinement_factor(mode, core) for mode in modes])
        weak = tuple(int(i) for i in np.flatnonzero(confinement < confinement_threshold))
        for index in weak:
            warnings.warn(f'Port mode {index} (neff {modes[index].neff:.5f}) has confinement '
                          f'{confinement[index]:.4f} below the declared threshold {confinement_threshold:.4f}; '
                          'its periodic images and aperture edges are not negligible.', WeakModeWarning, stacklevel=2)
    return PortDiagnostics(np.array([m.neff for m in modes]), np.array([m.beta_per_um for m in modes]),
                           clusters, gram, confinement, weak, float(confinement_threshold), port_normalization())


def shift_reference_plane(amplitude, mode, distance_um, *, direction='forward'):
    """Move the reference plane of a modal amplitude by distance_um along the port normal.

    With the phasor exp(+i beta w), a forward amplitude referenced at w0 is
    amplitude * exp(i beta d) at w0 + d and a backward amplitude is
    amplitude * exp(-i beta d). The mode's own beta is used; nothing is
    re-solved. amplitude may be a NumPy value or a Torch tensor, and a Torch
    graph on it is preserved.
    """
    if direction not in ('forward', 'backward'):
        raise ValueError('direction must be forward or backward.')
    if not math.isfinite(distance_um):
        raise ValueError('distance_um must be finite.')
    beta = abs(float(mode.beta_per_um))
    phase = complex(np.exp(1j*(1 if direction == 'forward' else -1)*beta*float(distance_um)))
    import torch
    if isinstance(amplitude, torch.Tensor):
        return amplitude*torch.tensor(phase, dtype=amplitude.dtype if amplitude.is_complex() else torch.complex64,
                                      device=amplitude.device)
    return np.asarray(amplitude)*phase


def deembed_s_matrix(s, modes, lengths_um):
    """Strip lengths_um of matched straight guide from every channel of a network S.

    S'[i, j] = S[i, j] exp(-i beta_i l_i) exp(-i beta_j l_j) with beta_k the
    propagation constant of channel k's fixed mode. A positive l_k moves the
    reference plane of channel k inward by l_k, so a straight guide of length
    2L between phase planes at -L and +L becomes the identity after
    de-embedding L at each end. Accepts a Torch tensor (graph preserved) or a
    NumPy array of shape (channels, channels).
    """
    modes, lengths = tuple(modes), tuple(float(v) for v in lengths_um)
    if len(modes) != len(lengths) or any(not math.isfinite(v) for v in lengths):
        raise ValueError('Provide one fixed mode and one finite length per channel.')
    factor = np.array([np.exp(-1j*abs(float(mode.beta_per_um))*length) for mode, length in zip(modes, lengths)])
    import torch
    if isinstance(s, torch.Tensor):
        if s.shape != (len(modes), len(modes)):
            raise ValueError('S must be square over the channels.')
        column = torch.as_tensor(factor, dtype=s.dtype, device=s.device)
        return column[:, None]*s*column[None, :]
    s = np.asarray(s)
    if s.shape != (len(modes), len(modes)):
        raise ValueError('S must be square over the channels.')
    return factor[:, None]*s*factor[None, :]


def separate_directions(plane, mode):
    """Forward and backward amplitudes of one fixed mode on a plane, per frequency.

    Both amplitudes use the Lorentz power overlap with the fixed collocated
    basis (a+ from E x Hm* + Em* x H, a- from their difference), so a pure
    forward field gives a- = 0 exactly. |a|^2 is the modal reduced power in
    the plane field units; divide by a reference to normalize. The plane must
    use the complete uniform midpoint quadrature of the mode supercell at the
    mode's single frequency, as normalized_mode_power requires. Gradients flow
    to plane.fields only; the mode is a fixed basis.
    """
    import torch
    if plane.normal != mode.normal or mode.beta_per_um.real <= 0:
        raise ValueError('Use a forward mode with the plane normal.')
    if plane.frequency_hz.numel() != 1 or not torch.isclose(
            plane.frequency_hz[0], plane.frequency_hz.new_tensor(C0/(mode.wavelength_um*1e-6)), rtol=1e-6, atol=0):
        raise ValueError('A fixed mode requires its single matching frequency.')
    _uniform_cell_quadrature(plane, mode)
    basis = torch.as_tensor(mode.sample_plane(plane.points_um.detach().cpu().numpy()),
                            dtype=plane.fields.dtype, device=plane.fields.device)
    basis_scale = basis.detach().abs().amax()
    weight_scale = plane.weights.detach().abs().amax()
    if not bool(torch.isfinite(basis_scale)) or not bool(basis_scale > 0) or not bool(weight_scale > 0):
        raise ValueError('Mode fields and quadrature weights must be finite and nonzero.')
    forward, backward = _amplitudes(plane.fields, basis/basis_scale, plane.weights/weight_scale, 'xyz'.index(mode.normal))
    return forward/basis_scale, backward/basis_scale
