"""Passive isotropic Drude/Lorentz sums, using a coupled trapezoidal ADE.

E uses exp(-i omega t). P is relative polarization and Q = dt*dP/dt.
Trapezoidal integration solves E and the oscillator together at each cell.
The scaled current Q avoids femtosecond-scale overflow in float32.
"""
from __future__ import annotations

import numpy as np
import torch

from .boundaries import absorber_loss, absorber_slabs


def permittivity(material, frequency_hz, dt=0):
    f = np.asarray(frequency_hz, dtype=float)
    if np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError('Material frequencies must be finite and positive.')
    if not np.isfinite(dt) or dt < 0 or (dt and np.any(f*dt >= .5)):
        raise ValueError('Material timestep must be nonnegative and frequencies below Nyquist.')
    omega = 2*np.pi*f if not dt else 2/dt*np.tan(np.pi*f*dt)
    epsilon = np.full(f.shape, material.instantaneous_epsilon, dtype=complex)
    for w0, strength, gamma in material.oscillators:
        with np.errstate(divide='ignore', invalid='ignore'):
            epsilon += strength/(w0*w0-omega*omega-1j*gamma*omega)
    return epsilon


class MaterialADE:
    """State only at the final, non-overlapping cells owned by one material."""
    def __init__(self, grid, material, indices, components=False):
        self.torch = grid.is_torch
        self.indices = (torch.as_tensor(indices, device=grid.E.device, dtype=torch.long)
                        if self.torch else indices)
        self.components = components
        sample_shape = (len(indices),) if components else (len(indices),3)
        self.oscillators = material.oscillators
        if not self.oscillators:
            raise ValueError('MaterialADE requires at least one oscillator.')
        # Retain the old single-pole array layout and operation order.
        self.multiple = len(self.oscillators) > 1
        shape = (len(self.oscillators), *sample_shape) if self.multiple else sample_shape
        self.P = grid._zeros(shape)
        self.Q = grid._zeros(shape)
        grid.memory_states.extend((self.P, self.Q))
        rates = np.asarray(self.oscillators, dtype=float)
        if self.multiple:
            rates = rates.reshape(len(rates), 3, *([1]*len(sample_shape)))
            w0, strength, gamma = (rates[:, j] for j in range(3))
        else:
            w0, strength, gamma = rates[0]
        dt = grid.time_step
        denominator = 1 + .5*gamma*dt + .25*(w0*dt)**2
        def coefficient(value):
            return grid._coefficient(value) if self.multiple else float(value)
        self.a = coefficient(.5*(w0*dt)**2)
        self.d = coefficient(denominator)
        self.k = coefficient(strength*dt*dt/(4*denominator))
        self.k_sum = self.k.sum(axis=0) if self.multiple else self.k
        self.eps = material.epsilon_inf
        # (eps_new, eps_free, carry) of samples inside the adiabatic absorber, see configure_materials.
        self.absorber = None

    def take(self, field):
        flat = field.reshape(-1) if self.components else field.reshape(-1, 3)
        return flat.index_select(0, self.indices) if self.torch else flat[self.indices]

    def prepare(self, field):
        return self.take(field), (self.Q-self.a*self.P)/self.d

    def correct(self, field, old, response):
        free = self.take(field)
        total_response = response.sum(axis=0) if self.multiple else response
        if self.absorber is None:
            new = (self.eps*free-self.k_sum*old-total_response)/(self.eps+self.k_sum)
        else:
            eps_new, eps_free, carry = self.absorber
            new = (eps_free*free+(carry-self.k_sum)*old-total_response)/(eps_new+self.k_sum)
        delta = response+self.k*(new+old)
        self.P += delta
        self.Q *= -1
        self.Q += 2*delta
        flat = field.reshape(-1) if self.components else field.reshape(-1, 3)
        if self.torch:
            flat.index_copy_(0, self.indices, new)
        else:
            flat[self.indices] = new


def pml_cell_mask(region, shape):
    """True at cells inside a PML layer of `shape` (region.shape or its Yee padding)."""
    mask = np.zeros(shape[:3], dtype=bool)
    for axis in range(3 if region.dimension == '3d' else 2):
        lo, hi = region.pml_layers(axis, 0), region.pml_layers(axis, 1)
        index = [slice(None)]*3
        if lo:
            index[axis] = slice(0, lo); mask[tuple(index)] = True
        if hi:
            index[axis] = slice(shape[axis]-hi, None); mask[tuple(index)] = True
    return mask


def frozen_pml_frequency_hz(project):
    """Centre frequency of the enabled pulsed sources, or None without one."""
    from .waveforms import pulse_parameters
    values = []
    for source in project.sources:
        s = project.resolved_source(source)
        if s.enabled and s.amplitude != 0 and s.pulse != 'sampled':
            values.append(pulse_parameters(s).frequency_hz)
    return float(np.mean(values)) if values else None


def absorber_reference_epsilon(material, reference_hz, dt):
    """Real discrete permittivity the absorber loss is matched to: at the source centre, else static; eps_inf if not positive."""
    eps = float(permittivity(material, reference_hz, dt).real if reference_hz else
                material.instantaneous_epsilon + sum(s/(w0*w0) for w0, s, _ in material.oscillators if w0))
    return eps if eps > 0 else material.epsilon_inf


def configure_materials(grid, project, ownership):
    """Attach one ADE state per dispersive material.

    With region.pml_dispersion == 'frozen' the pole update is not applied inside
    PML layers: those cells keep the real part of the material permittivity at
    the source centre frequency, so the absorber sees a constant dielectric.
    The coupled ADE/CPML update is unstable there for poles whose negative-
    permittivity band lies inside the grid band (surface-plasmon-like modes
    grow after the source has ended; a SiN post filling the outer five cells of
    a 20 nm CPML corner diverges, docs/BOUNDARIES.md).
    With 'absorber' the faces those materials reach carry no stretched
    coordinates (boundaries.absorber_faces) and the ADE runs everywhere. Its
    E conductivity is sigma*eps_ref, eps_ref the real
    permittivity at the reference frequency (eps_inf where that is not
    positive), so that eps_ref*sigma matches the magnetic loss sigma*mu there;
    the trapezoidal update then solves (eps_inf + s eps_ref) E_new + dP =
    (eps_inf - s eps_ref) E_old + courant*curl H with s = sigma*dt/2.
    """
    region = project.region
    frozen = region.pml_dispersion == 'frozen'
    pml = pml_cell_mask(region, ownership.shape).reshape(-1) if frozen else None
    if frozen and ownership.ndim == 4:
        pml = np.repeat(pml, ownership.shape[3])
    absorber = getattr(grid, 'absorber', None)
    reference_hz = frozen_pml_frequency_hz(project) if frozen or absorber is not None else None
    states = []
    grid.absorber_reference = {}
    for i, m in enumerate(project.materials):
        if not m.oscillators or not np.any(ownership == i):
            continue
        owned = ownership.reshape(-1) == i
        if frozen and np.any(owned & pml):
            eps = float(permittivity(m, reference_hz).real if reference_hz else
                        m.instantaneous_epsilon + sum(s/(w0*w0) for w0, s, _ in m.oscillators if w0))
            if eps <= 0:
                # A frozen cell with 1/eps <= 0 has no meaning (docs/BOUNDARIES.md) and diverges.
                at = f'at the reference frequency {reference_hz:.6g} Hz' if reference_hz else 'in the static limit'
                raise ValueError(f'{m.name}: pml_dispersion="frozen" has no meaning for a real permittivity of {eps:.6g} '
                                 f'{at}. Keep the material out of the PML or use pml_dispersion="ade".')
            eps = max(eps, 1e-3)
            inverse = grid.inverse_permittivity
            flat = inverse.reshape(-1) if ownership.ndim == 4 else inverse.reshape(-1, inverse.shape[-1])
            cells = np.flatnonzero(owned & pml)
            if grid.is_torch:
                flat[torch.as_tensor(cells, device=inverse.device, dtype=torch.long)] = 1/eps
            else:
                flat[cells] = 1/eps
            owned = owned & ~pml
            if not np.any(owned):
                continue
        lossy = ()
        if absorber is not None:
            # Samples with absorber loss get a state of their own with its coefficients. The loss is zero off the
            # slabs, so only the slab samples are indexed, box by box.
            view, parts = owned.reshape(ownership.shape), [np.zeros(0, dtype=np.int64)]
            for box in absorber_slabs(region, grid.absorber_faces):
                local = np.nonzero(view[box])
                parts.append(np.ravel_multi_index(tuple(c+b.start for c, b in zip(local, box))+local[3:], ownership.shape))
            candidates = np.sort(np.concatenate(parts))
            loss = absorber_loss(absorber, ownership.shape[:3], 'E', candidates, ownership.ndim == 4)
            keep = loss != 0 if ownership.ndim == 4 else np.any(loss != 0, axis=1)
            lossy, loss = candidates[keep], loss[keep]
            owned[lossy] = False
        indices = np.flatnonzero(owned)
        if len(indices):
            states.append(MaterialADE(grid, m, indices, ownership.ndim == 4))
        if len(lossy):
            state = MaterialADE(grid, m, lossy, ownership.ndim == 4)
            ref = grid.absorber_reference[m.name] = absorber_reference_epsilon(m, reference_hz, grid.time_step)
            state.absorber = tuple(grid._coefficient(v) for v in
                                   (m.epsilon_inf+loss*ref, m.epsilon_inf*(1+loss), loss*(m.epsilon_inf-ref)))
            states.append(state)
    grid.material_states = states
