"""Whole-domain decay diagnostics and explicit termination decisions.

The volume-weighted E/H and passive oscillator norm is a stopping heuristic,
not a conserved discrete energy or a bound on Fourier-observable error.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from .waveforms import pulse_parameters, TAIL_OUTER


def source_end_time(project):
    """Conservative envelope end, including delayed sources and sampled gaps."""
    cutoff = project.region.run_control.source_tail_amplitude
    ends = [0.]
    for source in project.sources:
        s = project.resolved_source(source)
        if not s.enabled or s.amplitude == 0:
            continue
        if s.pulse == 'continuous':
            return math.inf
        if s.pulse == 'sampled':
            drive_end=s.signal.time_s[-1]
        else:
            pulse=pulse_parameters(s)
            tail=TAIL_OUTER if s.eliminate_discontinuities else math.sqrt(-2*math.log(cutoff))
            drive_end=pulse.offset_s+tail*pulse.sigma_s
        if s.kind=='tfsf':
            from .tfsf import tfsf_plan
            axis,lo,hi,_=tfsf_plan(s,project.region)
            from .mesh import local_uniform_step
            flight=(8+hi[axis]-lo[axis])*local_uniform_step(project.region,axis,lo[axis]-1,hi[axis]+2)*1e-6*project.region.background_index/299792458.
            ends.append(drive_end+flight)
            continue
        if s.injection == 'oneway':
            from .injection import oneway_tables
            e, h = oneway_tables(s, project.region)
            envelope = np.maximum(abs(e),abs(h))
            active = np.flatnonzero(envelope > cutoff*np.max(envelope))
            # Inspect the complete prepared table, including later pulses and
            # incident-line travel/dispersion. Do not use the drive end alone.
            table_end=(int(active[-1])+1.5)*project.region.time_step if len(active) else 0.
            from .injection import oneway_plan
            from .mesh import local_uniform_step
            axis,k,_=oneway_plan(s,project.region)
            flight=8*local_uniform_step(project.region,axis,k-1,k+2)*1e-6*project.region.background_index/299792458.
            # A delayed drive can begin beyond this finite preview horizon.
            # An all-zero table must not make that source appear finished.
            ends.append(max(table_end,drive_end+flight))
            continue
        ends.append(drive_end)
    return max(ends)


class DecayDecision:
    """State independent of the backend. Sources must finish before decay tests."""
    def __init__(self, control, source_end_s, dt):
        self.control, self.dt = control, dt
        self.eligible_time = source_end_s + control.after_source_s
        self.peak = 0.
        self.frozen_peak = None
        self.below = 0
        self.history = []
        self.prev_fluxes = None
        self.last_energy_ratio = 1.0
        self.last_flux_delta = 1.0

    def update(self, step, energy, field_peak):
        c = self.control
        if not math.isfinite(energy) or not math.isfinite(field_peak):
            raise FloatingPointError(f'Non-finite full-domain or auxiliary state at step {step}.')
        if c.divergence_check and c.field_limit is not None and field_peak > c.field_limit:
            raise FloatingPointError(f'Field magnitude limit exceeded at step {step}: {field_peak:.6g}.')
        eligible = step >= c.min_steps and step*self.dt >= self.eligible_time
        self.peak = max(self.peak, energy)
        if eligible and self.frozen_peak is None:
            self.frozen_peak = self.peak
        baseline = self.frozen_peak if self.frozen_peak is not None else self.peak
        ratio = energy/baseline if baseline else 0.
        self.last_energy_ratio = float(ratio)
        if c.divergence_check and eligible and ratio > c.growth_limit:
            raise FloatingPointError(f'Post-source state norm grew by {ratio:.6g} at step {step}.')
        self.below = self.below+1 if eligible and ratio <= c.decay_threshold else 0
        self.history.append(dict(step=step, time_s=step*self.dt, state_norm=energy,
                                 relative_norm=ratio, field_peak=field_peak, source_finished=eligible))
        return c.auto_shutoff and self.below >= c.consecutive_checks

    def update_with_flux(self, step, energy, field_peak=0.0, monitor_fluxes=None):
        """Dual-criterion verified auto-shutoff: energy ratio <= 1e-9 AND flux delta <= 1e-6."""
        c = self.control
        if not math.isfinite(energy) or not math.isfinite(field_peak):
            raise FloatingPointError(f'Non-finite full-domain or auxiliary state at step {step}.')
        if c.divergence_check and c.field_limit is not None and field_peak > c.field_limit:
            raise FloatingPointError(f'Field magnitude limit exceeded at step {step}: {field_peak:.6g}.')

        # Enforce pulse cessation condition: t >= source_end + after_source and step + 1 >= min_steps
        eligible = (step + 1) >= c.min_steps and (step + 1) * self.dt >= self.eligible_time
        self.peak = max(self.peak, energy)
        if eligible and self.frozen_peak is None:
            self.frozen_peak = self.peak
        baseline = self.frozen_peak if self.frozen_peak is not None else self.peak
        ratio = energy / baseline if baseline else 0.
        self.last_energy_ratio = float(ratio)

        if c.divergence_check and eligible and ratio > c.growth_limit:
            raise FloatingPointError(f'Post-source state norm grew by {ratio:.6g} at step {step}.')

        # Monitor flux variation across consecutive inspection blocks
        if monitor_fluxes is not None and len(monitor_fluxes) > 0:
            if self.prev_fluxes is not None and len(self.prev_fluxes) == len(monitor_fluxes):
                deltas = [
                    abs(cur - prev) / max(abs(cur), 1e-12)
                    for cur, prev in zip(monitor_fluxes, self.prev_fluxes)
                ]
                flux_delta = max(deltas)
            else:
                flux_delta = 1.0
            self.prev_fluxes = list(monitor_fluxes)
        else:
            flux_delta = 0.0
        self.last_flux_delta = float(flux_delta)

        # Dual criteria: energy ratio <= 1e-9 AND relative monitor flux variation <= 1e-6
        decay_thresh = min(getattr(c, 'decay_threshold', 1e-9), 1e-9)
        flux_thresh = 1e-6
        passed = eligible and (ratio <= decay_thresh) and (flux_delta <= flux_thresh)
        self.below = self.below + 1 if passed else 0

        self.history.append(dict(
            step=step, time_s=(step + 1) * self.dt, state_norm=energy,
            relative_norm=ratio, flux_delta=flux_delta, field_peak=field_peak, source_finished=eligible
        ))
        consecutive = getattr(c, 'consecutive_checks', 2) or 2
        return bool(c.auto_shutoff and self.below >= consecutive)


class StateDiagnostics:
    def __init__(self, grid, *, fused=True):
        self.grid = grid
        self.torch = grid.is_torch
        r = grid.region
        widths = [np.diff(nodes)/r.reference_step if n>1 else np.ones(1)
                  for nodes,n in zip(r.mesh_nodes,r.shape)]
        volume = widths[0][:,None,None]*widths[1][None,:,None]*widths[2][None,None,:]
        self.volume = grid._coefficient(volume)
        self.material_weights = [state.take(
            self.volume[...,None].expand(*r.shape,3) if self.torch else
            np.broadcast_to(self.volume[...,None],(*r.shape,3))) for state in grid.material_states]
        # Dispersive subpixel interfaces (torchfdtd.subpixel_dispersive): E weighed with the instantaneous
        # inverse permittivity, which includes their D-driven shares, and the energies of their branches.
        self.inverse = getattr(grid, 'energy_inverse_permittivity', grid.inverse_permittivity)
        dispersive = getattr(getattr(grid, 'subpixel', None), 'dispersive', None)
        self.dispersive_terms = dispersive.energy_terms() if dispersive is not None else []
        self.cuda = None
        self.backend = 'torch' if self.torch else 'numpy'
        if fused and self.torch and grid.E.is_cuda:
            try:
                import cupy
            except ImportError:
                pass
            else:
                from .cuda_diagnostics import CudaStateDiagnostics
                self.cuda = CudaStateDiagnostics(self)
                self.backend = 'fused_cuda'

    def real64(self, array):
        if self.torch:
            return array.to(torch.complex128 if array.is_complex() else torch.float64)
        return np.asarray(array, dtype=np.complex128 if np.iscomplexobj(array) else np.float64)

    def square_sum(self, array, weight=1):
        a = self.real64(array)
        power = a.real*a.real + a.imag*a.imag if (a.is_complex() if self.torch else np.iscomplexobj(a)) else a*a
        return (power*weight).sum()

    def measure(self):
        if self.cuda is not None:
            return self.cuda.measure()
        g = self.grid
        energy = 0.
        for j in range(3):
            energy = energy + self.square_sum(g.E[...,j], self.real64(self.volume)/self.real64(self.inverse[...,j]))
            energy = energy + self.square_sum(g.H[...,j], self.volume)
        for state, weight in zip(g.material_states, self.material_weights):
            for j, (w0, strength, _) in enumerate(state.oscillators):
                p = state.P[j] if state.multiple else state.P
                q = state.Q[j] if state.multiple else state.Q
                # Promote before scaling to preserve weak float32 oscillators.
                energy = energy + self.square_sum(self.real64(q)/(g.time_step*math.sqrt(strength)), weight)
                energy = energy + self.square_sum(self.real64(p)*(w0/math.sqrt(strength)), weight)
        for array, weight in self.dispersive_terms:
            energy = energy + self.square_sum(array, weight)
        for state in g.incident_states:
            energy=energy+self.square_sum(self.real64(state.e)*g.region.background_index,state.norm_weight)
            energy=energy+self.square_sum(state.h,state.norm_weight)
        memories=[s['psi'] for segments in g.cpml.values() for s in segments]
        memories.extend(a for state in g.incident_states for a in (state.pe,state.ph))
        if self.torch:
            peak = torch.maximum(self.real64(g.E).abs().amax(), self.real64(g.H).abs().amax())
            finite = [torch.isfinite(a).all() for a in memories]
            valid = torch.stack(finite).all().to(torch.float64) if finite else torch.ones_like(energy)
            values = torch.stack((energy,peak,valid)).detach().cpu().tolist()
        else:
            values = [float(energy), float(max(np.max(abs(self.real64(g.E))),np.max(abs(self.real64(g.H))))),
                      all(np.isfinite(a).all() for a in memories)]
        if not values[2]:
            raise FloatingPointError('Non-finite CPML or incident-line memory detected.')
        return values[:2]
