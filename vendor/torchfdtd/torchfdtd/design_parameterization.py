"""Explicit, serializable 2D density parameterizations for Torch optimization."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def _images(value, symmetry):
    if symmetry == 'none':
        return (value,)
    if symmetry == 'mirror_x':
        return value, value.flip(0)
    if symmetry == 'mirror_y':
        return value, value.flip(1)
    if symmetry == 'mirror_xy':
        return value, value.flip(0), value.flip(1), value.flip((0, 1))
    if symmetry == 'rotate180':
        return value, value.flip((0, 1))
    rotated = tuple(torch.rot90(value, k, (0, 1)) for k in range(4))
    return rotated if symmetry == 'rotate90' else rotated+tuple(v.flip(0) for v in rotated)


class DensityParameterization(nn.Module):
    """Trainable xy density with filtering, symmetry, projection and fixed masks.

    ``initial`` contains raw logits in logits mode or density in density mode.
    Physical lengths use micrometres. A periodic filter uses minimum-image
    distances and counts each pixel once, even when radius exceeds the period.
    A truncate filter excludes outside-domain pixels and renormalizes locally.

    Forward never updates an optimizer, clips its parameters in-place, changes
    beta, or chooses a schedule. Direct-density parameters are clamped only in
    the computation graph, so values outside [0,1] have zero clamp derivative.
    Use project_parameters_ explicitly if a projected optimizer is intended.
    """
    def __init__(self, shape, *, spacing_um, initial=None, mode='logits',
                 filter_radius_um=0., boundary='periodic', symmetry='none',
                 beta=1., eta=.5, fixed_mask=None, fixed_values=0.,
                 dtype=torch.float32, device=None):
        super().__init__()
        shape = tuple(shape)
        if len(shape) != 2 or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in shape):
            raise ValueError('shape must contain two positive integer xy counts.')
        spacing_um = ((float(spacing_um),)*2 if isinstance(spacing_um, (int, float))
                      else tuple(spacing_um))
        if len(spacing_um) != 2 or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in spacing_um):
            raise ValueError('spacing_um must contain two fixed positive finite lengths.')
        spacing_um = tuple(float(v) for v in spacing_um)
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('Use float32 or float64 parameter precision.')
        if mode not in ('logits', 'density') or boundary not in ('periodic', 'truncate'):
            raise ValueError('Use logits/density mode and periodic/truncate boundary.')
        choices = ('none', 'mirror_x', 'mirror_y', 'mirror_xy', 'rotate180', 'rotate90', 'dihedral4')
        if symmetry not in choices:
            raise ValueError('Unknown reflection or rotation symmetry.')
        if symmetry in ('rotate90', 'dihedral4') and (shape[0] != shape[1] or spacing_um[0] != spacing_um[1]):
            raise ValueError('Quarter-turn symmetry requires a square grid with equal physical spacing.')
        if isinstance(filter_radius_um, torch.Tensor) or not math.isfinite(filter_radius_um) or filter_radius_um < 0:
            raise ValueError('filter_radius_um must be a fixed finite nonnegative length.')
        self._validate_projection(beta, eta)
        self._config = dict(version=1, shape=shape, spacing_um=spacing_um, mode=mode,
            filter_radius_um=float(filter_radius_um), boundary=boundary, symmetry=symmetry)
        self.shape, self.mode, self.boundary = shape, mode, boundary
        chosen_device = device if device is not None else initial.device if isinstance(initial, torch.Tensor) else 'cpu'
        initial = (0. if mode == 'logits' else .5) if initial is None else initial
        value = torch.as_tensor(initial, dtype=dtype, device=chosen_device)
        if value.ndim == 0:
            value = value.expand(shape)
        if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
            raise ValueError('initial must be finite, scalar or shaped like the xy grid.')
        if mode == 'density' and bool(((value < 0)|(value > 1)).any()):
            raise ValueError('Initial density must lie in [0,1].')
        self.design = nn.Parameter(value.detach().clone())
        mask = (torch.zeros(shape, dtype=torch.bool, device=self.design.device) if fixed_mask is None
                else torch.as_tensor(fixed_mask, device=self.design.device))
        if mask.dtype != torch.bool or tuple(mask.shape) != shape:
            raise ValueError('fixed_mask must be a boolean xy array.')
        fixed = torch.as_tensor(fixed_values, dtype=dtype, device=self.design.device)
        if fixed.ndim == 0:
            fixed = fixed.expand(shape)
        if tuple(fixed.shape) != shape or not bool(torch.isfinite(fixed).all()) or bool(((fixed < 0)|(fixed > 1)).any()):
            raise ValueError('fixed_values must be a finite scalar or xy array in [0,1].')
        active_fixed = torch.where(mask, fixed, torch.zeros_like(fixed))
        if any(not torch.equal(mask, image) for image in _images(mask, symmetry)) or any(
                not torch.equal(active_fixed, image) for image in _images(active_fixed, symmetry)):
            raise ValueError('Fixed masks and their values must respect the requested symmetry.')
        self.register_buffer('fixed_mask', mask.detach().clone())
        self.register_buffer('fixed_values', fixed.detach().clone())
        self.register_buffer('beta', self.design.new_tensor(float(beta)))
        self.register_buffer('eta', self.design.new_tensor(float(eta)))
        self.register_buffer('continuation_updates', torch.zeros((), dtype=torch.int64, device=self.design.device))
        grid = torch.arange(math.prod(shape), device=self.design.device).reshape(shape)
        representatives = torch.stack(_images(grid, symmetry)).amin(0).reshape(-1)
        _, ids = torch.unique(representatives, sorted=True, return_inverse=True)
        counts = torch.bincount(ids).to(dtype)
        self.register_buffer('_orbit_ids', ids)
        self.register_buffer('_orbit_counts', counts)
        radius = float(filter_radius_um)
        offsets = []
        for n, step in zip(shape, spacing_um):
            reach = min(n-1, math.ceil(radius/step))
            lo, hi = (-min(n//2, reach), min((n-1)//2, reach)) if boundary == 'periodic' else (-reach, reach)
            offsets.append(torch.arange(lo, hi+1, dtype=dtype, device=self.design.device)*step)
        distance = (offsets[0][:, None].square()+offsets[1][None, :].square()).sqrt()
        kernel = (1-distance/radius).clamp_min(0) if radius else torch.ones_like(distance)
        kernel = kernel/kernel.sum()
        self.register_buffer('_kernel', kernel[None, None])
        # Correlation maps index k to offset k-left. Even periodic grids use
        # one of the two equivalent half-period neighbors, never both images.
        left_x = min(shape[0]//2, math.ceil(radius/spacing_um[0])) if boundary == 'periodic' else (kernel.shape[0]-1)//2
        left_y = min(shape[1]//2, math.ceil(radius/spacing_um[1])) if boundary == 'periodic' else (kernel.shape[1]-1)//2
        self._padding = (left_y, kernel.shape[1]-left_y-1, left_x, kernel.shape[0]-left_x-1)
        norm = F.conv2d(F.pad(torch.ones((1, 1, *shape), dtype=dtype, device=self.design.device),
                             self._padding), self._kernel) if boundary == 'truncate' else self.design.new_ones((1, 1, 1, 1))
        self.register_buffer('_normalizer', norm)

    @staticmethod
    def _validate_projection(beta, eta):
        if any(isinstance(v, torch.Tensor) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (beta, eta)):
            raise ValueError('beta and eta must be fixed finite scalars.')
        if beta < 0 or not 0 < eta < 1:
            raise ValueError('beta must be nonnegative and eta strictly inside (0,1).')

    def get_extra_state(self):
        """Protect physical/filter configuration when resuming state_dict."""
        return dict(self._config)

    def set_extra_state(self, state):
        if state != self._config:
            raise ValueError('Parameterization configuration differs. Reconstruct it with the saved configuration.')

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Reject incompatible physical configuration before copying parameters.
        if '_extra_state' in state_dict:
            self.set_extra_state(state_dict['_extra_state'])
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Parent-module loading bypasses a child's public load_state_dict.
        if prefix+'_extra_state' in state_dict:
            self.set_extra_state(state_dict[prefix+'_extra_state'])
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    @torch.no_grad()
    def set_beta(self, value):
        """Explicit continuation update, persisted together with its update count."""
        self._validate_projection(value, float(self.eta))
        self.beta.fill_(float(value))
        self.continuation_updates.add_(1)

    @torch.no_grad()
    def advance_beta(self, factor=2., *, maximum=None):
        if isinstance(factor, torch.Tensor) or not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor < 1:
            raise ValueError('Continuation factor must be a fixed finite scalar at least one.')
        value = float(self.beta)*factor
        if maximum is not None:
            if not isinstance(maximum, (int, float)) or not math.isfinite(maximum) or maximum < float(self.beta):
                raise ValueError('Continuation maximum must be finite and no smaller than current beta.')
            value = min(value, maximum)
        self.set_beta(value)
        return float(self.beta)

    @torch.no_grad()
    def project_parameters_(self):
        """Explicit box projection for direct-density optimization only."""
        if self.mode != 'density':
            raise ValueError('Logits have no box constraint. Projection is only for density mode.')
        self.design.clamp_(0, 1)
        return self

    @staticmethod
    def gradient_semantics(*, hard=False, straight_through=False):
        if straight_through and not hard:
            raise ValueError('straight_through requires hard=True.')
        return ('straight-through surrogate of smooth density' if straight_through else
                'hard threshold, no design derivative' if hard else 'exact derivative of smooth parameterization')

    def forward(self, *, hard=False, straight_through=False):
        """Return a bounded 2D density, with hard/surrogate behavior only on request."""
        self.gradient_semantics(hard=hard, straight_through=straight_through)
        if not isinstance(hard, bool) or not isinstance(straight_through, bool):
            raise ValueError('hard and straight_through must be booleans.')
        if not bool(torch.isfinite(self.design).all()):
            raise ValueError('Design parameters must remain finite.')
        self._validate_projection(float(self.beta), float(self.eta))
        density = self.design.sigmoid() if self.mode == 'logits' else self.design.clamp(0, 1)
        density = torch.where(self.fixed_mask, self.fixed_values, density)
        padded = F.pad(density[None, None], self._padding,
                       mode='circular' if self.boundary == 'periodic' else 'constant')
        filtered = (F.conv2d(padded, self._kernel)/self._normalizer)[0, 0].clamp(0, 1)
        sums = filtered.new_zeros(self._orbit_counts.shape).scatter_add(0, self._orbit_ids, filtered.reshape(-1))
        symmetric = (sums/self._orbit_counts)[self._orbit_ids].reshape(self.shape)
        if float(self.beta) == 0:
            projected = symmetric
        else:
            lower = torch.tanh(self.beta*self.eta)
            projected = ((lower+torch.tanh(self.beta*(symmetric-self.eta)))
                         /(lower+torch.tanh(self.beta*(1-self.eta)))).clamp(0, 1)
        density = torch.where(self.fixed_mask, self.fixed_values, projected)
        if hard:
            values = self.fixed_values[self.fixed_mask]
            if bool(((values != 0)&(values != 1)).any()):
                raise ValueError('Hard output requires binary fixed values.')
            binary = (density >= .5).to(density.dtype)
            density = density+(binary-density).detach() if straight_through else binary
        return density
