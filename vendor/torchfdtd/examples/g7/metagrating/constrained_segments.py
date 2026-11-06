"""Development parameterization for three fabricable ridges in one G7-01 period.

This is a candidate design method for a future declared revision. It does not
replace G7-01r3 or turn its failed record into a passing result.
"""
from __future__ import annotations

import math

import torch

from torchfdtd.design_parameterization import DensityParameterization


PIXELS = 100
RUNS = 6  # ridge, gap, ridge, gap, ridge, wrap-around gap
MIN_RUN = 3
FIRST_PIXEL = 2


def logits_from_runs(runs, *, dtype=torch.float32):
    """Seed the six run lengths. Every ridge and gap must span at least three pixels."""
    if len(runs) != RUNS or any(int(value) != value or value < MIN_RUN for value in runs) or sum(runs) != PIXELS:
        raise ValueError('Six integer runs of at least three pixels must fill the 100-pixel period.')
    excess = torch.as_tensor([value - MIN_RUN for value in runs], dtype=dtype)
    return (excess + 0.01).log().reshape(RUNS, 1)


class ThreeRidgeDensity(DensityParameterization):
    """Six positive segment lengths with a fixed 3-pixel minimum.

    The smooth density gives the FDTD objective an ordinary Torch derivative.
    ``hard=True`` quantizes lengths by largest remainder while preserving their
    sum, the three-pixel minimum and all six periodic run lengths. The first
    ridge starts at pixel 2; translating a periodic cell does not change its
    diffraction efficiencies.
    """

    def __init__(self, *, initial, beta=8.):
        initial = torch.as_tensor(initial)
        if initial.shape != (RUNS, 1):
            raise ValueError('initial must be six length logits shaped (6, 1).')
        super().__init__((RUNS, 1), spacing_um=(.02, .5), initial=initial,
                         mode='logits', filter_radius_um=0., boundary='periodic',
                         beta=beta, eta=.5, dtype=initial.dtype)
        self._config.update(parameterization='three-ridge-segments-v1', period_pixels=PIXELS,
                            minimum_run_pixels=MIN_RUN, first_pixel=FIRST_PIXEL)

    def lengths(self):
        return MIN_RUN + (PIXELS - RUNS*MIN_RUN)*torch.softmax(self.design.reshape(-1), dim=0)

    def integer_runs(self):
        with torch.no_grad():
            lengths = self.lengths().detach()
            floors = torch.floor(lengths).to(torch.int64)
            missing = PIXELS - int(floors.sum())
            if not 0 <= missing < RUNS:
                raise RuntimeError('Largest-remainder quantization lost the period.')
            if missing:
                order = torch.argsort(lengths-floors, descending=True, stable=True)
                floors[order[:missing]] += 1
            runs = tuple(int(value) for value in floors.tolist())
        if sum(runs) != PIXELS or min(runs) < MIN_RUN:
            raise RuntimeError('The integer design lost its minimum run length.')
        return runs

    def _hard_density(self):
        result = self.design.new_zeros((PIXELS, 1))
        cursor = FIRST_PIXEL
        for index, width in enumerate(self.integer_runs()):
            if index % 2 == 0:
                result[cursor:cursor+width] = 1
            cursor += width
        if cursor != FIRST_PIXEL+PIXELS or int(result.sum()) < 3*MIN_RUN:
            raise RuntimeError('The three-ridge layout is inconsistent.')
        return result

    def _smooth_density(self):
        runs = self.lengths()
        x = torch.arange(PIXELS, dtype=self.design.dtype, device=self.design.device)
        cursor = self.design.new_tensor(float(FIRST_PIXEL)-.5)
        beta = float(self.beta)
        parts = []
        for index, length in enumerate(runs):
            if index % 2 == 0:
                parts.append(torch.sigmoid(beta*(x-cursor)) - torch.sigmoid(beta*(x-cursor-length)))
            cursor = cursor+length
        return torch.stack(parts).sum(0).clamp(0, 1).reshape(PIXELS, 1)

    def forward(self, *, hard=False, straight_through=False):
        self.gradient_semantics(hard=hard, straight_through=straight_through)
        if not isinstance(hard, bool) or not isinstance(straight_through, bool):
            raise ValueError('hard and straight_through must be booleans.')
        if not bool(torch.isfinite(self.design).all()) or not math.isfinite(float(self.beta)):
            raise ValueError('Nonfinite segment parameters.')
        if not hard:
            return self._smooth_density()
        binary = self._hard_density()
        if straight_through:
            smooth = self._smooth_density()
            return binary + smooth - smooth.detach()
        return binary
