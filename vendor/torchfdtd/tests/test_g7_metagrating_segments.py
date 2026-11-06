"""Mechanics of the proposed three-ridge G7-01 design parameterization."""

import pytest
import torch

from examples.g7.metagrating.constrained_segments import (
    MIN_RUN, PIXELS, ThreeRidgeDensity, logits_from_runs,
)
from torchfdtd.design_problem import DesignProblem


def periodic_runs(bits):
    values = [int(value) for value in bits]
    transitions = [i for i in range(PIXELS) if values[i] != values[(i-1) % PIXELS]]
    return [(transitions[(j+1) % len(transitions)]-start) % PIXELS
            for j, start in enumerate(transitions)]


@pytest.mark.parametrize('runs', [(3, 21, 5, 7, 13, 51), (4, 23, 11, 3, 3, 56)])
def test_hard_design_keeps_six_fabricable_periodic_runs(runs):
    design = ThreeRidgeDensity(initial=logits_from_runs(runs, dtype=torch.float64))
    assert design.integer_runs() == runs
    hard = design(hard=True).reshape(-1)
    assert set(hard.tolist()) == {0., 1.}
    assert hard.sum() == runs[0]+runs[2]+runs[4]
    assert len(periodic_runs(hard)) == 6
    assert min(periodic_runs(hard)) >= MIN_RUN


def test_smooth_density_gradient_and_design_problem_step():
    design = ThreeRidgeDensity(initial=logits_from_runs((3, 21, 5, 7, 13, 51), dtype=torch.float64))
    weights = torch.linspace(.2, 1.1, PIXELS, dtype=torch.float64).reshape(-1, 1)

    def value():
        return (design()*weights).sum()

    value().backward()
    analytic = float(design.design.grad[1, 0])
    baseline = design.design.detach().clone()
    step = 1e-5
    with torch.no_grad():
        design.design[1, 0] += step
        plus = float(value())
    with torch.no_grad():
        design.design[1, 0] -= 2*step
        minus = float(value())
    with torch.no_grad():
        design.design.copy_(baseline)
    assert analytic == pytest.approx((plus-minus)/(2*step), rel=1e-4, abs=1e-5)

    problem = DesignProblem(design, lambda density: -(density*weights).sum(),
                            torch.optim.Adam(design.parameters(), lr=.01))
    before = float(value().detach())
    problem.step()
    assert float(value().detach()) > before
    assert min(design.integer_runs()) >= MIN_RUN


def test_invalid_runs_are_refused():
    with pytest.raises(ValueError, match='Six integer runs'):
        logits_from_runs((2, 22, 5, 7, 13, 51))
