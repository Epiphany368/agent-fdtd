"""Stage-timed G7-03 coupler iteration for the G7-05 cost study."""
from __future__ import annotations

from dataclasses import replace
import math

import torch

from examples.g7.cost.timing import StageTimes
from examples.g7.coupler import workflow as w


class G703Iteration:
    """One seeded PIC-coupler trajectory, with cold model setup in iteration one."""

    def __init__(self, *, seed=1, reduced=False, backend=None):
        self.settings = w.REDUCED if reduced else w.Settings()
        if backend is not None:
            self.settings = replace(self.settings, device=backend)
        self.seed = int(seed)
        self.model = None
        self.problem = None

    def step(self):
        timer = StageTimes(device=self.settings.device)
        if self.model is None:
            with timer.stage('T_setup'):
                self.model = w.Coupler(self.settings.mesh_um, self.settings.steps,
                                       w.DESIGN_UM, self.settings.device)
                self.problem = w.build_problem(self.seed, self.settings,
                                               self.model.objective)
        with timer.stage('T_geometry'):
            self.problem.optimizer.zero_grad(set_to_none=True)
            density = self.problem.parameterization()
            epsilon = self.model.epsilon(density)
        with timer.stage('T_forward'):
            result = self.model.network(epsilon)
        with timer.stage('T_monitor'):
            loss = -result.s[1, 0].abs().square()
            metrics = self.model.metrics(result)
        with timer.stage('T_backward'):
            loss.backward()
        with timer.stage('T_transfer_io'):
            objective = float((-loss).detach().cpu())
            gradient_norm = float(torch.linalg.vector_norm(
                self.problem.parameterization.design.grad.detach().cpu()))
            metric_values = {key: float(value.detach().cpu()) for key, value in metrics.items()}
        with timer.stage('T_optimizer'):
            self.problem.optimizer.step()
            self.problem.iteration += 1
            continuation = self.problem.continuation
            if continuation is not None and self.problem.iteration % continuation.every == 0:
                self.problem.parameterization.advance_beta(
                    continuation.factor, maximum=continuation.maximum)
        if not math.isfinite(objective) or not math.isfinite(gradient_norm):
            raise ValueError('The timed coupler iteration has a nonfinite objective or gradient.')
        return dict(stages=timer.report(), objective=objective,
                    gradient_norm=gradient_norm, metrics=metric_values)
