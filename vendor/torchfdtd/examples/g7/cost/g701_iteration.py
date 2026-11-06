"""Stage-timed G7-01 design iteration for the G7-05 cost study.

The entry point imports the already-installed torchfdtd wheel before adding
the examples checkout to sys.path. The reduced CPU fixture is for development
only; the declared cost run uses the G7-01 case and CUDA.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from torchfdtd import (AdjointExecutionPolicy, AdjointOptions,
                       DifferentiablePlaneSimulation, StreamedAdjointOptions)
from torchfdtd.design_parameterization import DensityParameterization

from examples.g7.cost.timing import StageTimes
from examples.g7.metagrating import workflow as w


def execution_policy(settings, storage):
    """The two declared storage choices with the same CPU input contract."""
    device = settings.backend
    host_budget = 8 * 1024**3
    if storage == 'resident':
        return AdjointExecutionPolicy(
            resident=AdjointOptions(checkpoints=settings.checkpoints),
            device=device, host_budget_bytes=host_budget)
    if storage == 'host':
        slab_width = 32 if settings.small else 200
        temporal_depth = 4 if settings.small else 48
        options = StreamedAdjointOptions(
            device=device, slab_width=slab_width, temporal_depth=temporal_depth,
            checkpoints=min(settings.checkpoints, 16), gpu_budget_bytes=2 * 1024**3,
            host_budget_bytes=host_budget, state_storage='host',
            tile_transfers='async' if device == 'cuda' else 'sync')
        return AdjointExecutionPolicy(
            streamed=options, device=device, host_budget_bytes=host_budget)
    raise ValueError('Storage must be resident or host.')


class TimedTransmissionObjective(w.TransmissionObjective):
    """G7-01 objective with synchronized solver and monitor stage boundaries."""

    def __init__(self, geometry, settings, timer, *, storage, frequencies):
        self.g = geometry
        self.timer = timer
        self.storage = storage
        self.frequency_mode = frequencies
        # Both policies receive the same CPU design and return CPU observables.
        # Solver transfer costs then stay inside the timed forward/backward stages.
        self.device = torch.device('cpu')
        self.wavelength = np.asarray(w.DESIGN_WAVELENGTHS_UM)
        self.frequency = w.C0 / (self.wavelength * 1e-6)
        project = w.build_project(geometry, settings.design_mesh_um, 'TE', 'normal',
                                  settings, settings.physical_time_fs,
                                  monitors=('transmission',))
        self.region = project.region
        self.policy = execution_policy(settings, storage)
        self.model = self.policy.simulation(project)
        reference_project = w.build_project(geometry, settings.design_mesh_um, 'TE',
                                             'normal', settings, settings.physical_time_fs,
                                             monitors=('reflection',))
        reference = DifferentiablePlaneSimulation(reference_project, AdjointOptions(checkpoints=0))
        reference_device = torch.device(settings.backend)
        with torch.no_grad():
            bare = w.layer_epsilon(torch.zeros(w.pixel_count(geometry), device=reference_device),
                                   self.region, geometry)
            planes = reference(bare, self.frequency)
            self.incident = tuple(value.to(self.device) for value in
                                  w.incident(planes, self.wavelength, geometry, 0., 'TE'))

    def efficiencies(self, density):
        with self.timer.stage('T_geometry'):
            epsilon = w.layer_epsilon(density.to(self.device), self.region, self.g)
        if self.frequency_mode == 'broadband':
            with self.timer.stage('T_forward'):
                plane = self.model(epsilon, frequency_hz=self.frequency)['transmission']
            with self.timer.stage('T_monitor'):
                return w.transmitted_efficiency(plane, *self.incident, self.wavelength,
                                                self.g, 0., 'TE', w.ORDERS)
        if self.frequency_mode != 'sequential':
            raise ValueError('Frequencies must be broadband or sequential.')
        rows = []
        for index in range(len(self.frequency)):
            with self.timer.stage('T_forward'):
                plane = self.model(epsilon, frequency_hz=self.frequency[index:index+1])['transmission']
            with self.timer.stage('T_monitor'):
                rows.append(w.transmitted_efficiency(
                    plane, self.incident[0][index:index+1], self.incident[1][index:index+1],
                    self.wavelength[index:index+1], self.g, 0., 'TE', w.ORDERS))
        return torch.cat(rows, dim=0)


class G701Iteration:
    """One seeded G7-01 trajectory, with cold setup in iteration one."""

    def __init__(self, *, seed=1, storage='resident', frequencies='broadband',
                 reduced=False, backend=None):
        self.case, self.geometry, _ = w.declared()
        if reduced:
            self.settings = w.Settings.small_run()
        else:
            selection = w.load_json(w.ROOT/'docs/validation/g7/G7-01/threshold-selection.json')[0]
            self.settings = w.Settings.declared(self.case, selection)
        if backend is not None:
            self.settings = replace(self.settings, backend=backend)
        self.seed = int(seed)
        self.storage = storage
        self.frequencies = frequencies
        self.design = None
        self.optimizer = None
        self.objective = None

    def step(self):
        settings = self.settings
        timer = StageTimes(device=settings.backend)
        if self.design is None:
            with timer.stage('T_geometry'):
                fixture = self.case['fixture']
                initial = .5 * torch.randn((fixture['pixels'], 1),
                                            generator=torch.Generator().manual_seed(self.seed))
                initial = initial.to('cpu')
                options = dict(spacing_um=(fixture['pixel_um'],
                                           fixture['design_layer_um'][1]-fixture['design_layer_um'][0]),
                               initial=initial, mode='logits',
                               filter_radius_um=settings.filter_radius_um,
                               boundary='periodic', beta=settings.betas[0], eta=.5)
                if settings.threshold_shift is None:
                    self.design = DensityParameterization((fixture['pixels'], 1), **options)
                else:
                    self.design = w.RobustDensity((fixture['pixels'], 1),
                                                  threshold_shift=settings.threshold_shift,
                                                  **options)
            with timer.stage('T_setup'):
                self.objective = TimedTransmissionObjective(
                    self.geometry, settings, timer, storage=self.storage,
                    frequencies=self.frequencies)
                self.optimizer = torch.optim.Adam(self.design.parameters(), lr=settings.learning_rate)
        else:
            self.objective.timer = timer
        with timer.stage('T_geometry'):
            self.optimizer.zero_grad(set_to_none=True)
            density = self.design()
        loss, metrics = self.objective(density)
        with timer.stage('T_backward'):
            loss.backward()
        with timer.stage('T_transfer_io'):
            objective_value = float((-loss).detach().cpu())
            gradient = self.design.design.grad.detach().cpu()
            gradient_norm = float(torch.linalg.vector_norm(gradient))
            metric_values = {key: float(value.detach().cpu()) if isinstance(value, torch.Tensor)
                             else float(value) for key, value in metrics.items()}
        with timer.stage('T_optimizer'):
            self.optimizer.step()
        if not np.isfinite(objective_value) or not np.isfinite(gradient_norm):
            raise ValueError('The timed iteration has a nonfinite objective or gradient.')
        return dict(stages=timer.report(), objective=objective_value,
                    gradient_norm=gradient_norm, metrics=metric_values)
