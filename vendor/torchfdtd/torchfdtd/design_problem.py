"""Minimal high-level inverse-design interface on the existing Torch APIs.

DesignProblem runs objective -> parameterization -> optimizer step ->
history record -> resume from a saved state -> fabrication checks -> binary
export -> re-import -> final evaluation with an independent forward. It owns
no solver: the objective is any callable that maps the parameterization's
density to a scalar Torch loss through PeriodicLayerResponse,
DifferentiablePlaneSimulation, ModeNetwork or another differentiable model,
and the final evaluation is any object that evaluates a density or a native
structure list with a forward the caller builds at a finer mesh.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from .design_parameterization import DensityParameterization
from .fabrication import binary_structures, fabrication_perturbation, measure_feature_sizes

STATE_MARKER = 'torchfdtd-design-problem'


def _float(value):
    value = float(value.detach() if isinstance(value, torch.Tensor) else value)
    if not math.isfinite(value):
        raise RuntimeError('Nonfinite design objective or metric.')
    return value


def _metrics(values):
    return {str(key): _float(value) for key, value in dict(values).items()}


def _optimizer_signature(optimizer):
    return dict(type=type(optimizer).__name__,
                defaults={k: (v if isinstance(v, (int, float, bool, str, type(None))) else repr(v))
                          for k, v in optimizer.defaults.items()},
                groups=[len(group['params']) for group in optimizer.param_groups])


@dataclass(frozen=True)
class Continuation:
    """Beta continuation applied by the problem after every `every` steps."""
    every: int
    factor: float = 2.
    maximum: float | None = None

    def __post_init__(self):
        if isinstance(self.every, bool) or not isinstance(self.every, int) or self.every < 1:
            raise ValueError('Continuation every must be a positive integer number of steps.')
        if not math.isfinite(self.factor) or self.factor < 1:
            raise ValueError('Continuation factor must be finite and at least one.')
        if self.maximum is not None and (not math.isfinite(self.maximum) or self.maximum <= 0):
            raise ValueError('Continuation maximum must be positive and finite.')


class DesignProblem:
    """One density design: parameterization, objective, optimizer, history and resume.

    ``objective(density)`` returns the scalar loss to minimize, or a pair
    ``(loss, metrics)`` whose metrics (floats) are recorded in the history
    without entering the gradient. ``optimizer`` must already hold the
    parameterization's parameters. Every ``step`` records the objective, the
    metrics, the gradient norm and the beta used, then applies the optimizer
    and the continuation. ``run`` saves a checkpoint after every step when a
    path is given and ``resume=True`` continues from it; the history of an
    uninterrupted run and of a resumed run is identical on the CPU because
    the parameterization, optimizer and RNG states are saved after every
    step and restored before the next one.
    """
    def __init__(self, parameterization, objective, optimizer, *, continuation=None, name='design'):
        if not isinstance(parameterization, DensityParameterization):
            raise ValueError('parameterization must be a DensityParameterization.')
        if not callable(objective):
            raise ValueError('objective must be callable.')
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError('optimizer must be a torch.optim.Optimizer over the parameterization parameters.')
        owned = {id(p) for p in parameterization.parameters()}
        held = {id(p) for group in optimizer.param_groups for p in group['params']}
        if held != owned:
            raise ValueError('The optimizer must hold exactly the parameterization parameters.')
        if continuation is not None and not isinstance(continuation, Continuation):
            raise ValueError('continuation must be a Continuation or None.')
        if not isinstance(name, str) or not name:
            raise ValueError('name must be a nonempty string.')
        self.parameterization = parameterization
        self.objective = objective
        self.optimizer = optimizer
        self.continuation = continuation
        self.name = name
        self.history = []
        self.iteration = 0

    def fingerprint(self):
        """What a saved state must share to be resumed: configuration, optimizer and schedule."""
        payload = dict(marker=STATE_MARKER, name=self.name, parameterization=self.parameterization.get_extra_state(),
                       optimizer=_optimizer_signature(self.optimizer),
                       continuation=None if self.continuation is None else vars(self.continuation))
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=repr).encode()).hexdigest()

    @torch.no_grad()
    def density(self, *, hard=False):
        """The current density (continuous, or hard-thresholded at one half)."""
        return self.parameterization(hard=hard).detach().clone()

    def evaluate(self, density):
        """Objective and metrics of a fixed density without gradients."""
        with torch.no_grad():
            value = self.objective(density)
        loss, metrics = value if isinstance(value, tuple) else (value, {})
        return _float(loss), _metrics(metrics)

    def step(self):
        """One objective, gradient, record, optimizer and continuation update."""
        started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        beta = float(self.parameterization.beta)
        density = self.parameterization()
        value = self.objective(density)
        loss, metrics = value if isinstance(value, tuple) else (value, {})
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise ValueError('The objective must return one scalar Torch tensor, optionally with a metrics mapping.')
        objective = _float(loss)
        loss.backward()
        gradients = [p.grad for p in self.parameterization.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError('Nonfinite or missing design gradient.')
        norm = float(torch.sqrt(sum(g.square().sum() for g in gradients)))
        self.optimizer.step()
        if self.parameterization.mode == 'density':
            self.parameterization.project_parameters_()
        self.iteration += 1
        if self.continuation is not None and self.iteration % self.continuation.every == 0:
            self.parameterization.advance_beta(self.continuation.factor, maximum=self.continuation.maximum)
        entry = dict(iteration=self.iteration, objective=objective, metrics=_metrics(metrics), gradient_norm=norm,
                     beta=beta, beta_next=float(self.parameterization.beta), elapsed_s=time.perf_counter()-started)
        self.history.append(entry)
        return entry

    def state(self):
        return dict(marker=STATE_MARKER, fingerprint=self.fingerprint(), iteration=self.iteration,
                    history=[dict(h) for h in self.history], parameterization=self.parameterization.state_dict(),
                    optimizer=self.optimizer.state_dict(), rng=torch.get_rng_state())

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix+'.tmp')
        torch.save(self.state(), temporary)
        temporary.replace(path)
        return path

    def load(self, path):
        # The state holds tensors and plain containers only; a pickled object is refused, not executed.
        try:
            state = torch.load(Path(path), map_location='cpu', weights_only=True)
        except pickle.UnpicklingError as exc:
            raise ValueError('DesignProblem state file holds a pickled object that is not a tensor or a plain container; '
                             'it was refused and nothing in it was executed.') from exc
        if not isinstance(state, dict) or state.get('marker') != STATE_MARKER:
            raise ValueError('Not a DesignProblem state file.')
        if state['fingerprint'] != self.fingerprint():
            raise ValueError('Saved state belongs to a different parameterization, optimizer or continuation. Reconstruct the problem as saved.')
        self.parameterization.load_state_dict(state['parameterization'])
        self.optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng'])
        self.history = [dict(h) for h in state['history']]
        self.iteration = int(state['iteration'])
        return self

    def run(self, iterations, *, checkpoint=None, resume=False):
        """Advance to `iterations` total steps, checkpointing after each when a path is given."""
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
            raise ValueError('iterations must be a nonnegative integer total.')
        if resume:
            if checkpoint is None:
                raise ValueError('resume requires a checkpoint path.')
            if Path(checkpoint).exists():
                self.load(checkpoint)
        while self.iteration < iterations:
            self.step()
            if checkpoint is not None:
                self.save(checkpoint)
        return list(self.history)

    def fabrication(self, *, spacing_um, min_linewidth_um, min_gap_um, perturbation_um, boundary='extend'):
        """Measured feature sizes of the binary design and the objective under erosion and dilation.

        The linewidth and gap come from morphological openings of the
        thresholded design (fabrication.measure_feature_sizes), not from the
        filter radius. The perturbation evaluates the objective with the
        eroded and the dilated binary design, so the recorded changes are the
        forward model's response to the declared over- and under-etch.
        """
        binary = self.density(hard=True)
        sizes = measure_feature_sizes(binary, spacing_um, boundary=boundary)
        violations = sizes.violations(min_linewidth_um=min_linewidth_um, min_gap_um=min_gap_um)
        eroded, dilated, realized = fabrication_perturbation(binary, perturbation_um, spacing_um, boundary=boundary)
        nominal = self.evaluate(binary)
        as_density = lambda array: torch.as_tensor(array, dtype=binary.dtype, device=binary.device)
        under = self.evaluate(as_density(eroded))
        over = self.evaluate(as_density(dilated))
        return dict(feature_sizes=sizes.report(), declared=dict(min_linewidth_um=min_linewidth_um, min_gap_um=min_gap_um,
                    perturbation_um=perturbation_um, boundary=boundary), violations=list(violations),
                    satisfies_declared_constraints=not violations,
                    filter_radius_um=self.parameterization._config['filter_radius_um'],
                    perturbation=dict(realized_um=realized,
                        nominal=dict(objective=nominal[0], metrics=nominal[1]),
                        eroded=dict(objective=under[0], metrics=under[1], objective_change=under[0]-nominal[0],
                                    solid_fraction=float(np.mean(eroded))),
                        dilated=dict(objective=over[0], metrics=over[1], objective_change=over[0]-nominal[0],
                                     solid_fraction=float(np.mean(dilated)))))

    def export(self, directory, *, origin_um, spacing_um, z_min_um, z_max_um, material, gds_layer=(1, 0), cell='DESIGN'):
        """Write the binary design as a native structure list (JSON) and as GDS; return both paths.

        The structure list is the public Structure model serialized with
        model_dump; the GDS is export_gds of the same rectangles with its
        layer-stack sidecar. Both are re-read by reimport().
        """
        from .gds import export_gds
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        binary = self.density(hard=True).cpu().numpy().astype(bool)
        structures = binary_structures(binary, origin_um=origin_um, spacing_um=spacing_um, z_min_um=z_min_um,
                                       z_max_um=z_max_um, material=material, id_prefix=self.name)
        structure_path = directory/f'{self.name}-structures.json'
        structure_path.write_text(json.dumps([s.model_dump(mode='json') for s in structures], indent=1), encoding='utf-8')
        gds_path = directory/f'{self.name}.gds'
        sidecar = export_gds(gds_path, structures, layers={s.id: tuple(gds_layer) for s in structures}, cell=cell)
        sidecar_path = directory/f'{self.name}-layer-stack.json'
        sidecar_path.write_text(json.dumps(sidecar, indent=1), encoding='utf-8')
        binary_path = directory/f'{self.name}-binary.json'
        binary_path.write_text(json.dumps(dict(origin_um=list(origin_um), spacing_um=spacing_um, z_min_um=z_min_um,
                               z_max_um=z_max_um, material=material, pixels=binary.astype(int).tolist())), encoding='utf-8')
        return dict(structures=structures, structure_path=structure_path, gds_path=gds_path, sidecar=sidecar,
                    sidecar_path=sidecar_path, binary_path=binary_path, cell=cell, gds_layer=tuple(gds_layer),
                    z_min_um=z_min_um, z_max_um=z_max_um, material=material)

    @staticmethod
    def reimport(exported):
        """Read the structure list and the GDS back through the public paths.

        Returns the structures parsed from the JSON list and the polygons
        import_gds produced from the GDS file, with the import report.
        """
        from .models import Structure
        from .gds import GDSLayer, import_gds
        listed = tuple(Structure.model_validate(item) for item in json.loads(Path(exported['structure_path']).read_text(encoding='utf-8')))
        layout = import_gds(exported['gds_path'], cell=exported['cell'],
                            layers=[GDSLayer(layer=exported['gds_layer'][0], datatype=exported['gds_layer'][1],
                                             z_min=exported['z_min_um'], z_max=exported['z_max_um'], material=exported['material'])])
        return dict(structures=listed, gds_structures=layout.structures, gds_report=layout.report)

    def final_evaluation(self, evaluator, exported):
        """Independent evaluation of the smooth, thresholded, re-imported and GDS designs.

        ``evaluator.evaluate_density(density)`` and
        ``evaluator.evaluate_structures(structures)`` must return metric
        mappings with the same keys, computed by a forward the caller built
        independently of the objective (a finer mesh). The record lists every
        stage and the metric differences between consecutive stages: mesh
        refinement (fine smooth minus the last coarse objective evaluation),
        thresholding (binary minus smooth, both by the density transfer),
        smoothing (staircase structures minus the volume-averaged binary
        density), and the GDS polygon path (GDS minus structures). A negative
        difference of a metric to be maximized is a loss. An optional
        ``evaluator.evaluate_holdout(structures)`` is called once on the GDS
        polygons for metrics the objective never saw (another wavelength or
        angle) and recorded under ``holdout``.
        """
        for method in ('evaluate_density', 'evaluate_structures'):
            if not callable(getattr(evaluator, method, None)):
                raise ValueError(f'evaluator must provide {method}.')
        smooth = self.density()
        binary = self.density(hard=True)
        coarse_loss, coarse_metrics = self.evaluate(smooth)
        stages = {}
        stages['coarse_smooth'] = dict(coarse_metrics)
        stages['fine_smooth'] = _metrics(evaluator.evaluate_density(smooth))
        stages['fine_binary'] = _metrics(evaluator.evaluate_density(binary))
        reimported = self.reimport(exported)
        stages['fine_structures'] = _metrics(evaluator.evaluate_structures(reimported['structures']))
        stages['fine_gds'] = _metrics(evaluator.evaluate_structures(reimported['gds_structures']))
        keys = list(stages['fine_gds'])
        for name, values in stages.items():
            if set(values) != set(keys):
                raise ValueError(f'Stage {name} reports metrics {sorted(values)} but the evaluator reports {sorted(keys)}.')

        def difference(after, before):
            return {key: stages[after][key]-stages[before][key] for key in keys}
        differences = dict(mesh_refinement=difference('fine_smooth', 'coarse_smooth'),
                           thresholding=difference('fine_binary', 'fine_smooth'),
                           smoothing=difference('fine_structures', 'fine_binary'),
                           gds=difference('fine_gds', 'fine_structures'),
                           total=difference('fine_gds', 'coarse_smooth'))
        holdout = getattr(evaluator, 'evaluate_holdout', None)
        holdout = None if holdout is None else _metrics(holdout(reimported['gds_structures']))
        return dict(coarse_objective=coarse_loss, stages=stages, differences=differences, holdout=holdout,
                    binary_solid_fraction=float(binary.mean()),
                    export=dict(structure_count=len(exported['structures']), reimported_structure_count=len(reimported['structures']),
                                gds_structure_count=len(reimported['gds_structures']), gds_sha256=exported['sidecar']['source_sha256'],
                                gds_maximum_rounding_um=exported['sidecar']['maximum_rounding_um'],
                                gds_native_vertex_count=reimported['gds_report']['native_vertex_count']),
                    conventions=dict(difference='after minus before; a negative difference of a maximized metric is a loss',
                                     fine_density='the evaluator transfers the same pixel density at its own mesh',
                                     fine_structures='inclusive staircase voxelization of the exported rectangles',
                                     fine_gds='polygons read back by import_gds from the exported file'))
