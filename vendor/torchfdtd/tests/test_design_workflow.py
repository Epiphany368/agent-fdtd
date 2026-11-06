"""G6-05 and G6-06: the DesignProblem interface, resume determinism, fabrication
checks and binary export on the two runnable examples, CPU only. The judged
three-start runs of G6-07 live in tests/test_design_reimport.py."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from torchfdtd.design_parameterization import DensityParameterization
from torchfdtd.design_problem import Continuation, DesignProblem
from torchfdtd.fabrication import (FeatureSizes, binary_structures, fabrication_perturbation, measure_feature_sizes,
                                   morphological_open, square_offsets)
from examples import design_metagrating, design_mode_coupler

torch.set_num_threads(2)
FAST_METAGRATING = dict(mesh_um=.05, steps=300)
FAST_COUPLER = dict(mesh_um=.2, steps=200)


def strip_times(history):
    return [{k: v for k, v in entry.items() if k != 'elapsed_s'} for entry in history]


def toy_problem(seed=0, **keywords):
    generator = torch.Generator().manual_seed(seed)
    design = DensityParameterization((6, 4), spacing_um=.1, initial=torch.randn((6, 4), generator=generator),
                                     filter_radius_um=.15, beta=2., **keywords)
    target = torch.linspace(0, 1, 24).reshape(6, 4)

    def objective(density):
        loss = (density-target).square().sum()
        return loss, dict(mean=density.mean())
    optimizer = torch.optim.Adam(design.parameters(), lr=.05)
    return DesignProblem(design, objective, optimizer, continuation=Continuation(every=3, factor=2., maximum=16.))


def test_design_problem_contracts_history_and_continuation():
    problem = toy_problem()
    with pytest.raises(ValueError, match='parameterization'):
        DesignProblem(torch.nn.Linear(1, 1), problem.objective, problem.optimizer)
    with pytest.raises(ValueError, match='exactly the parameterization parameters'):
        DesignProblem(problem.parameterization, problem.objective, torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=.1))
    with pytest.raises(ValueError, match='every'):
        Continuation(every=0)
    history = problem.run(7)
    assert [entry['iteration'] for entry in history] == list(range(1, 8))
    assert [entry['beta'] for entry in history] == [2., 2., 2., 4., 4., 4., 8.]
    assert history[-1]['beta_next'] == 8. and problem.iteration == 7
    assert all('mean' in entry['metrics'] and entry['gradient_norm'] > 0 for entry in history)
    assert history[2]['objective'] < history[0]['objective']   # constant beta over the first three steps
    loss, metrics = problem.evaluate(problem.density(hard=True))
    assert np.isfinite(loss) and set(metrics) == {'mean'}
    with pytest.raises(ValueError, match='scalar'):
        DesignProblem(problem.parameterization, lambda d: d, problem.optimizer).step()


def test_toy_resume_reproduces_the_uninterrupted_history_bitwise(tmp_path):
    straight = toy_problem(seed=3).run(9)
    checkpoint = tmp_path/'toy.pt'
    first = toy_problem(seed=3)
    first.run(4, checkpoint=checkpoint)
    assert checkpoint.exists() and first.iteration == 4
    resumed = toy_problem(seed=3)
    resumed.run(9, checkpoint=checkpoint, resume=True)
    assert strip_times(resumed.history) == strip_times(straight)
    reference = toy_problem(seed=3)
    reference.run(9)
    assert torch.equal(resumed.density(), reference.density())
    other = toy_problem(seed=3, symmetry='mirror_x')
    with pytest.raises(ValueError, match='different parameterization'):
        other.load(checkpoint)
    with pytest.raises(ValueError, match='resume requires'):
        other.run(2, resume=True)


def test_state_file_with_a_pickled_object_is_refused_without_executing_it(tmp_path):
    executed = tmp_path/'executed'

    class Hostile:
        def __reduce__(self):
            return os.makedirs, (str(executed),)

    hostile = tmp_path/'hostile.pt'
    torch.save(dict(marker='torchfdtd-design-problem', rng=Hostile()), hostile)
    with pytest.raises(ValueError, match='pickled object'):
        toy_problem().load(hostile)
    assert not executed.exists()
    torch.load(hostile, weights_only=False)   # control: an unrestricted load runs the payload
    assert executed.is_dir()


def test_metagrating_resume_reproduces_the_uninterrupted_history_bitwise(tmp_path):
    straight, _ = design_metagrating.build_problem(1, **FAST_METAGRATING)
    straight.run(4)
    checkpoint = tmp_path/'metagrating.pt'
    partial, _ = design_metagrating.build_problem(1, **FAST_METAGRATING)
    partial.run(2, checkpoint=checkpoint)
    resumed, _ = design_metagrating.build_problem(1, **FAST_METAGRATING)
    resumed.run(4, checkpoint=checkpoint, resume=True)
    assert strip_times(resumed.history) == strip_times(straight.history)
    assert torch.equal(resumed.parameterization.design.detach(), straight.parameterization.design.detach())
    assert torch.equal(resumed.density(hard=True), straight.density(hard=True))


def test_coupler_resume_reproduces_the_uninterrupted_history_bitwise(tmp_path):
    straight, _ = design_mode_coupler.build_problem(2, **FAST_COUPLER)
    straight.run(3)
    checkpoint = tmp_path/'coupler.pt'
    partial, _ = design_mode_coupler.build_problem(2, **FAST_COUPLER)
    partial.run(1, checkpoint=checkpoint)
    resumed, _ = design_mode_coupler.build_problem(2, **FAST_COUPLER)
    resumed.run(3, checkpoint=checkpoint, resume=True)
    assert strip_times(resumed.history) == strip_times(straight.history)
    assert torch.equal(resumed.parameterization.design.detach(), straight.parameterization.design.detach())


def check_record(record, example, keys):
    assert record['example'] == example and record['iterations'] == 2
    assert len(record['history']) == 2 and all(np.isfinite(h['objective']) for h in record['history'])
    fabrication = record['fabrication']
    sizes = fabrication['feature_sizes']
    assert sizes['linewidth_pixels'] >= 1 and sizes['gap_pixels'] >= 1
    assert set(fabrication['violations']) <= {'min_linewidth', 'min_gap'}
    for kind in ('eroded', 'dilated'):
        assert np.isfinite(fabrication['perturbation'][kind]['objective_change'])
    assert fabrication['perturbation']['eroded']['solid_fraction'] <= fabrication['perturbation']['dilated']['solid_fraction']
    final = record['final_evaluation']
    assert list(final['stages']) == ['coarse_smooth', 'fine_smooth', 'fine_binary', 'fine_structures', 'fine_gds']
    for stage in final['stages'].values():
        assert set(stage) == keys and all(np.isfinite(v) for v in stage.values())
    assert set(final['differences']) == {'mesh_refinement', 'thresholding', 'smoothing', 'gds', 'total'}
    for key in keys:
        total = sum(final['differences'][name][key] for name in ('mesh_refinement', 'thresholding', 'smoothing', 'gds'))
        assert total == pytest.approx(final['differences']['total'][key], abs=1e-9)
    assert final['export']['reimported_structure_count'] == final['export']['structure_count'] == record['export']['structure_count']
    assert final['export']['gds_structure_count'] == final['export']['structure_count']
    assert Path(record['export']['gds']).is_file() and Path(record['export']['structures']).is_file()


def test_metagrating_example_runs_end_to_end_with_fine_reimport(tmp_path):
    record = design_metagrating.main(['--seed', '1', '--iterations', '2', '--steps', '300', '--threads', '2',
                                      '--export-dir', str(tmp_path/'export'), '--output', str(tmp_path/'record.json')])
    check_record(record, 'metagrating', {'efficiency', 'holdout_efficiency', 'zero_order', 'minus_one'})
    assert record['final_evaluation']['holdout'] is None
    # Pixel-aligned rectangles survive the 1 nm GDS rounding: the GDS stage equals the structure stage exactly.
    assert record['final_evaluation']['differences']['gds'] == {k: 0. for k in record['final_evaluation']['differences']['gds']}
    saved = json.loads((tmp_path/'record.json').read_text(encoding='utf-8'))
    assert saved['binary'] == record['binary'] and len(saved['binary']) == design_metagrating.PIXELS


def test_coupler_example_runs_end_to_end_with_fine_reimport_and_holdout(tmp_path):
    record = design_mode_coupler.main(['--seed', '1', '--iterations', '2', '--steps', '200', '--threads', '2',
                                       '--export-dir', str(tmp_path/'export'), '--output', str(tmp_path/'record.json')])
    check_record(record, 'coupler', {'transmission', 'reflection', 'reverse_transmission'})
    holdout = record['final_evaluation']['holdout']
    assert set(holdout) == {'holdout_transmission', 'holdout_reflection', 'holdout_reverse_transmission'}
    assert all(0 <= v <= 1.05 for v in holdout.values())
    assert record['final_evaluation']['differences']['gds'] == {k: 0. for k in record['final_evaluation']['differences']['gds']}
    # The design box respects the declared 180-degree rotation symmetry after thresholding.
    binary = np.array(record['binary'])
    assert np.array_equal(binary, binary[::-1, ::-1])


def test_measured_gap_violates_although_the_filter_radius_is_satisfied():
    # Two solid blocks separated by one 100 nm pixel, through a 150 nm filter
    # radius and a hard projection: the filter radius does not bound the gap.
    logits = torch.full((12, 12), -8.)
    logits[1:11, 1:5] = 8.
    logits[1:11, 6:11] = 8.
    design = DensityParameterization((12, 12), spacing_um=.1, initial=logits, filter_radius_um=.15, boundary='truncate', beta=64.)
    weights = torch.linspace(-1, 1, 144).reshape(12, 12)
    optimizer = torch.optim.Adam(design.parameters(), lr=.05)
    problem = DesignProblem(design, lambda density: (density*weights).sum(), optimizer)
    report = problem.fabrication(spacing_um=.1, min_linewidth_um=.15, min_gap_um=.15, perturbation_um=.1)
    assert report['filter_radius_um'] == .15
    assert report['feature_sizes']['gap_pixels'] == 1 and report['feature_sizes']['min_gap_um'] == pytest.approx(.1)
    assert report['feature_sizes']['linewidth_pixels'] == 4
    assert report['violations'] == ['min_gap'] and not report['satisfies_declared_constraints']
    perturbation = report['perturbation']
    assert perturbation['realized_um'] == pytest.approx(.1)
    assert perturbation['eroded']['objective_change'] != 0 and perturbation['dilated']['objective_change'] != 0
    assert perturbation['eroded']['solid_fraction'] < perturbation['dilated']['solid_fraction']
    binary = problem.density(hard=True).numpy().astype(bool)
    eroded, dilated, _ = fabrication_perturbation(binary, .1, .1)
    assert dilated[:, 5].all() and not eroded[1, :].any()


def test_feature_size_measurement_is_exact_on_runs_and_rectangles():
    line = np.zeros((30, 1), dtype=bool)
    line[2:6] = line[7:8] = line[12:20] = True
    sizes = measure_feature_sizes(line, .05, boundary=('periodic', 'extend'))
    assert isinstance(sizes, FeatureSizes)
    assert (sizes.linewidth_pixels, sizes.gap_pixels) == (1, 1) and sizes.saturated == ()
    line[7:8] = False
    sizes = measure_feature_sizes(line, .05, boundary=('periodic', 'extend'))
    assert (sizes.linewidth_pixels, sizes.gap_pixels) == (4, 6)   # gaps of 6 and, wrapped, 12 pixels
    blocks = np.zeros((12, 12), dtype=bool)
    blocks[2:7, 2:7] = blocks[8:12, 2:7] = True
    sizes = measure_feature_sizes(blocks, .1)
    # The lower block touches the array edge: 'extend' continues it outward (5), 'periodic' bounds it by row 0 (4).
    assert (sizes.linewidth_pixels, sizes.gap_pixels) == (5, 1)
    assert measure_feature_sizes(blocks, .1, boundary='periodic').linewidth_pixels == 4
    assert sizes.violations(min_linewidth_um=.3, min_gap_um=.2) == ('min_gap',)
    assert sizes.violations(min_linewidth_um=.6) == ('min_linewidth',)
    uniform = measure_feature_sizes(np.ones((5, 5), dtype=bool), .1)
    assert uniform.saturated == ('min_linewidth', 'min_gap')
    assert len(square_offsets(1)) == 1 and len(square_offsets(2)) == 4 and len(square_offsets(3)) == 9
    assert np.array_equal(morphological_open(blocks, 5), blocks) and not np.array_equal(morphological_open(blocks, 6), blocks)
    with pytest.raises(ValueError, match='binary'):
        measure_feature_sizes(np.full((3, 3), .5), .1)
    with pytest.raises(ValueError, match='square pixels'):
        measure_feature_sizes(blocks, (.1, .2))
    with pytest.raises(ValueError, match='whole number'):
        fabrication_perturbation(blocks, .15, .1)


def test_binary_structures_tile_solid_pixels_exactly():
    blocks = np.zeros((6, 5), dtype=bool)
    blocks[0:3, 1:3] = blocks[4:6, 1:3] = blocks[2:4, 4] = True
    structures = binary_structures(blocks, origin_um=(-.6, -.5), spacing_um=.2, z_min_um=-.1, z_max_um=.1, material='m')
    assert len(structures) == 3 and all(s.kind == 'rectangle' and s.material == 'm' for s in structures)
    covered = np.zeros_like(blocks)
    for s in structures:
        x0 = round((s.center[0]-s.size[0]/2+.6)/.2)
        x1 = round((s.center[0]+s.size[0]/2+.6)/.2)
        y0 = round((s.center[1]-s.size[1]/2+.5)/.2)
        y1 = round((s.center[1]+s.size[1]/2+.5)/.2)
        assert not covered[x0:x1, y0:y1].any()
        covered[x0:x1, y0:y1] = True
        assert s.size[2] == pytest.approx(.2) and s.center[2] == 0
    assert np.array_equal(covered, blocks)
    assert binary_structures(np.zeros((2, 2), dtype=bool), origin_um=(0, 0), spacing_um=.1, z_min_um=0, z_max_um=1, material='m') == ()
