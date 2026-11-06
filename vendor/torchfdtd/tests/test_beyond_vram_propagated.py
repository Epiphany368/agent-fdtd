"""The propagated beyond-VRAM case: declared cases, the rehearsal record, the report and the time model.

Nothing here simulates. The rehearsal record under docs/validation/g5 is judged
against the declared cases exactly as the workstation record will be, and the
wall-time model of the driver is checked against the recorded runs it was
fitted on.
"""
import json
import math
from argparse import Namespace
from pathlib import Path

import pytest

from benchmarks.beyond_vram_propagated import TIME_MODEL, fixture, pillar_mask, plan, time_estimate
from benchmarks.report_beyond_vram_propagated import evaluate, load_cases, render, sanitize

ROOT = Path(__file__).resolve().parents[1]
REHEARSAL = ROOT/'docs/validation/g5/G5-05_rehearsal_3060.json'
JUDGED = ROOT/'docs/validation/beyond_vram_propagated_3060.json'


@pytest.fixture(scope='module')
def cases():
    return load_cases(ROOT)


@pytest.fixture(scope='module')
def rehearsal():
    return sanitize(json.loads(REHEARSAL.read_text(encoding='utf-8')))


def test_cases_are_declared_before_the_run_and_match_the_fixture_generator(cases):
    for task in ('G5-05', 'G5-06'):
        case = cases[task]
        assert case['kind'] == 'acceptance_case' and case['task'] == task and case['declared_before_run'] is True
    family = cases['G5-05']['fixture']
    radius = cases['G5-05']['acceptance']['gradient']['finite_difference']['radius_um']
    for context, key in (('rtx3060', 'judged_run_rtx3060'), ('rtx5880', 'optional_run_rtx5880')):
        declared = family[key]
        spec = fixture(declared['footprint_um'], family['mesh_um'], duration_fs=175.)
        region = spec['project'].region
        assert list(region.shape) == declared['grid'] and region.steps == family['steps']
        assert math.prod(region.shape) == declared['cells'] and len(spec['radii']) == declared['pillars']
        assert list(spec['pillar_layers']) == family['pillar_layers']
        assert 6*math.prod(region.shape)*4 == declared['planner']['eh_bytes']
        of_record = cases['G5-05']['acceptance']['fixture_of_record'][context]
        assert of_record['grid'] == declared['grid'] and of_record['steps'] == family['steps']
        assert of_record['minimum_eh_bytes'] == declared['planner']['eh_bytes']
        # The finite-difference direction covers the same 316 pillars in both instances.
        assert sum(math.hypot(*c) <= radius for c in spec['centres']) == 316
    assert family['judged_run_rtx3060']['execution']['preset'] == 'workstation-3060'
    rehearsal = family['rehearsal']
    small = fixture(rehearsal['footprint_um'], family['mesh_um'], duration_fs=175.)
    assert list(small['project'].region.shape) == rehearsal['grid'] and small['project'].region.steps == rehearsal['steps']
    assert len(small['radii']) == rehearsal['pillars']
    assert sum(math.hypot(*c) <= radius for c in small['centres']) == rehearsal['pillars']
    assert pillar_mask(small, radius).sum() == pillar_mask(small).sum()


def test_rehearsal_record_passes_every_rehearsal_criterion(cases, rehearsal):
    verdict = evaluate(rehearsal, cases, 'rehearsal')
    failed = [row['name'] for row in verdict['criteria'] if not row['passed']]
    assert not failed, failed
    assert verdict['all_passed']
    assert {row['case'] for row in verdict['criteria']} == {'G5-05', 'G5-06'}
    streamed = rehearsal['executions']['streamed']
    assert streamed['fd']['kind'] == 'central' and streamed['fd']['relative_error'] <= .03
    assert rehearsal['comparison']['design_slab_gradient_relative_l2'] <= 1e-4
    assert rehearsal['runs']['streamed_forward']['energy_decayed_fraction'] <= 1e-3
    assert rehearsal['environment']['hardware'].startswith('NVIDIA GeForce RTX 3060')
    assert set(rehearsal['executions']) == {'resident', 'streamed'}


def _judged_evidence():
    if not JUDGED.exists():
        pytest.skip('The RTX 3060 record of the 64 um run has not been produced yet; G5-05 and G5-06 stay NOT_RUN (see docs/DEVELOPMENT_HANDOFF.md).')
    return json.loads(JUDGED.read_text(encoding='utf-8'))


def _judged_rows(cases, case):
    evidence = _judged_evidence()
    assert evidence['context'] == 'rtx3060'
    verdict = evaluate(evidence['record'], cases, 'rtx3060')
    assert verdict['criteria'] == evidence['verdict']['criteria'], 'the stored verdict must be reproducible from the stored record'
    return [row for row in verdict['criteria'] if row['case'] == case]


def test_judged_record_meets_g5_05(cases):
    rows = _judged_rows(cases, 'G5-05')
    failed = [row['name'] for row in rows if not row['passed']]
    assert rows and not failed, failed
    record = _judged_evidence()['record']
    assert record['arguments']['preset'] == 'workstation-3060' and record['executions']['streamed']['fd']['kind'] == 'central'


def test_judged_record_meets_g5_06(cases):
    rows = _judged_rows(cases, 'G5-06')
    failed = [row['name'] for row in rows if not row['passed']]
    assert rows and not failed, failed
    evidence = _judged_evidence()
    assert evidence['record']['environment']['hardware'].startswith('NVIDIA GeForce RTX 3060')
    assert 'system_counters' in evidence['record'], 'the whole-machine counter CSV must be supplied to the report'


def test_rehearsal_record_is_complete_for_the_report_and_carries_no_machine_paths(rehearsal):
    for key in ('fixture', 'grid', 'cells', 'steps', 'eh_bytes', 'runs', 'executions', 'comparison', 'source_sha256',
                'source_amplitude_ratio', 'frequency_resolution_hz', 'elapsed_seconds', 'artifact_sha256'):
        assert key in rehearsal, key
    for phase in ('resident_forward', 'streamed_forward', 'streamed_fd_plus', 'streamed_fd_minus'):
        run = rehearsal['runs'][phase]
        assert run['peak_torch_allocated_bytes'] > 0 and run['memory']['peak_process_rss_bytes'] > 0
    assert 'artifacts_directory' not in rehearsal
    assert rehearsal['arguments']['output'] == '<redacted path>'
    text = json.dumps(rehearsal)
    assert 'D:/' not in text and 'D:\\\\' not in text and 'C:\\\\Users' not in text


def test_report_renders_from_the_record_alone(cases, rehearsal, tmp_path):
    verdict = evaluate(rehearsal, cases, 'rehearsal')
    document = render(dict(record=rehearsal, verdict=verdict))
    assert document.startswith('# Beyond-VRAM propagated case')
    assert 'Verdict against the declared cases: **PASS**' in document
    for heading in ('## Fixture', '## Policy and reservation', '## Results', '## Resident reference (rehearsal)', '## Timings',
                    '## Memory and disk, measured separately', '## Criteria', '## What this does and does not show'):
        assert heading in document
    assert f"{rehearsal['runs']['streamed_forward']['objective']:.6e}" in document
    assert '| plane energy decayed fraction at the final step |' in document
    assert 'not added together' in document
    # The judged contexts refuse a rehearsal record as the judged run.
    for context in ('rtx3060', 'rtx5880'):
        judged = evaluate(rehearsal, cases, context)
        assert not judged['all_passed']
        failed = {row['name'] for row in judged['criteria'] if not row['passed']}
        assert 'grid equals the declared grid' in failed
        assert ('physical VRAM recorded' in failed) == (context == 'rtx5880')
    with pytest.raises(ValueError, match='context'):
        evaluate(rehearsal, cases, 'workstation')


def test_time_model_reproduces_the_records_it_was_fitted_on():
    from torchfdtd import Monitor, Project, Region, Source, StreamedAdjointOptions
    from torchfdtd.streamed_work import estimate_streamed_work
    # The RTX 3060 refit against the two streamed rehearsals of this case.
    for name, policy in (('G5-05_rehearsal_3060', dict(slab_width=32, temporal_depth=16, checkpoints=2)),
                         ('G5-05_rehearsal_3060_streamed_24um', dict(slab_width=32, temporal_depth=16, checkpoints=4))):
        record = json.loads((ROOT/'docs/validation/g5'/f'{name}.json').read_text(encoding='utf-8'))
        spec = fixture(record['fixture']['footprint_um'], record['fixture']['mesh_um'], duration_fs=175.)
        options = StreamedAdjointOptions(device='cuda', local_checkpoints=1, host_budget_bytes=64*1024**3, gpu_budget_bytes=8*1024**3, **policy)
        assert {k: record['executions']['streamed']['options'][k] for k in policy} == policy
        estimate = time_estimate(estimate_streamed_work(spec['project'], options), model='rtx3060')
        assert abs(estimate['forward_seconds']/record['runs']['streamed_forward']['seconds']-1) < .2
        assert abs(estimate['backward_seconds']/record['executions']['streamed']['backward']['seconds']-1) < .2
    for name in ('compact-host-large-5880', 'compact-host-capacity-5880'):
        record = json.loads((ROOT/'docs/validation'/f'{name}.json').read_text(encoding='utf-8'))
        shape, steps = record['grid'], record['steps']
        region = Region(dimension='3d', size=tuple(n*.1 for n in shape), mesh=.1, steps=steps, precision='float32',
                        pml_cells=3, memory_mode='streamed')
        project = Project(region=region, sources=[Source(center=(0, 0, 0), pulse='continuous')],
                          monitors=[Monitor(center=(.1, 0, 0)), Monitor(center=(0, .1, 0), component='Hy')])
        for row in record['records']:
            report = row['report']
            options = StreamedAdjointOptions(device='cuda', slab_width=report['slab_width'], temporal_depth=report['temporal_depth'],
                                             checkpoints=report['checkpoint_capacity'], local_checkpoints=report['local_checkpoint_capacity'],
                                             host_budget_bytes=256*1024**3, gpu_budget_bytes=64*1024**3)
            estimate = time_estimate(estimate_streamed_work(project, options))
            assert abs(estimate['forward_seconds']/report['forward_seconds']-1) < .35
            assert abs(estimate['backward_seconds']/report['backward_seconds']-1) < .35
    assert TIME_MODEL['payload_gb_per_s'] > 0 and TIME_MODEL['gcell_steps_per_s'] > 0


def test_plan_against_the_recorded_workstation_needs_no_device(cases):
    from torchfdtd import StreamedAdjointOptions
    declared = cases['G5-05']['fixture']
    spec = fixture(declared['rehearsal']['footprint_um'], declared['mesh_um'], duration_fs=175.)
    options = StreamedAdjointOptions(device='cuda', slab_width=32, temporal_depth=16, checkpoints=2, local_checkpoints=1,
                                     host_budget_bytes=100*1024**3, gpu_budget_bytes=40*1024**3, state_storage='host')
    args = Namespace(assume='rtx5880', time_model='rtx5880', fd_check='central', journal=None, journal_every=4, banks='host')
    result = plan(args, spec, options)
    assert result['assumed_workstation']['hardware'] == 'NVIDIA RTX 5880 Ada Generation'
    assert result['reservation']['observers'] == 58800 and result['reservation']['host_reservation_bytes'] > 0
    assert result['estimate']['fd_forward_seconds'] == pytest.approx(2*result['estimate']['forward_seconds'])
    slower = plan(Namespace(assume='rtx3060', time_model='rtx3060', fd_check='central', journal=None, journal_every=4, banks='host'), spec, options)
    assert slower['assumed_workstation']['name'] == 'rtx3060' and slower['estimate']['total_seconds'] > result['estimate']['total_seconds']
    assert result['eh_bytes'] == 6*result['cells']*4 and result['disk_writes_estimate_gb'] == 0.
    import torchfdtd.streamed as streamed_module
    from torchfdtd.memory_profile import host_memory
    assert streamed_module.host_memory is host_memory
