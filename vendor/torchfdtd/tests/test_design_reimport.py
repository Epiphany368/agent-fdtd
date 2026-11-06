"""G6-07: three deterministic starts of each design example, judged at the finer mesh
on the pre-declared final objective and holdout thresholds of docs/validation/cases/G6-07.json.

The judged runs take about half an hour on a CPU and run only with
TORCHFDTD_G6_FULL=1; TORCHFDTD_G6_RECORD=<dir> writes one JSON record per start there
(docs/validation/g6 for the recorded evidence). Without the flag the fast test below
re-judges whatever records exist under docs/validation/g6 and skips when there are none.
Every start is reported; the assertion covers all of them, never a selection.
"""
import json
import os
from pathlib import Path

import pytest

from examples import design_metagrating, design_mode_coupler

ROOT = Path(__file__).resolve().parents[1]
CASE = json.loads((ROOT/'docs/validation/cases/G6-07.json').read_text(encoding='utf-8'))
FULL = os.environ.get('TORCHFDTD_G6_FULL') == '1'
RECORD = os.environ.get('TORCHFDTD_G6_RECORD')
EXAMPLES = {'metagrating': design_metagrating, 'coupler': design_mode_coupler}


def judged_values(record, declared):
    stages = record['final_evaluation']['stages']
    objective = stages['fine_gds'][declared['objective_key']]
    holdout = (stages['fine_gds'] if declared['holdout_stage'] == 'fine_gds_metrics' else record['final_evaluation']['holdout'])
    return dict(seed=record['seed'], iterations=record['iterations'], final_objective=objective,
                holdout=holdout[declared['holdout_key']], coarse_last_objective=record['history'][-1]['metrics'][declared['objective_key']],
                violations=record['fabrication']['violations'],
                differences={name: values[declared['objective_key']] for name, values in record['final_evaluation']['differences'].items()})


def judge(example, records):
    declared = CASE['acceptance'][example]
    rows = [judged_values(record, declared) for record in records]
    assert sorted(row['seed'] for row in rows) == list(CASE['seed']['starts']), 'every declared start must be present'
    assert all(row['iterations'] == declared['iterations'] for row in rows)
    report = json.dumps(dict(example=example, thresholds=dict(final_objective_min=declared['final_objective_min'],
                                                               holdout_min=declared['holdout_min']), starts=rows), indent=1)
    print(report)
    failing = [row['seed'] for row in rows
               if row['final_objective'] < declared['final_objective_min'] or row['holdout'] < declared['holdout_min']]
    assert not failing, f'starts below the declared thresholds: {failing}\n{report}'


@pytest.mark.skipif(not FULL, reason='set TORCHFDTD_G6_FULL=1 for the judged three-start runs (about half an hour on a CPU)')
@pytest.mark.parametrize('example', sorted(EXAMPLES))
def test_three_starts_meet_the_declared_final_and_holdout_thresholds(example, tmp_path):
    declared = CASE['acceptance'][example]
    directory = Path(RECORD) if RECORD else tmp_path
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in CASE['seed']['starts']:
        records.append(EXAMPLES[example].main(['--seed', str(seed), '--iterations', str(declared['iterations']),
                                                '--export-dir', str(directory/'export'/example),
                                                '--output', str(directory/f'{example}-seed{seed}.json')]))
    judge(example, records)


@pytest.mark.parametrize('example', sorted(EXAMPLES))
def test_recorded_starts_meet_the_declared_thresholds(example):
    paths = sorted((ROOT/'docs/validation/g6').glob(f'{example}-seed*.json'))
    if not paths:
        pytest.skip('no recorded starts under docs/validation/g6')
    judge(example, [json.loads(path.read_text(encoding='utf-8')) for path in paths])
