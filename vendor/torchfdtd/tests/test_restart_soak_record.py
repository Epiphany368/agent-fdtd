"""The recorded restart soak (G5-10) against its pre-declared criteria; nothing long runs here.

`benchmarks/restart_soak.py` writes docs/validation/g5/G5-10_soak_3060.json and
renders docs/RESTART_SOAK.md from it. This module re-judges the record with
the case file's limits, checks that the record was produced under those
limits, and that the rendered document reports the same verdict.
"""
import json
from pathlib import Path

import pytest

from benchmarks.restart_soak import judge, render

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / 'docs' / 'validation' / 'cases' / 'G5-10_restart_soak.json'
RECORD = ROOT / 'docs' / 'validation' / 'g5' / 'G5-10_soak_3060.json'
DOCUMENT = ROOT / 'docs' / 'RESTART_SOAK.md'


@pytest.fixture(scope='module')
def case():
    return json.loads(CASE.read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def record():
    if not RECORD.exists():
        pytest.fail(f'{RECORD.relative_to(ROOT)} is missing; run benchmarks/restart_soak.py first')
    return json.loads(RECORD.read_text(encoding='utf-8'))


def test_record_was_produced_under_the_declared_criteria(case, record):
    assert record['task'] == 'G5-10' and record['criteria'] == case['acceptance']
    assert record['case_path'] == 'docs/validation/cases/G5-10_restart_soak.json'
    assert record['environment']['cuda_available'] in (True, False) and record['environment']['commit']


def test_long_forward_meets_the_declared_bounds(case, record):
    limits = case['acceptance']['long_forward']
    long = record['long_forward']
    mb = 1024**2
    assert long['steps'] >= limits['min_steps'] and long['blocks'] == long['steps'] // long['temporal_depth']
    assert long['journal_records_written']['forward'] == long['blocks'] // long['restart_every_blocks']
    growth = long['memory_growth']['rss_bytes']
    assert growth['samples_after_warmup'] >= 10
    assert growth['slope_bytes_per_unit'] <= limits['max_rss_growth_mb_per_1000_steps']*mb
    for key in ('torch_allocated_bytes', 'torch_reserved_bytes'):
        assert long['memory_growth'][key]['slope_bytes_per_unit'] <= case['acceptance']['max_torch_growth_bytes_per_unit']
    assert long['max_journal_bytes'] <= long['restart_reservation_bytes']
    energy = long['energy']
    assert energy['samples_after_source'] >= 900 and not energy['nonfinite'] and long['signals_finite']
    assert energy['max_relative_norm_after_source'] <= limits['energy_growth_limit']
    assert energy['max_relative_norm_last_half'] <= limits['max_relative_norm_last_half']
    # Monotone decay is not demanded: only the bounds above are judged.


def test_repeated_runs_meet_the_declared_bounds(case, record):
    limits = case['acceptance']['repeated_runs']
    runs = record['repeated_runs']
    assert runs['runs'] >= limits['min_runs'] and len(runs['samples']) == runs['runs']+1
    growth = runs['memory_growth']['rss_bytes']
    assert growth['samples_after_warmup'] == runs['runs']-limits['warmup_runs']+1
    assert growth['slope_bytes_per_unit'] <= limits['max_rss_growth_mb_per_run']*1024**2
    assert runs['gradients_identical']


def test_optimization_meets_the_declared_bounds(case, record):
    limits = case['acceptance']['optimization']
    opt = record['optimization']
    assert opt['updates'] >= limits['min_updates'] and len(opt['history']) == opt['updates']
    growth = opt['memory_growth']['rss_bytes']
    assert growth['samples_after_warmup'] == opt['updates']-limits['warmup_updates']+1
    assert growth['slope_bytes_per_unit'] <= limits['max_rss_growth_mb_per_update']*1024**2
    # beta doubles at updates 25, 50 and 75 (the cap of 16 is not reached within 100 updates).
    assert opt['objectives_finite'] and opt['final_beta'] == 8.


def test_recorded_verdicts_match_a_fresh_judgement_and_the_document(case, record, tmp_path):
    fresh = judge(record, case['acceptance'])
    assert fresh == record['verdicts'] and fresh['all'] is True
    render(record, tmp_path / 'soak.md')
    rendered = (tmp_path / 'soak.md').read_text(encoding='utf-8')
    assert rendered == DOCUMENT.read_text(encoding='utf-8')
    assert '## Verdict: pass' in rendered
