"""Every G3 acceptance case names its oracle classes and layers; docs/ORACLE_BUDGET.md covers every case.

The bookkeeping (oracle_class and tests.layers) lives in the case file itself
when it was declared there, or in a sidecar <case_id>.oracles.json next to a
case whose file is hash-bound to recorded evidence and must not change.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT/'docs/validation/cases'
CASES = sorted(p for p in CASE_DIR.glob('G3-*.json') if not p.name.endswith('.oracles.json'))
LAYER_A = {'shared_discrete_operator', 'finite_difference_same_discretisation', 'independent_implementation_same_scheme'}
LAYER_B = {'analytic_continuum', 'analytic_discrete', 'independent_solver', 'physical_invariant', 'convergence_study',
           'causal_reference'}
EITHER = {'stored_record'}
ALLOWED = LAYER_A | LAYER_B | EITHER
LAYER_B_REQUIRED = {'G3-01', 'G3-02', 'G3-03', 'G3-04', 'G3-05', 'G3-06', 'G3-07', 'G3-08', 'G3-09', 'G3-10', 'G3-11',
                    'G3-12', 'G3-13', 'G3-16'}
LAYER_A_ONLY = {'G3-14', 'G3-15'}
RECORDED = LAYER_B_REQUIRED | LAYER_A_ONLY | {'G3-17'}


def _load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _bookkeeping(path):
    """The case itself when it declares oracle_class, else its sidecar; neither is a failure."""
    case = _load(path)
    if 'oracle_class' in case:
        return case, case
    sidecar = path.with_name(path.stem+'.oracles.json')
    assert sidecar.is_file(), f'{path.name} declares no oracle_class and has no sidecar {sidecar.name}'
    book = _load(sidecar)
    assert book['kind'] == 'oracle_bookkeeping' and book['case_id'] == case['case_id'], sidecar.name
    assert book['case_file'] == f'docs/validation/cases/{path.name}', sidecar.name
    if book['tests']['existing'] != case['tests']['existing']:
        # Only a superseded case may list other tests than its file: the ones of the recorded revision.
        superseding = ROOT/book['superseded_by']
        assert superseding.is_file(), (sidecar.name, book.get('superseded_by'))
        assert book['tests']['existing'] == _bookkeeping(superseding)[1]['tests']['existing'], sidecar.name
        missing = set(case['tests']['existing'])-set(book['tests']['existing'])
        assert missing == set(book['case_tests_not_in_tree']), (sidecar.name, missing)
    return case, book


def test_the_recorded_g3_cases_are_present():
    tasks = {_load(path)['task'] for path in CASES}
    assert RECORDED <= tasks


def test_every_g3_case_names_allowed_oracle_classes_and_layers():
    assert CASES
    for path in CASES:
        case, book = _bookkeeping(path)
        assert case['kind'] == 'acceptance_case' and case['task'].startswith('G3-'), path.name
        assert case['declared_before_run'] is True, path.name
        classes = book['oracle_class']
        assert isinstance(classes, list) and classes and set(classes) <= ALLOWED, (path.name, classes)
        tests = book['tests']
        assert tests['existing'], path.name
        layers = tests['layers']
        assert set(layers) == set(tests['existing']), (path.name, set(layers) ^ set(tests['existing']))
        used = set()
        for test_id, entry in layers.items():
            assert entry['layer'] in ('A', 'B'), (path.name, test_id)
            oracle = entry['oracle_class']
            assert oracle in ALLOWED, (path.name, test_id, oracle)
            if oracle in LAYER_A:
                assert entry['layer'] == 'A', (path.name, test_id, oracle)
            elif oracle in LAYER_B:
                assert entry['layer'] == 'B', (path.name, test_id, oracle)
            assert entry['oracle'], (path.name, test_id)
            used.add(oracle)
        assert used == set(classes), (path.name, used ^ set(classes))
        if case['task'] in LAYER_B_REQUIRED:
            assert any(entry['layer'] == 'B' for entry in layers.values()), path.name
        if case['task'] in LAYER_A_ONLY:
            assert all(entry['layer'] == 'A' for entry in layers.values()), path.name


def test_every_listed_test_exists_in_its_source_file():
    for path in CASES:
        for test_id in _bookkeeping(path)[1]['tests']['existing']:
            source, *classes, name = test_id.split('::')
            file = ROOT/source
            assert file.is_file(), (path.name, test_id)
            text = file.read_text(encoding='utf-8')
            for class_name in classes:
                assert re.search(rf'^class {re.escape(class_name)}\b', text, re.M), (path.name, test_id, class_name)
            assert re.search(rf'^\s*def {re.escape(name)}\(', text, re.M), (path.name, test_id)


def test_oracle_budget_document_covers_every_g3_case_and_class():
    text = (ROOT/'docs/ORACLE_BUDGET.md').read_text(encoding='utf-8')
    for path in CASES:
        case = _load(path)
        assert case['case_id'] in text, case['case_id']
    for name in ALLOWED:
        assert f'`{name}`' in text, name
    for heading in ('Precision floor', 'Time window', 'PML'):
        assert heading in text
