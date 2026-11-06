#!/usr/bin/env python3
"""Leg 14 negative controls, on archived runs, in memory. The run files are never written.

Usage: ./.venv/bin/python src/leg14_controls.py <slab-check-run> <grating-run>

Exit 0 only when EVERY control fails as declared and both unperturbed cases still pass.
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analytic_cases as ac


def load(run):
    run = Path(run)
    return (json.loads((run / 'readout.json').read_text()),
            json.loads((run / 'config.input.json').read_text()))


def status(readout, entry):
    row = ac.evaluate_analytic_cases(readout, dict(analytic_cases=[entry]))[0]
    return row['status'], row['error']


def main(argv):
    slab_readout, slab_config = load(argv[0])
    grating_readout, grating_config = load(argv[1])
    slab = slab_config['analytic_cases'][0]
    grating = grating_config['analytic_cases'][0]
    before = copy.deepcopy((slab_readout, grating_readout))
    controls = [
        ('slab unperturbed', status(slab_readout, slab), 'passed'),
        ('grating unperturbed', status(grating_readout, grating), 'passed'),
        ('slab offset +0.25', status(ac.negative_control_scalar(slab, slab_readout, 0.25), slab),
         'failed'),
        ('slab asymmetric stack (air upper cladding) vs symmetric law',
         status(slab_readout, dict(slab, case_id='slab-tm0-asymmetric-control')), 'failed'),
        ('grating offset +0.25',
         status(ac.negative_control_scalar(grating, grating_readout, 0.25), grating), 'failed'),
        ('grating judged by the s-polarised law',
         status(grating_readout, dict(grating, law='film-ts')), 'failed'),
    ]
    ok = True
    for name, (got, error), want in controls:
        ok &= got == want
        print(f'{name}: status={got} error={error:.6e}; expected {want}')
    assert (slab_readout, grating_readout) == before, 'a control mutated its input'
    print('Source run files were not modified.')
    print('ALL CONTROLS AS DECLARED' if ok else 'A CONTROL DID NOT BEHAVE AS DECLARED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
