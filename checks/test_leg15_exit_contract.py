"""LEG 32 -- the leg-15 diagnostics driver must refuse when its own criteria fail.

``src/leg15_closure.py`` copies its pre-declared hypotheses and negative control
into the readout (``:204-205``) and then returns ``0`` unconditionally
(``:209``). Nothing compares the readout with the criteria it carries, so a lost
or vacuous negative control changes nothing. The assertions that were meant to
hold those criteria live in ``checks/test_leg15_closure.py`` and skip in any
workspace without archived run data.

The contract pinned in ``plan.md`` section LEG 32 is a pure function

    verdict_exit(readout: dict) -> int

with four outcomes:

    0   every pre-declared criterion holds, and the negative control is present,
        marked as having failed in the required direction, and carries evidence
        that it ran;
    61  a pre-declared criterion is violated (the message names which one);
    62  the negative control is missing, vacuous, or passed in the wrong
        direction;
    63  the readout carries no pre-declared criteria block at all.

The readout shape used by these tests, built entirely in memory:

    {
      'policy': 'leg15-closure-v1',
      'criteria': {'C1': True, ..., 'C6': True},
      'negative_control': {
        'case': 'lossy-control',
        'expect': '<what the control must show>',
        'status': 'failed-as-required',
        'executed': True,
        'observed': {'energy_residual': 0.34032},
      },
      'cases': [{'id': ..., 'energy_residual': ...}, ...],
    }

``criteria`` is the project's existing convention (``leg16`` and ``leg19`` both
store a name -> bool map under that key). The negative control's direction and
evidence are read from ``status``, ``executed`` and ``observed``.

The driver's top level does ``from parity import *``, which imports the solver.
The tests stub ``parity`` so the contract is exercised CPU-only, with no
``src/torchfdtd`` import, no GPU, no ``./tools/run.sh`` and no archived data.
``verdict_exit`` is required to be pure, so the stub is sufficient.
"""
import contextlib
import importlib
import io
import sys
import types
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

MISSING_CONTRACT = (
    'the pinned LEG 32 contract is missing: src/leg15_closure.py must expose '
    'verdict_exit(readout: dict) -> int, returning 0 only when every '
    "pre-declared criterion in readout['criteria'] holds and "
    "readout['negative_control'] ran and failed in the required direction; "
    '61 when a criterion is violated; 62 when the negative control is missing, '
    'vacuous or passed in the wrong direction; and 63 when the readout carries '
    'no criteria block at all'
)


@contextlib.contextmanager
def driver():
    """Import ``src/leg15_closure.py`` without importing the solver.

    The stub provides the names the driver would receive from ``parity``, so the
    module imports on a machine with no fork checkout. ``verdict_exit`` must not
    need the solver; if it does, the test that calls it reports that failure.
    """
    parity = types.ModuleType('parity')
    parity.np = np
    saved = sys.modules.pop('leg15_closure', None)
    try:
        with mock.patch.dict(sys.modules, {'parity': parity}):
            try:
                module = importlib.import_module('leg15_closure')
            except Exception as error:
                raise AssertionError(
                    'the pinned LEG 32 contract could not be imported: '
                    f'src/leg15_closure.py raised {type(error).__name__}: {error}. '
                    + MISSING_CONTRACT) from error
            yield module
    finally:
        sys.modules.pop('leg15_closure', None)
        if saved is not None:
            sys.modules['leg15_closure'] = saved


def leg15_readout():
    """A passing leg-15 readout, built in memory. No archived run is read."""
    return {
        'policy': 'leg15-closure-v1',
        'status': 'completed',
        'simulation_executed': True,
        'criteria': {'C1': True, 'C2': True, 'C3': True, 'C4': True, 'C5': True, 'C6': True},
        'negative_control': {
            'case': 'lossy-control',
            'expect': 'energy_residual at least 1e4 times the effect under test '
                      '(leg 12 measured 3.4032e-01)',
            'status': 'failed-as-required',
            'executed': True,
            'observed': {'energy_residual': 0.34032},
        },
        'cases': [
            {'id': 'baseline', 'energy_residual': 7.50792e-07},
            {'id': 'lossy-control', 'energy_residual': 0.34032},
        ],
    }


class VerdictExitContract(unittest.TestCase):
    def contract(self):
        """Return ``verdict_exit``, or fail with an explanation of what is missing."""
        with driver() as module:
            function = getattr(module, 'verdict_exit', None)
            if function is None:
                self.fail(MISSING_CONTRACT)
            return function

    def test_all_criteria_hold_and_negative_control_failed_as_required_returns_0(self):
        """The negative control for this leg: a function that refuses everything must fail here."""
        verdict_exit = self.contract()
        readout = leg15_readout()
        status = verdict_exit(readout)
        self.assertEqual(
            status, 0,
            'a readout whose six pre-declared criteria all hold, and whose negative control '
            'ran and failed in the required direction, must return 0. A verdict function that '
            'refuses every readout is not a verdict; it is a constant. '
            f'verdict_exit returned {status!r} for {readout!r}')

    def test_one_violated_criterion_returns_61_and_names_it(self):
        """A violated criterion is refused, and the message says which one."""
        verdict_exit = self.contract()
        readout = leg15_readout()
        readout['criteria']['C1'] = False
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            status = verdict_exit(readout)
        message = output.getvalue()
        self.assertEqual(
            status, 61,
            "criterion 'C1' is false, so the pinned outcome is 61 (a pre-declared criterion "
            f'is violated); verdict_exit returned {status!r}. Readout: {readout!r}')
        self.assertIn(
            'C1', message,
            "verdict_exit returned 61 but its message does not name the violated criterion "
            f"'C1', so the reader cannot tell which gate failed. Captured output: {message!r}")

    def test_negative_control_removed_or_vacuous_returns_62(self):
        """A control that is gone, or that carries no evidence it ran, cannot let the run pass."""
        verdict_exit = self.contract()
        removed = leg15_readout()
        del removed['negative_control']
        vacuous = leg15_readout()
        vacuous['negative_control'].update(executed=False, observed=None)
        for label, readout in (('removed', removed), ('present but vacuous', vacuous)):
            with self.subTest(negative_control=label):
                status = verdict_exit(readout)
                self.assertEqual(
                    status, 62,
                    f'the negative control is {label}, so the pinned outcome is 62 (missing, '
                    f'vacuous or wrong direction); verdict_exit returned {status!r}. '
                    f'Readout: {readout!r}')

    def test_negative_control_that_passed_returns_62(self):
        """A negative control that cannot fail is a description, not a control."""
        verdict_exit = self.contract()
        readout = leg15_readout()
        readout['negative_control']['status'] = 'passed'
        status = verdict_exit(readout)
        self.assertEqual(
            status, 62,
            'the negative control is marked as having passed, not as having failed in the '
            'required direction, so the pinned outcome is 62; '
            f'verdict_exit returned {status!r}. Readout: {readout!r}')

    def test_no_criteria_block_returns_63(self):
        """A readout that cannot state its own criteria is not a passing run."""
        verdict_exit = self.contract()
        readout = leg15_readout()
        del readout['criteria']
        status = verdict_exit(readout)
        self.assertEqual(
            status, 63,
            'the readout carries no pre-declared criteria block, so the pinned outcome is 63; '
            f'verdict_exit returned {status!r}. Readout: {readout!r}')


if __name__ == '__main__':
    unittest.main()
