"""Leg 15 contracts: the archived data alone already attributes the `7.5e-7` residual.

The test reads leg 12's archived non-empty-box run (monitor arrays plus readout) and shows,
without running anything, that the residual is the gap between the two ways of composing `R` from
the same stored numbers -- not a physical flux mismatch. It also shows the measure can fail, and
that the "closed box" of leg 12 is not closed.
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))

import parity  # noqa: E402


def find_leg12_slab_run():
    """The archived leg-12 run that carries the slab monitor arrays, or None."""
    for readout in sorted((ROOT / 'runs').glob('*/readout.json')):
        try:
            data = json.loads(readout.read_text())
        except (OSError, ValueError):
            continue
        if data.get('policy') != 'leg12-closed-box-v1':
            continue
        case = next((c for c in data.get('cases', []) if c.get('id') == 'slab'), None)
        arrays = readout.parent / 'slab-sample-monitors.npz'
        if case is not None and arrays.exists():
            return readout.parent, case['measured'], np.load(arrays)
    return None, None, None


RUN_DIR, MEASURED, ARRAYS = find_leg12_slab_run()


def find_case(policy, case_id):
    """A case's measured block from any archived run of ``policy``, or None."""
    for readout in sorted((ROOT / 'runs').glob('*/readout.json')):
        try:
            data = json.loads(readout.read_text())
        except (OSError, ValueError):
            continue
        if data.get('policy') != policy:
            continue
        for case in data.get('cases', []):
            if case.get('id') == case_id:
                return case['measured']
    return None


def leg15_results():
    """Every case of the newest archived leg-15 run: id -> measured block, or {}."""
    runs = []
    for readout in sorted((ROOT / 'runs').glob('*/readout.json')):
        try:
            data = json.loads(readout.read_text())
        except (OSError, ValueError):
            continue
        if data.get('policy') == 'leg15-closure-v1':
            runs.append(data)
    return {c['id']: c['measured'] for c in runs[-1]['cases']} if runs else {}


class TestLeg15Closure(unittest.TestCase):
    def test_the_fixture_is_not_a_closed_box(self):
        """Leg 12 called it a closed box; `parity.project_for` gives both x ends a PML."""
        project = parity.project_for('slab', 3.5, 40, 260)
        self.assertEqual(project.region.pml_cells, 32)
        self.assertEqual(project.region.boundaries.y_min.kind, 'periodic')
        self.assertEqual(project.region.boundaries.y_max.kind, 'periodic')
        # No wall is declared on x: the default faces are the PML, so both x exits absorb.
        self.assertEqual(project.region.boundaries.x_min.kind, 'pml')
        self.assertEqual(project.region.boundaries.x_max.kind, 'pml')

    def test_the_residual_is_the_r_composition_gap(self):
        """The archived residual equals the gap between the two ways of composing R."""
        if MEASURED is None:
            self.skipTest('no archived leg-12 run on this machine')
        R = np.asarray(MEASURED['R'], dtype=float)
        T = np.asarray(MEASURED['T'], dtype=float)
        residual = float(np.max(np.abs(R + T - 1.0)))
        self.assertAlmostEqual(residual, MEASURED['energy_residual'], places=15)
        # 1 - flux_sample/|flux_empty| is R composed from the stored totals, i.e. the same
        # identity the closure test asks for. `R` itself is the difference-field flux over the
        # same denominator. The two differ by exactly the residual.
        flux_sample = np.asarray(ARRAYS['reflection_flux'], dtype=float)
        flux_empty = np.asarray(MEASURED['reference_power_reduced'], dtype=float)
        composition_gap = np.max(np.abs(R - (1.0 - flux_sample / np.abs(flux_empty))))
        self.assertEqual(float(np.argmax(np.abs(R + T - 1.0))),
                         float(np.argmax(np.abs(R - (1.0 - flux_sample / np.abs(flux_empty))))))
        self.assertLess(abs(composition_gap - residual), 0.05 * residual)
        # And the sample's own net flux at the two monitors agrees far better than the residual:
        # the slab's excess over that mismatch is a composition term, not a leak.
        flux_trans = np.asarray(ARRAYS['transmission_flux'], dtype=float)
        mismatch = float(np.max(np.abs(flux_trans / flux_sample - 1.0)))
        self.assertGreater(mismatch, 1e-9)
        self.assertLess(mismatch, 0.1 * residual)

    def test_the_two_empty_run_denominators_are_not_the_limit(self):
        """The denominators of R and T agree far better than the residual they produce."""
        if MEASURED is None:
            self.skipTest('no archived leg-12 run on this machine')
        T = np.asarray(MEASURED['T'], dtype=float)
        flux_trans = np.asarray(ARRAYS['transmission_flux'], dtype=float)
        denominator_transmission = flux_trans / T
        denominator_reflection = np.asarray(MEASURED['reference_power_reduced'], dtype=float)
        asymmetry = float(np.max(np.abs(denominator_reflection / denominator_transmission - 1.0)))
        self.assertLess(asymmetry, 1e-8)

    def test_the_interface_case_residual_is_the_same_flux_mismatch(self):
        """Leg 3's single-interface residual is the monitor-pair mismatch, nothing else."""
        if MEASURED is None:
            self.skipTest('no archived leg-12 run on this machine')
        # leg 4's ppw-40 interface rung: the same monitors, the same source, a single interface.
        interface = find_case('leg4-convergence-v1', 'ladder-interface-ppw40')
        if interface is None:
            self.skipTest('no archived leg-4 run on this machine')
        flux_sample = np.asarray(ARRAYS['reflection_flux'], dtype=float)
        flux_trans = np.asarray(ARRAYS['transmission_flux'], dtype=float)
        mismatch = float(np.max(np.abs(flux_trans / flux_sample - 1.0)))
        self.assertLess(abs(interface['energy_residual'] - mismatch), 0.05 * mismatch)

    def test_the_measure_can_fail(self):
        """A deliberately wrong denominator must move the composition gap by orders of magnitude."""
        if MEASURED is None:
            self.skipTest('no archived leg-12 run on this machine')
        R = np.asarray(MEASURED['R'], dtype=float)
        flux_sample = np.asarray(ARRAYS['reflection_flux'], dtype=float)
        flux_empty = np.asarray(MEASURED['reference_power_reduced'], dtype=float)
        wrong = np.max(np.abs(R - (1.0 - flux_sample / (1.001 * np.abs(flux_empty)))))
        self.assertGreater(wrong, 1e-4)


class TestLeg15Ladder(unittest.TestCase):
    """The ladder run itself: the knob that controls the residual, and the two laws."""

    def setUp(self):
        self.results = leg15_results()
        if not self.results:
            self.skipTest('no archived leg-15 run on this machine')

    def residual(self, case):
        return self.results[case]['energy_residual']

    def test_c3_the_baseline_reproduces_the_archived_number_exactly(self):
        """The ladder's baseline row is leg 12's fixture, to the last digit."""
        if MEASURED is None:
            self.skipTest('no archived leg-12 run on this machine')
        self.assertEqual(self.residual('baseline'), MEASURED['energy_residual'])

    def test_pml_cells_is_the_controlling_knob(self):
        """Twice the PML at 8 -> 16 -> 32 cells removes ~13x of the residual each time."""
        ladder = [8, 16, 32]
        values = [self.residual({8: 'pml8', 16: 'pml16', 32: 'baseline'}[cells]) for cells in ladder]
        for coarse, fine in zip(values, values[1:]):
            self.assertGreater(coarse / fine, 5.0)
        self.assertLess(self.residual('pml50'), self.residual('pml48'))
        self.assertLess(self.residual('pml48'), self.residual('pml40'))
        # The single-interface fixture obeys the same knob.
        self.assertGreater(self.residual('iface-pml8') / self.residual('iface-pml16'), 5.0)
        self.assertGreater(self.residual('iface-pml16') / self.residual('iface'), 5.0)

    def test_the_residual_is_linear_in_the_reflected_amplitude(self):
        """An interference term scales with r; a power leak would scale with r squared."""
        for pml, keys in ((32, {3.5: 'iface', 2.0: 'iface-n2', 1.5: 'iface-n15'}),
                          (50, {3.5: 'iface-pml50', 2.0: 'iface-n2-pml50',
                                1.5: 'iface-n15-pml50'})):
            coefficients = [self.residual(keys[n]) / ((n - 1) / (n + 1)) for n in (3.5, 2.0, 1.5)]
            spread = max(coefficients) / min(coefficients)
            self.assertLess(spread, 1.05, f'pml {pml}: residual/r is not constant: {coefficients}')

    def test_the_unrelated_knobs_do_not_move_the_magnitude(self):
        """ppw, duration, padding and grid density leave the magnitude where it was."""
        baseline = self.residual('baseline')
        for case in ('ppw20', 'ppw80', 'long', 'fine', 'pad12', 'pad16', 'slab-before'):
            self.assertLess(abs(self.residual(case) / baseline - 1.0), 0.10, case)

    def test_the_null_has_no_residual_and_the_control_is_far_above(self):
        """No interface -> exactly zero; a lossy slab -> 10^5 times the effect under test."""
        self.assertEqual(self.residual('null-index1'), 0.0)
        self.assertGreater(self.residual('lossy-control') / self.residual('baseline'), 1e4)


if __name__ == '__main__':
    unittest.main()
