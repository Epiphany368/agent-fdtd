"""Leg 20 contracts: the opt-in realised-axis-minimum time-step bound, without a GPU.

  - the config carries the 28 frozen P0-b grids with their source hashes and layouts, and no
    path that would read a reference repository at run time;
  - the per-axis-minimum form in SI reproduces every stored vendor dt to 2e-16 (C1);
  - the operator eigenvalue helper matches the closed forms on uniform grids, so the spectral
    margin it reports is the operator's and not the helper's;
  - the fork rule is opt-in: the default dump does not carry it, an explicit grid gives the same
    time step under both rules, a graded grid reads its minimum from the realised nodes, and an
    override above the opt-in bound is refused.
"""
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import leg20_dt_rule as leg20  # noqa: E402
from parity import C0, Project, Region  # noqa: E402

CONFIG = json.loads((ROOT / 'configs' / 'leg20-dt-rule.json').read_text())


class Config(unittest.TestCase):
    def test_frozen_grids_and_provenance(self):
        grids = CONFIG['frozen_grids']
        self.assertEqual(len(grids), 28)
        sources = {s['path'].replace('$PARITY_REF/', ''): s['sha256'] for s in CONFIG['provenance']['sources']}
        self.assertEqual(len(sources), 11)
        for grid in grids:
            path = grid['id'].rsplit(':', 1)[0]
            self.assertEqual(sources[path], grid['source_sha256'])
            self.assertEqual(grid['layout']['dimension'], '2d')
            self.assertIn(grid['layout']['interface'], ('staircase', 'conformal'))
            self.assertGreater(grid['vendor_dt_s'], 0)
        self.assertEqual(sum(g['layout']['interface'] == 'staircase' for g in grids), 20)
        self.assertEqual(CONFIG['relative_tolerance'], 2e-16)
        self.assertFalse(CONFIG['grating']['gated'])

    def test_no_reference_root_at_run_time(self):
        source = (ROOT / 'src' / 'leg20_dt_rule.py').read_text()
        self.assertNotIn('PARITY_REF', source.replace('`$PARITY_REF/src/vendor_adapter.py`', ''))
        self.assertNotIn('20261010_torchfdtd-lumerical-parity', source)


class Arithmetic(unittest.TestCase):
    def test_c1_every_frozen_grid(self):
        for grid in CONFIG['frozen_grids']:
            nodes = [grid['nodes_m'][a] for a in ('x', 'y')]
            bound, minima = leg20.axis_minimum_bound_si(nodes, CONFIG['courant_factor'])
            self.assertLessEqual(leg20.relative(bound, grid['vendor_dt_s']), 2e-16, grid['id'])
            self.assertTrue(all(h <= grid['nominal_override_m'][k] * (1 + 1e-12)
                                for h, k in zip(minima, ('dx', 'dy'))), grid['id'])

    def test_form_uses_each_axis_minimum(self):
        x = np.r_[0, np.cumsum([3, 1, 2, 4])] * 1e-9
        y = np.r_[0, np.cumsum([5, 5, 7])] * 1e-9
        bound, minima = leg20.axis_minimum_bound_si([x, y], 0.99)
        np.testing.assert_allclose(minima, [1e-9, 5e-9], rtol=1e-15)
        self.assertLess(leg20.relative(bound, 0.99 / (C0 * math.sqrt(1e18 + 1 / 25e-18))), 1e-15)
        # The global-minimum form is smaller on an anisotropic grid; it is not the same rule.
        self.assertLess(leg20.global_minimum_bound_si([x, y], 0.99), bound)

    def test_rejects_degenerate_spacing(self):
        with self.assertRaises(ValueError):
            leg20.axis_minimum_bound_si([[0, 1e-9, 1e-9], [0, 1e-9]], 0.99)

    def test_eigenvalue_closed_forms(self):
        h, n = 2e-9, 40
        dirichlet = leg20.largest_eigenvalue(np.full(n, h), periodic=False)
        self.assertAlmostEqual(dirichlet / (4 / h**2 * math.sin((n - 1) * math.pi / (2 * n))**2), 1, delta=1e-12)
        periodic = leg20.largest_eigenvalue(np.full(n, h), periodic=True)
        self.assertAlmostEqual(periodic / (4 / h**2), 1, delta=1e-12)
        # At the 2-D bound with factor 1 the uniform periodic operator sits exactly at the limit.
        dt = 1 / (C0 * math.sqrt(2) / h)
        self.assertAlmostEqual(leg20.courant_ratio(dt, 2 * periodic), 1, delta=1e-12)

    def test_graded_widths_stay_under_the_bound(self):
        widths = np.r_[np.full(10, 1.0), 1.25 ** np.arange(1, 6), np.full(10, 1.25 ** 5)] * 1e-8
        lam = leg20.largest_eigenvalue(widths, False) + leg20.largest_eigenvalue(widths, True)
        dt = 1 / (C0 * math.sqrt(2) / widths.min())
        self.assertLess(leg20.courant_ratio(dt, lam), 1)


def graded_region(**extra):
    return Region(dimension='2d', size=(6.0, 0.6, 0.1), mesh=0.008571428571428572, mesh_type='graded',
                  material_sampling='yee', pml_cells=8,
                  boundaries=dict(y_min=dict(kind='periodic'), y_max=dict(kind='periodic')), **extra)


class ForkRule(unittest.TestCase):
    def test_default_dump_is_unchanged(self):
        self.assertNotIn('time_step_rule', Region().model_dump())
        self.assertNotIn('time_step_rule', graded_region().model_dump())
        self.assertEqual(graded_region(time_step_rule='realised_axis_minimum').model_dump()['time_step_rule'],
                         'realised_axis_minimum')
        self.assertNotEqual(graded_region(), graded_region(time_step_rule='realised_axis_minimum'))

    def test_explicit_grid_same_under_both_rules(self):
        grid = CONFIG['frozen_grids'][0]
        a, b = leg20.fork_region_um(grid, 'nominal'), leg20.fork_region_um(grid, 'realised_axis_minimum')
        self.assertEqual(a.cfl_time_step, b.cfl_time_step)
        self.assertLess(leg20.relative(b.cfl_time_step, grid['vendor_dt_s']), 1e-12)

    def test_graded_reads_realised_nodes(self):
        region = graded_region(time_step_rule='realised_axis_minimum')
        minima = [float(np.diff(n).min()) for n in region.mesh_nodes[:2]]
        self.assertEqual(list(region.realised_axis_minima), minima)
        expected = region.courant_factor * 1e-6 / (C0 * math.sqrt(sum(1 / h**2 for h in minima)))
        self.assertEqual(region.cfl_time_step, expected)
        self.assertEqual(region.time_step, expected)
        self.assertAlmostEqual(region.rectangular_courant,
                               region.time_step * C0 / (region.reference_step * 1e-6), delta=1e-15)
        nominal = graded_region()
        self.assertLess(abs(region.time_step / nominal.time_step - 1), 1e-10)

    def test_override_above_opt_in_bound_is_refused(self):
        bound = graded_region(time_step_rule='realised_axis_minimum').cfl_time_step
        graded_region(time_step_rule='realised_axis_minimum', time_step_override=bound)
        with self.assertRaises(ValueError):
            graded_region(time_step_rule='realised_axis_minimum', time_step_override=bound * (1 + 1e-9))

    def test_native_graded_project_round_trips(self):
        case = CONFIG['native_graded_cases'][0]
        project = Project.model_validate(case['project'])
        self.assertEqual(project.region.mesh_type, 'graded')
        self.assertNotIn('time_step_rule', project.model_dump()['region'])
        row = leg20.native_graded(case)
        self.assertLess(row['spectral_courant_ratio_realised_dt'], 1)
        self.assertTrue(all(v >= 0 for v in row['minimum_below_nominal_relative']))


if __name__ == '__main__':
    unittest.main()
