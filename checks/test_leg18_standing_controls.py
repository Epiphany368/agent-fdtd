"""Leg 18 contracts: the absorber depth is an explicit input, and the default path is inert.

`parity.project_for` builds the fixture of legs 3, 4, 12 and 13. Leg 18 made its PML depth an input
whose default stays 32. These checks hold that change to its promise without a GPU:

  - the default builds the same project as an explicit 32, and the archived name, span and shape;
  - a deeper absorber moves only the absorber: interior, source and monitors stay put;
  - the configs say what the leg registered (48 cells, tolerance 1e-6, a default-depth twin);
  - every number in the rewritten `basis` appears in the leg-18 run report (C4).

When archived runs are present they are compared leaf by leaf (C2, C3). The archive of a worktree
holds only its own runs; set `LEG18_ARCHIVE_RUNS` to another `runs/` directory to search it too.

CLI:  python checks/test_leg18_standing_controls.py compare <run-dir-a> <run-dir-b>
"""
import inspect
import json
import math
import os
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import parity  # noqa: E402

REPORT = ROOT / 'docs' / 'leg18-2026-10-19.md'
STANDING = ROOT / 'configs' / 'leg13-standing-controls.json'
DEPTH_48 = ROOT / 'configs' / 'leg18-standing-controls-48.json'
DEPTH_DEFAULT = ROOT / 'configs' / 'leg18-standing-controls-default.json'
# The leg-13 run whose numbers C2 must reproduce, and its recorded closure residual.
LEG13_RUN = '20261019T032219Z-21564'
LEG13_SLAB_RESIDUAL = 7.507917795246399e-07
RELATIVE = 1e-12

# Leaves that measure the machine, not the physics: excluded from the comparison and reported.
TIMING = re.compile(r'(seconds|elapsed|_utc)$')


def load(path):
    return json.loads(Path(path).read_text())


def leaves(value, path=''):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, f'{path}.{key}' if path else key)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from leaves(item, f'{path}[{i}]')
    else:
        yield path, value


def relative(a, b):
    if a == b:
        return 0.0
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.inf
    return abs(a - b) / max(abs(a), abs(b))


def compare_readouts(a, b):
    """Every shared numeric leaf of two readouts, with the worst relative difference.

    Timing leaves are skipped and counted. Leaves present on one side only are listed, never
    silently dropped; a non-numeric leaf that differs is a mismatch.
    """
    left, right = dict(leaves(a)), dict(leaves(b))
    shared = sorted(set(left) & set(right))
    numeric = exact = 0
    worst, worst_path, mismatched, timing = 0.0, None, [], []
    for path in shared:
        x, y = left[path], right[path]
        if TIMING.search(path.split('.')[-1].split('[')[0]):
            timing.append(path)
            continue
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            numeric += 1
            exact += x == y
            r = relative(float(x), float(y))
            if r > worst:
                worst, worst_path = r, path
            if r > RELATIVE:
                mismatched.append(path)
        elif x != y:
            mismatched.append(path)
    return dict(numeric_leaves=numeric, bit_identical_leaves=exact, max_relative=worst,
                worst_path=worst_path, mismatched=mismatched, timing_skipped=len(timing),
                only_in_first=sorted(set(left) - set(right)),
                only_in_second=sorted(set(right) - set(left)))


def run_roots():
    roots = [ROOT / 'runs']
    if os.environ.get('LEG18_ARCHIVE_RUNS'):
        roots.append(Path(os.environ['LEG18_ARCHIVE_RUNS']))
    return roots


def find_run(name):
    for root in run_roots():
        if (root / name / 'readout.json').is_file():
            return root / name
    return None


def standing_runs(pml_cells):
    """Standing-control runs after leg 13 at one depth (read from the archived config), oldest first."""
    found = []
    for root in run_roots():
        for readout in sorted(root.glob('*/readout.json')):
            try:
                data = json.loads(readout.read_text())
            except (OSError, ValueError):
                continue
            if data.get('policy') != 'leg13-standing-controls-v1':
                continue
            config = readout.parent / 'config.input.json'
            if not config.is_file():
                continue
            depth = json.loads(config.read_text()).get('pml_cells', parity.DEFAULT_PML_CELLS)
            if depth == pml_cells and readout.parent.name > LEG13_RUN:
                found.append(readout.parent)
    return found


def dump(project):
    """A project dump without structure ids, which `Structure` draws at random on every call."""
    data = project.model_dump()
    for structure in data['structures']:
        structure.pop('id', None)
    return json.dumps(data, sort_keys=True)


class DefaultDepth(unittest.TestCase):
    def test_default_is_32(self):
        default = inspect.signature(parity.project_for).parameters['pml_cells'].default
        self.assertEqual(default, 32)
        self.assertEqual(parity.DEFAULT_PML_CELLS, 32)

    def test_default_equals_explicit_32(self):
        for kind in ('empty', 'interface', 'slab'):
            for index, ppw in ((3.5, 40), (1.5, 20), (3.5, 80)):
                with self.subTest(kind=kind, index=index, ppw=ppw):
                    self.assertEqual(dump(parity.project_for(kind, index, ppw, 260)),
                                     dump(parity.project_for(kind, index, ppw, 260, pml_cells=32)))

    def test_default_keeps_the_archived_fixture(self):
        """Leg 13 ran shape [998, 6, 1] under these names; leg 18 must not move either."""
        for kind in ('interface', 'slab', 'empty'):
            project = parity.project_for(kind, 3.5, 40, 260, frequency_points=21)
            self.assertEqual(project.name, f'{kind}-n3.5-ppw40')
            self.assertEqual(tuple(project.region.shape), (998, 6, 1))
            self.assertEqual(project.region.pml_cells, 32)

    def test_rejects_a_depth_that_is_not_a_positive_integer(self):
        for bad in (0, -1, 48.0, True, '48'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parity.project_for('slab', 3.5, 40, 260, pml_cells=bad)


class DeeperAbsorber(unittest.TestCase):
    def test_only_the_absorber_moves(self):
        for kind in ('interface', 'slab', 'empty'):
            with self.subTest(kind=kind):
                p32 = parity.project_for(kind, 3.5, 40, 260, frequency_points=21)
                p48 = parity.project_for(kind, 3.5, 40, 260, frequency_points=21, pml_cells=48)
                self.assertEqual(tuple(p48.region.shape), (998 + 32, 6, 1))
                self.assertEqual(p48.region.pml_cells, 48)
                self.assertEqual(p48.name, f'{kind}-n3.5-ppw40-pml48')
                self.assertEqual(p32.region.interior_bounds(0), p48.region.interior_bounds(0))
                self.assertEqual(p32.region.time_step, p48.region.time_step)
                self.assertEqual(p32.region.steps, p48.region.steps)
                self.assertEqual(p32.sources[0].model_dump(), p48.sources[0].model_dump())
                self.assertEqual([m.model_dump() for m in p32.monitors],
                                 [m.model_dump() for m in p48.monitors])

    def test_slab_keeps_its_physical_thickness(self):
        project = parity.project_for('slab', 3.5, 40, 260, frequency_points=21, pml_cells=48)
        parity.geometry_assertions(project)


class Configs(unittest.TestCase):
    def test_standing_set_runs_at_48_with_the_tolerance_unchanged(self):
        cfg = load(STANDING)
        self.assertEqual(cfg['pml_cells'], 48)
        self.assertEqual(cfg['controls']['energy_closure']['tolerance'], 1e-06)
        self.assertEqual(cfg['controls']['symmetric_source']['tolerance'], 0.001)

    def test_default_twin_is_the_standing_set_with_no_depth(self):
        standing, twin = load(STANDING), load(DEPTH_DEFAULT)
        self.assertNotIn('pml_cells', twin)
        for cfg in (standing, twin):
            cfg.pop('task')
            cfg.pop('pml_cells', None)
            cfg['controls']['energy_closure'].pop('basis')
        self.assertEqual(standing, twin)

    def test_48_config_is_the_standing_set(self):
        standing, deep = load(STANDING), load(DEPTH_48)
        for cfg in (standing, deep):
            cfg.pop('task')
            cfg['controls']['energy_closure'].pop('basis')
        self.assertEqual(standing, deep)


class Basis(unittest.TestCase):
    """C4: the basis carries the measurement, and the report carries every number in it."""

    def basis(self):
        return load(STANDING)['controls']['energy_closure']['basis']

    def test_basis_says_what_the_floor_is(self):
        text = self.basis()
        self.assertIn('48', text)
        self.assertIn('margin', text)
        self.assertIn('echo', text)
        self.assertIn('fixture-bound', text)
        self.assertNotIn('rounded up', text)

    def test_every_basis_number_is_in_the_run_report(self):
        report = REPORT.read_text()
        numbers = set(re.findall(r'\d+(?:\.\d+)?(?:e[-+]?\d+)?x?', self.basis()))
        missing = sorted(n for n in numbers if n not in report)
        self.assertEqual(missing, [], f'basis numbers absent from {REPORT.name}')


@unittest.skipUnless(find_run(LEG13_RUN) and standing_runs(32),
                     'needs the leg-13 run and a leg-18 default-depth run (see LEG18_ARCHIVE_RUNS)')
class ReproducesLeg13(unittest.TestCase):
    """C2: the default depth reproduces the leg-13 recorded numbers to 1e-12 relative."""

    def test_every_numeric_leaf(self):
        archived = load(find_run(LEG13_RUN) / 'readout.json')
        for run in standing_runs(32):
            with self.subTest(run=run.name):
                result = compare_readouts(archived, load(run / 'readout.json'))
                self.assertGreater(result['numeric_leaves'], 100)
                self.assertEqual(result['mismatched'], [])
                self.assertLessEqual(result['max_relative'], RELATIVE)
                self.assertEqual(result['only_in_first'], [])
                # The first leg-18 runs also wrote the depth into the readout; that was withdrawn
                # (the coverage audit does not classify it) and is the only extra key allowed.
                self.assertLessEqual(set(result['only_in_second']), {'pml_cells'})

    def test_slab_residual(self):
        for run in standing_runs(32):
            slab = next(c for c in load(run / 'readout.json')['cases'] if c['id'] == 'control-slab')
            self.assertLessEqual(relative(slab['measured']['energy_residual'], LEG13_SLAB_RESIDUAL),
                                 RELATIVE)


@unittest.skipUnless(standing_runs(48), 'needs a leg-18 48-cell run')
class PassesAt48(unittest.TestCase):
    """C1: all three cases pass at 48 cells, and the closure residuals sit below the 32-cell ones."""

    def test_all_controls_pass(self):
        for run in standing_runs(48):
            readout = load(run / 'readout.json')
            self.assertEqual(readout['status'], 'completed')
            self.assertTrue(all(c['passed'] for c in readout['standing_controls']))
            self.assertTrue(all(n['behaved'] for n in readout['negative_controls']))
            self.assertEqual(readout['ledger_failures'], [])
            slab = next(c for c in readout['standing_controls'] if c['case'] == 'control-slab')
            self.assertLess(slab['value'], LEG13_SLAB_RESIDUAL)


def main(argv):
    if len(argv) == 3 and argv[0] == 'compare':
        result = compare_readouts(load(Path(argv[1]) / 'readout.json'), load(Path(argv[2]) / 'readout.json'))
        print(json.dumps(result, indent=2))
        return 0 if not result['mismatched'] else 1
    return unittest.main(argv=[sys.argv[0], *argv])


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
