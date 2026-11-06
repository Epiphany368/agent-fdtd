"""CPU contracts owned by this project; no simulation or reference-tree imports.

Drivers are imported, but main() is never run. The small comparison block is
compiled directly from each driver's AST to exercise its actual tolerance and
digest arithmetic without constructing a GPU simulation.
"""
import ast
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
EXPECTED_STAGES = ('F1-structure', 'F2-mesh', 'F3-source', 'F4-monitors', 'F5-run', 'F6-readout')
UPSTREAM = 'ebcc0f7d1361c29cb478dbd2a478fb4c58654223'


class GateContractTests(unittest.TestCase):
    def gate(self, config=None, *, upstream=UPSTREAM, upstream_dirty='', fork_dirty='',
             role='fork', cuda=False):
        config = {'task': 'CPU gate contract', 'kind': 'audit'} if config is None else config
        cuda_api = mock.Mock()
        cuda_api.is_available.return_value = cuda
        cuda_api.get_device_name.return_value = 'CPU test placeholder'

        def git_output(args, **kwargs):
            self.assertEqual(args[:2], ['git', '-C'])
            is_upstream = args[2].endswith('torchfdtd_upstream')
            if args[3:] == ['rev-parse', '--show-toplevel']:
                return str(Path(args[2]).resolve()) + '\n'
            if args[3:] == ['rev-parse', 'HEAD']:
                return (upstream if is_upstream else 'fork-test-commit') + '\n'
            self.assertEqual(args[3:], ['status', '--porcelain'])
            return upstream_dirty if is_upstream else fork_dirty

        with tempfile.TemporaryDirectory() as directory:
            config_file = Path(directory) / 'config.json'
            config_file.write_text(json.dumps(config))
            env = dict(PARITY_CONFIG=str(config_file), PARITY_SOLVER_ROLE=role,
                       PARITY_SOLVER_ROOT='', PARITY_SOLVER_UPSTREAM='')
            with mock.patch.dict(os.environ, env), \
                    mock.patch.object(subprocess, 'check_output', side_effect=git_output), \
                    mock.patch.dict(sys.modules, {'torch': mock.Mock(cuda=cuda_api)}), \
                    contextlib.redirect_stdout(io.StringIO()):
                namespace = runpy.run_path(str(ROOT / 'checks/pre_environment.py'))
        return namespace, cuda_api

    def test_pin_contains_relative_project_roots(self):
        pin = json.loads((ROOT / 'configs/baseline-pin.json').read_text())
        self.assertEqual(pin['solver_root'], 'src/torchfdtd')
        self.assertEqual(pin['upstream']['solver_root'], 'src/torchfdtd_upstream')
        self.assertEqual(pin['python'], '.venv/bin/python')
        self.assertEqual(pin['upstream']['solver_commit'], UPSTREAM)

    def test_audit_gate_records_dirty_fork_without_cuda(self):
        namespace, cuda = self.gate(fork_dirty=' M tracked.py\n?? new.py\n')
        self.assertEqual(namespace['solver_dirty'], 2)
        self.assertEqual(namespace['solver_commit'], 'fork-test-commit')
        cuda.is_available.assert_not_called()

    def test_upstream_role_selects_frozen_commit(self):
        namespace, _ = self.gate(role='upstream')
        self.assertEqual(namespace['solver_commit'], UPSTREAM)
        self.assertEqual(namespace['solver_dirty'], 0)

    def test_moved_upstream_fails_for_fork_role(self):
        with self.assertRaisesRegex(AssertionError, 'FROZEN UPSTREAM MOVED'):
            self.gate(upstream='different')

    def test_dirty_upstream_fails(self):
        with self.assertRaisesRegex(AssertionError, 'upstream is dirty'):
            self.gate(upstream_dirty=' M solver.py')

    def test_missing_task_fails(self):
        with self.assertRaisesRegex(AssertionError, 'task'):
            self.gate({'kind': 'audit'})

    def test_unknown_kind_fails(self):
        with self.assertRaisesRegex(AssertionError, 'Unknown task kind'):
            self.gate({'task': 'contract', 'kind': 'unknown'})

    def test_simulation_gate_requires_cuda(self):
        with self.assertRaisesRegex(AssertionError, 'GPU chain unavailable'):
            self.gate({'task': 'contract', 'kind': 'canary'})

    def test_resolve_relative_path_and_environment_override(self):
        namespace, _ = self.gate()
        resolve = namespace['resolve']
        with mock.patch.dict(os.environ, {'CPU_CONTRACT_ROOT': ''}):
            self.assertEqual(resolve('src/solver', 'CPU_CONTRACT_ROOT'), ROOT / 'src/solver')
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, {'CPU_CONTRACT_ROOT': directory}):
            self.assertEqual(resolve('unused', 'CPU_CONTRACT_ROOT'), Path(directory))

    def test_upstream_pin_fallback_precedence(self):
        namespace, _ = self.gate()
        expected = namespace['up_expected']
        self.assertEqual(expected({'upstream': {'solver_commit': 'nested'},
                                   'solver_commit_at_split': 'split'}), 'nested')
        self.assertEqual(expected({'solver_commit_at_split': 'split'}), 'split')
        self.assertEqual(expected({}), UPSTREAM)


class DriverContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Imports resolve the locally linked solver, never either reference project.
        # No bytecode/cache files are written into the shared solver tree.
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, {'MPLCONFIGDIR': directory}), \
                mock.patch.object(sys, 'dont_write_bytecode', True):
            cls.forward = importlib.import_module('leg1_forward')
            cls.objective = importlib.import_module('leg1_objective')
            cls.parity = importlib.import_module('parity')

    def drivers(self):
        return (self.forward, self.objective)

    def compare(self, driver, measured, recorded, tolerance):
        tree = ast.parse(Path(driver.__file__).read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'main')
        assignments = []
        for node in ast.walk(main):
            if isinstance(node, ast.Assign):
                target = node.targets[0]
                name = target.id if isinstance(target, ast.Name) else (
                    target.value.id if isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name) else None)
                if name in ('errors', 'ok'):
                    assignments.append(node)
        self.assertEqual(len(assignments), 3, 'Review the comparison contract if main changes')
        module = ast.Module(body=sorted(assignments, key=lambda node: node.lineno), type_ignores=[])
        namespace = dict(measured=measured, recorded=recorded, tolerance=tolerance,
                         relative_error=driver.relative_error)
        exec(compile(module, driver.__file__, 'exec'), namespace)
        return namespace['ok'], namespace['errors']

    def values(self, driver, value=1.0):
        keys = ('T', 'R') if driver is self.forward else (
            'J', 'P_up_reduced', 'P_in_reduced', 'unit_mode_flux_reduced')
        return dict.fromkeys(keys, value) | {'epsilon_sha256': 'digest-for-arithmetic-test'}

    def test_stage_order_is_pinned_in_both_drivers(self):
        for driver in self.drivers():
            with self.subTest(driver=driver.__name__):
                self.assertEqual(driver.STAGES, EXPECTED_STAGES)
                self.assertEqual(tuple(driver.STAGE_SOURCE), EXPECTED_STAGES)

    def test_forward_stage_sources_are_pinned(self):
        self.assertEqual(self.forward.STAGE_SOURCE, {
            'F1-structure': 'run_b9_native_shape_n8.scene',
            'F2-mesh': 'run_b9_native_shape_n8.scene (Region)',
            'F3-source': 'torchfdtd.mode_branches.prepare_aperture_modal_launch',
            'F4-monitors': 'run_b9_native_shape_n8.scene (FieldMonitor list)',
            'F5-run': 'torchfdtd.mode_injection.ModeInjectedPlaneSimulation',
            'F6-readout': 'torchfdtd.mode_injection.modal_plane_amplitudes',
        })

    def test_objective_stage_sources_are_pinned(self):
        self.assertEqual(self.objective.STAGE_SOURCE, {
            'F1-structure': 'run_g5_discretisation.g5_fixture (build), on run_g1b_gradient_taylor.fixture',
            'F2-mesh': 'run_b9_native_shape_n8.scene (Region) via run_g1b_gradient_taylor.fixture',
            'F3-source': 'torchfdtd.mode_branches.prepare_aperture_modal_launch',
            'F4-monitors': 'run_b9_native_shape_n8.scene (FieldMonitor list)',
            'F5-run': 'torchfdtd.mode_injection.ModeInjectedPlaneSimulation',
            'F6-readout': 'torchfdtd.mode_injection.modal_plane_amplitudes + the detector basis',
        })

    def test_stages_are_actual_imports_from_existing_modules(self):
        scene = importlib.import_module('run_b9_native_shape_n8').scene
        native = importlib.import_module('run_g1b_gradient_taylor')
        g5 = importlib.import_module('run_g5_discretisation')
        branches = importlib.import_module('torchfdtd.mode_branches')
        injection = importlib.import_module('torchfdtd.mode_injection')
        self.assertIs(self.forward.scene, scene)
        self.assertIs(self.objective.g5_fixture, g5.g5_fixture)
        self.assertIs(g5.fixture, native.fixture)
        self.assertIs(native.scene, scene)
        for module in (self.forward, native):
            self.assertIs(module.prepare_aperture_modal_launch, branches.prepare_aperture_modal_launch)
            self.assertIs(module.ModeInjectedPlaneSimulation, injection.ModeInjectedPlaneSimulation)
            self.assertIs(module.modal_plane_amplitudes, injection.modal_plane_amplitudes)

    def test_load_run_requires_launcher(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'tools/run.sh'):
                self.parity.load_run()

    def test_load_run_binds_config_run_pin_and_effective_json(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            config = {'task': 'CPU load contract', 'kind': 'audit', 'extra': [1, 2]}
            config_path = run / 'config.json'
            effective = run / 'effective.json'
            config_path.write_text(json.dumps(config))
            with mock.patch.dict(os.environ, dict(RUN_DIR_ABS=str(run),
                    PARITY_CONFIG=str(config_path), EFFECTIVE_JSON=str(effective))):
                loaded, loaded_run = self.parity.load_run()
            self.assertEqual((loaded, loaded_run), (config, run))
            result = json.loads(effective.read_text())
            self.assertEqual(result['config'], config)
            self.assertEqual(result['solver_pin'], json.loads((ROOT / 'configs/baseline-pin.json').read_text()))
            self.assertEqual(result['resolved_cases'], [])
            self.assertEqual(result['conventions']['geometry_units'], 'um')
            self.assertEqual(result['conventions']['time_units'], 's')

    def test_load_run_rejects_malformed_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'invalid.json'
            config.write_text('{')
            with mock.patch.dict(os.environ, dict(RUN_DIR_ABS=directory, PARITY_CONFIG=str(config))):
                with self.assertRaises(json.JSONDecodeError):
                    self.parity.load_run()

    def test_relative_error_signed_and_zero_targets(self):
        for driver in self.drivers():
            for measured, target, expected in ((2., 2., 0.), (3., 2., .5),
                    (-3., -2., .5), (2., -2., 2.), (-.25, 0., .25), (0., 0., 0.)):
                with self.subTest(driver=driver.__name__, measured=measured, target=target):
                    self.assertEqual(driver.relative_error(measured, target), expected)

    def test_relative_error_preserves_tiny_power_scale(self):
        for driver in self.drivers():
            self.assertAlmostEqual(driver.relative_error(3.3e-42, 3e-42), .1)

    def test_tolerance_boundary_is_inclusive(self):
        for driver in self.drivers():
            measured, recorded = self.values(driver, 1.25), self.values(driver)
            ok, errors = self.compare(driver, measured, recorded, .25)
            self.assertTrue(ok)
            self.assertEqual(max(errors.values()), .25)
            self.assertFalse(self.compare(driver, measured, recorded, .249)[0])

    def test_zero_target_uses_absolute_error_in_comparison(self):
        for driver in self.drivers():
            self.assertTrue(self.compare(driver, self.values(driver, .125),
                                         self.values(driver, 0.), .125)[0])

    def test_any_number_mismatch_fails_comparison(self):
        for driver in self.drivers():
            for key in self.values(driver):
                if key == 'epsilon_sha256':
                    continue
                measured = self.values(driver)
                measured[key] = 2.
                with self.subTest(driver=driver.__name__, key=key):
                    self.assertFalse(self.compare(driver, measured, self.values(driver), 1e-12)[0])

    def test_digest_mismatch_fails_comparison(self):
        for driver in self.drivers():
            measured = self.values(driver)
            measured['epsilon_sha256'] = 'changed'
            ok, errors = self.compare(driver, measured, self.values(driver), 1e-12)
            self.assertFalse(ok)
            self.assertEqual(errors['epsilon_sha256'], 1.)

    def test_nonfinite_measurement_fails_driver_comparison(self):
        for driver in self.drivers():
            for value in (float('nan'), float('inf')):
                self.assertFalse(self.compare(driver, self.values(driver, value),
                                              self.values(driver), 1e-12)[0])

    def test_registered_configs_pin_tolerance_and_case_ids(self):
        for name in ('leg1-n8-forward', 'leg1-objective'):
            cfg = json.loads((ROOT / f'configs/{name}.json').read_text())
            self.assertEqual(cfg['expected']['tolerance_relative'], 1e-12)
            self.assertEqual({case['id'] for case in cfg['cases']}, set(cfg['expected']['cases']))
            self.assertTrue(cfg['expected']['source'])
