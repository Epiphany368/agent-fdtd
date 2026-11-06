"""LEG 28 -- a missing reference contract must be a declared refusal, not a traceback.

``checks/pre_environment.py`` enters the reference branch with
``from reference_contract import load_reference`` *outside* the ``try``, and no
``reference_contract.py`` exists in this repository. Any config with
``kind: alignment`` or a ``reference`` field therefore dies with a bare
``ModuleNotFoundError`` instead of exiting through the declared refusal path.

The policy pinned in ``plan.md`` section LEG 28 is exercised here end to end:
pre-flight runs as a subprocess (the defect is that the process dies), against a
temporary config, and must exit ``65`` with a message that names, in order,
(1) that the reference contract module is missing, (2) the config field that
required it, and (3) that no simulation was run.

The invalid-reference refusal at ``checks/pre_environment.py:102-105`` is not
invoked: reaching it would require creating the missing module, which this leg
is forbidden to do. Test 4 therefore asserts message-level distinguishability
from the invalid-reference vocabulary already in that branch.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / 'checks' / 'pre_environment.py'
REFUSAL_EXIT = 65
PREFLIGHT_TIMEOUT = 120

MISSING_CONTRACT = re.compile(
    r'(?:'
    r'no module named\s+[`\'"]?reference_contract'
    r'|reference[_ ]contract[^.\n]{0,80}(?:missing|absent|not found|cannot be imported)'
    r'|(?:missing|absent|not found|cannot be imported)[^.\n]{0,80}reference[_ ]contract'
    r')',
    re.IGNORECASE,
)

ALIGNMENT_FIELD = re.compile(r'kind[^\n]{0,20}alignment', re.IGNORECASE)

REFERENCE_FIELD = re.compile(
    r'(?:config(?:uration)?\s*)?(?:field|key)\s*[`\'"]?reference'
    r'|[`\'"]reference[`\'"]\s*(?:field|key)'
    r'|config\s*\[\s*[`\'"]reference[`\'"]?\s*\]',
    re.IGNORECASE,
)

NO_SIMULATION = re.compile(r'no simulation\b', re.IGNORECASE)

INVALID_REFERENCE = re.compile(
    r'reference[^.\n]{0,40}unavailable/invalid'
    r'|invalid reference'
    r'|reference unavailable',
    re.IGNORECASE,
)


def run_preflight(config):
    with tempfile.TemporaryDirectory() as directory:
        config_path = Path(directory) / 'config.json'
        config_path.write_text(json.dumps(config))
        env = {
            name: value for name, value in os.environ.items()
            if name not in ('PARITY_CONFIG', 'PARITY_SOLVER_ROOT',
                            'PARITY_SOLVER_UPSTREAM', 'PARITY_SOLVER_ROLE')
        }
        env['PARITY_CONFIG'] = str(config_path)
        return subprocess.run(
            [sys.executable, '-B', str(PREFLIGHT)],
            cwd=str(ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=PREFLIGHT_TIMEOUT,
        )


class MissingReferenceContractTests(unittest.TestCase):
    def assert_declared_refusal(self, result, field_pattern):
        output = result.stdout or ''
        self.assertEqual(
            result.returncode, REFUSAL_EXIT,
            f'pre-flight exited {result.returncode}, expected {REFUSAL_EXIT}:\n{output}')
        self.assertNotIn('Traceback (most recent call last)', output, output)
        self.assertNotIn('ModuleNotFoundError', output, output)

        missing = MISSING_CONTRACT.search(output)
        self.assertIsNotNone(
            missing, f'message does not name the missing reference contract module:\n{output}')
        field = field_pattern.search(output)
        self.assertIsNotNone(
            field, f'message does not name the config field that required the contract:\n{output}')
        no_simulation = NO_SIMULATION.search(output)
        self.assertIsNotNone(
            no_simulation, f'message does not say no simulation was run:\n{output}')

        self.assertLess(
            missing.start(), field.start(),
            f'the three facts are not in the pinned order (missing, field, no simulation):\n{output}')
        self.assertLess(
            field.start(), no_simulation.start(),
            f'the three facts are not in the pinned order (missing, field, no simulation):\n{output}')
        self.assertIn('reference_contract', output.lower())
        return output

    def test_alignment_config_without_contract_exits_65(self):
        result = run_preflight({'task': 'leg 28 alignment probe', 'kind': 'alignment'})
        self.assert_declared_refusal(result, ALIGNMENT_FIELD)

    def test_reference_field_without_contract_exits_65(self):
        result = run_preflight({
            'task': 'leg 28 reference probe',
            'kind': 'audit',
            'reference': 'refs/absent-leg28-reference.json',
        })
        self.assert_declared_refusal(result, REFERENCE_FIELD)

    def test_config_without_reference_need_is_not_refused(self):
        result = run_preflight({'task': 'leg 28 negative control', 'kind': 'audit'})
        output = result.stdout or ''
        self.assertEqual(
            result.returncode, 0,
            f'the negative control was refused, so a fix that refuses everything would pass:\n{output}')
        self.assertIn('preflight:', output)
        self.assertIsNone(
            MISSING_CONTRACT.search(output),
            f'the negative control reported a missing reference contract:\n{output}')
        self.assertIsNone(
            INVALID_REFERENCE.search(output),
            f'the negative control reported an invalid reference:\n{output}')

    def test_missing_contract_refusal_differs_from_invalid_reference_refusal(self):
        result = run_preflight({'task': 'leg 28 distinguishability probe', 'kind': 'alignment'})
        output = self.assert_declared_refusal(result, ALIGNMENT_FIELD)
        self.assertIsNone(
            INVALID_REFERENCE.search(output),
            f'the missing-contract refusal uses the invalid-reference vocabulary, so a reader '
            f'cannot tell which one happened:\n{output}')
        self.assertNotIn('unavailable/invalid', output.lower())


if __name__ == '__main__':
    unittest.main()
