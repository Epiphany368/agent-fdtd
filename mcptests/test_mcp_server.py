"""LEG 40 acceptance tests: the six MCP tools over an in-process stdio client.

No network and no GPU. A1 exercises every tool; A2 and A3 are the two negative
controls; A4 starts the server with no solver link. A5 is the existing suite and
is run separately.
"""
import json
from pathlib import Path
import re
import unittest

try:
    from support import ROOT, call, isolated_project, open_session
except ImportError:
    from mcptests.support import ROOT, call, isolated_project, open_session


TOOL_NAMES = {
    "list_configs",
    "describe_run",
    "validate_config",
    "submit_run",
    "read_ledger",
    "explain_exit",
}


class ToolSurfaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_six_tools_over_stdio(self):
        async with open_session() as (session, _errlog):
            listed = await session.list_tools()
            names = {tool.name for tool in listed.tools}
            self.assertTrue(TOOL_NAMES.issubset(names))

            configs = await call(session, "list_configs")
            self.assertTrue(configs["ok"])
            self.assertGreater(len(configs["configs"]), 10)
            for pair in configs["configs"]:
                self.assertTrue((ROOT / pair["config"]).is_file(), pair)
                self.assertTrue((ROOT / pair["driver"]).is_file(), pair)
                self.assertTrue(pair["purpose"])

            described = await call(
                session, "describe_run", {"config": "configs/leg24-graded-check.json"}
            )
            self.assertTrue(described["ok"], described)
            self.assertEqual(described["driver"], "src/leg24_graded_modal.py")
            self.assertEqual(described["solver"]["role"], "fork")
            self.assertGreater(len(described["criteria"]), 0, described)
            self.assertRegex(
                described["criteria_source"] or "",
                r"^(?:plan\.md:\d+|docs/leg-criteria\.json#leg-24)$",
            )

            selftest = await call(
                session, "validate_config", {"config": "configs/selftest-spine.json"}
            )
            self.assertTrue(selftest["ok"], selftest)
            self.assertEqual(selftest["validator"], "checks/pre_environment.py")

            invalid = await call(
                session,
                "validate_config",
                {"config": "mcptests/fixtures/invalid-config.json"},
            )
            self.assertFalse(invalid["ok"], invalid)
            self.assertTrue(invalid["errors"], invalid)

            dry = await call(
                session,
                "submit_run",
                {"config": "configs/selftest-spine.json", "dry_run": True},
            )
            self.assertTrue(dry["dry_run"], dry)
            self.assertFalse(dry["executed"], dry)
            self.assertTrue(dry["argv"][0].endswith("tools/run.sh"), dry)
            self.assertIn("tools/run.sh", dry["command"])

            ledger = await call(session, "read_ledger", {"limit": 3})
            self.assertTrue(ledger["ok"], ledger)
            self.assertEqual(ledger["count"], 3)
            last_lines = [
                json.loads(line)
                for line in (ROOT / "runs/ledger.jsonl").read_text().splitlines()[-3:]
            ]
            self.assertEqual(ledger["rows"], last_lines)

            finding = await call(session, "explain_exit", {"code": 57})
            self.assertTrue(finding["known"], finding)
            self.assertFalse(finding["ok"], finding)
            self.assertEqual(finding["kind"], "finding")

            unknown = await call(session, "explain_exit", {"code": 991})
            self.assertFalse(unknown["known"], unknown)
            self.assertEqual(unknown["kind"], "unknown")


class StructuredRefusalTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_config_refuses_and_starts_no_process(self):
        with isolated_project(with_solver=False) as root:
            async with open_session(root=root) as (session, _errlog):
                result = await call(
                    session,
                    "submit_run",
                    {"config": "configs/invalid-config.json", "dry_run": False},
                )
            self.assertFalse(result["ok"], result)
            self.assertEqual(result["kind"], "invalid_config", result)
            self.assertFalse(result["executed"], result)
            self.assertIsNone(result["exit_status"], result)
            self.assertTrue(result["errors"], result)
            self.assertIn("task", " ".join(result["errors"]).lower(), result)
            self.assertFalse((root / "runs").exists(), "a validation refusal started a run")


class Exit57Tests(unittest.IsolatedAsyncioTestCase):
    async def test_driver_exit_57_is_a_finding_not_success(self):
        with isolated_project(with_solver=True) as root:
            async with open_session(root=root) as (session, _errlog):
                result = await call(
                    session,
                    "submit_run",
                    {"config": "configs/exit57.json", "dry_run": False},
                )
            self.assertEqual(result["exit_status"], 57, result)
            self.assertFalse(result["ok"], result)
            self.assertEqual(result["kind"], "finding", result)
            self.assertTrue(result["executed"], result)
            self.assertTrue(result["run_id"], result)
            self.assertEqual(result["ledger_row"]["exit_code"], 57, result)
            self.assertIn("finding", result["stderr_tail"].lower(), result)

            ledger = (root / "runs" / "ledger.jsonl").read_text().splitlines()
            self.assertEqual(len(ledger), 1)
            self.assertEqual(json.loads(ledger[0])["exit_code"], 57)


class NoSolverTests(unittest.IsolatedAsyncioTestCase):
    async def test_server_starts_without_solver_and_refuses_cleanly(self):
        with isolated_project(with_solver=False) as root:
            self.assertFalse((root / "src" / "torchfdtd").exists())
            async with open_session(root=root) as (session, errlog):
                configs = await call(session, "list_configs")
                self.assertTrue(configs["ok"], configs)

                validated = await call(
                    session, "validate_config", {"config": "configs/exit57.json"}
                )
                self.assertTrue(validated["ok"], validated)

                result = await call(
                    session,
                    "submit_run",
                    {"config": "configs/exit57.json", "dry_run": False},
                )
            self.assertFalse(result["ok"], result)
            self.assertEqual(result["kind"], "solver_unavailable", result)
            self.assertFalse(result["executed"], result)
            self.assertIn("bootstrap_solver", " ".join(result["errors"]))
            log = errlog.getvalue()
            self.assertNotIn("Traceback", log, log)
            self.assertFalse((root / "runs").exists(), "a solver refusal started a run")


class DependencyTests(unittest.TestCase):
    def test_no_model_sdk_dependency(self):
        forbidden = re.compile(
            r"^\s*(?:from|import)\s+(openai|anthropic|litellm|google\.generativeai|cohere)\b",
            re.MULTILINE,
        )
        for path in sorted((ROOT / "mcp_server").glob("*.py")):
            source = path.read_text(encoding="utf-8")
            self.assertIsNone(forbidden.search(source), f"model SDK dependency in {path}")

    def test_solver_is_not_imported_at_module_scope(self):
        for path in sorted((ROOT / "mcp_server").glob("*.py")):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn(
                "import torchfdtd",
                source,
                f"the server must not import the solver at module scope: {path}",
            )


if __name__ == "__main__":
    unittest.main()
