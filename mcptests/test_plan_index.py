"""LEG 41 acceptance: the published criteria index is read without plan.md.

The public export withholds ``plan.md`` and ships ``docs/leg-criteria.json``
instead.  These tests run unchanged in both trees: the working repository has
``plan.md`` and no index, the export has the index and no ``plan.md``.
"""
import json
from pathlib import Path
import tempfile
import unittest

from mcp_server.plan import criteria_for, write_criteria_index

try:
    from support import ROOT, call, open_session
except ImportError:
    from mcptests.support import ROOT, call, open_session


LEG = 24
CONFIG = "configs/leg24-graded-check.json"


class CriteriaIndexTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="leg41-index-")
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _index(self):
        """The tree's index, copied or extracted into the temp root."""
        target = self.root / "docs" / "leg-criteria.json"
        source = ROOT / "docs" / "leg-criteria.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            write_criteria_index(ROOT / "plan.md", target)
        return target

    def test_index_is_used_when_plan_md_is_absent(self):
        self._index()
        self.assertFalse((self.root / "plan.md").exists())
        result = criteria_for(self.root / CONFIG, self.root / "plan.md")
        self.assertTrue(result["criteria"], result)
        self.assertEqual(result["criteria_source"], f"docs/leg-criteria.json#leg-{LEG}")
        self.assertIsNone(result["criteria_note"])

    def test_registered_leg_has_criteria_in_this_tree(self):
        result = criteria_for(ROOT / CONFIG, ROOT / "plan.md")
        self.assertTrue(result["criteria"], result)
        self.assertIsNone(result["criteria_note"], result)

    def test_neither_source_is_a_note_not_a_traceback(self):
        result = criteria_for(self.root / CONFIG, self.root / "plan.md")
        self.assertEqual(result["criteria"], [])
        self.assertIn("not readable", result["criteria_note"])

    def test_index_carries_four_fields_for_every_leg(self):
        index = self._index()
        payload = json.loads(index.read_text(encoding="utf-8"))
        self.assertEqual(payload["format"], "leg-criteria-index v1")
        legs = payload["legs"]
        self.assertTrue(legs)
        for entry in legs:
            self.assertEqual(set(entry), {"leg", "goal", "resource", "exits"}, entry)
        numbers = [entry["leg"] for entry in legs]
        self.assertEqual(numbers, sorted(numbers))
        self.assertIn(LEG, numbers)


class DescribeRunCriteriaTests(unittest.IsolatedAsyncioTestCase):
    async def test_describe_run_reports_criteria_in_this_tree(self):
        async with open_session() as (session, _errlog):
            described = await call(session, "describe_run", {"config": CONFIG})
        self.assertTrue(described["ok"], described)
        self.assertTrue(described["criteria"], described)
        self.assertRegex(
            described["criteria_source"] or "",
            rf"^(?:plan\.md:\d+|docs/leg-criteria\.json#leg-{LEG})$",
        )


if __name__ == "__main__":
    unittest.main()
