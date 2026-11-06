"""Leg 21 assessment gate: pinned source evidence, no solver import or simulation.

The resolver reads Git objects only. The fabricated line is submitted to the
same assertion as every real citation. A nested unittest run also demonstrates
that replacing that assertion with an always-successful resolver fails the
fabrication test. Neither experiment changes the record or any source file.
"""

from functools import lru_cache
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PIN = "ad436c9"
CONFIG_PATH = ROOT / "configs/leg21-o6-assessment.json"
RECORD = json.loads(CONFIG_PATH.read_text())
OPTIONS = {"decline", "further-scoping", "register-implementation-candidate"}


@lru_cache(maxsize=1)
def solver_root():
    """Resolve the pinned solver from the checkout, not from machine-local state.

    ``src/torchfdtd`` is the link ``tools/bootstrap_solver.sh`` creates and is the
    correct source of truth for a clone. ``$SOLVER_FORK`` is only a fallback for a
    working machine whose link is missing; it is never required.
    """
    link = ROOT / "src" / "torchfdtd"
    if link.is_dir():
        return str(link)
    fallback = os.environ.get("SOLVER_FORK", "")
    if fallback and Path(fallback).is_dir():
        return fallback
    raise AssertionError(
        "cannot resolve the solver: neither src/torchfdtd (create it with "
        "tools/bootstrap_solver.sh <torchfdtd checkout>) nor $SOLVER_FORK exists"
    )


@lru_cache(maxsize=None)
def pinned_lines(path):
    """No working-tree fallback, line search, dynamic ref, or solver import."""
    result = subprocess.run(
        ["git", "-C", solver_root(), "show", f"{PIN}:{path}"],
        text=True, capture_output=True,
    )
    if result.returncode:
        raise AssertionError(f"unresolved Git object {PIN}:{path}: {result.stderr.strip()}")
    return result.stdout.splitlines()


def assert_citation(citation):
    """Raise an actual AssertionError on an invalid anchor; never relocate it."""
    if not isinstance(citation, dict) or set(citation) != {"commit", "path", "line", "must_contain"}:
        raise AssertionError("citation requires exactly commit, path, line, must_contain")
    if citation["commit"] != PIN:
        raise AssertionError(f"citation commit must be {PIN}")
    path = citation["path"]
    if (not isinstance(path, str) or not path.startswith("torchfdtd/")
            or ".." in PurePosixPath(path).parts or ":" in path or "\\" in path):
        raise AssertionError("citation path must be relative to the pinned torchfdtd tree")
    line, needle = citation["line"], citation["must_contain"]
    if type(line) is not int or line < 1:
        raise AssertionError("citation line must be a positive integer")
    if not isinstance(needle, str) or not needle.strip() or "\n" in needle:
        raise AssertionError("must_contain must be nonempty single-line text")
    lines = pinned_lines(path)
    if line > len(lines):
        raise AssertionError(f"{PIN}:{path}:{line}: line out of range ({len(lines)} lines)")
    actual = lines[line - 1]
    if needle not in actual:
        raise AssertionError(
            f"{PIN}:{path}:{line}: must_contain {needle!r} absent; actual={actual!r}"
        )


def citations_in(value):
    """Visit all citation objects, including future evidence outside invariants."""
    if isinstance(value, dict):
        if {"path", "line", "must_contain"}.intersection(value):
            yield value
        else:
            for child in value.values():
                yield from citations_in(child)
    elif isinstance(value, list):
        for child in value:
            yield from citations_in(child)


def modal_anchor():
    return {
        "commit": PIN, "path": "torchfdtd/mode_branches.py", "line": 131,
        "must_contain": "mesh_type != 'uniform'",
    }


class FabricationContract(unittest.TestCase):
    def test_fabricated_wrong_line_is_rejected(self):
        fabricated = modal_anchor()
        fabricated["line"] = 132  # Real file and string, deliberately wrong line.
        with self.assertRaisesRegex(AssertionError, "must_contain .* absent") as rejected:
            assert_citation(fabricated)
        print(f"\nEXPECTED NEGATIVE CONTROL: {rejected.exception}", flush=True)


class AssessmentContract(unittest.TestCase):
    def nonempty_text(self, value):
        self.assertIsInstance(value, str)
        self.assertTrue(value.strip())

    def text_list(self, value):
        self.assertIsInstance(value, list)
        self.assertTrue(value)
        for item in value:
            self.nonempty_text(item)

    def test_complete_record_schema(self):
        self.assertEqual(set(RECORD), {
            "schema_version", "leg", "track_item", "fork_commit", "implementation_authorized",
            "acceptance_kind", "gate", "scope", "interpretation", "invariants",
            "cross_cutting_cost", "dropped_invariants", "cost_benefit", "recommendation",
            "recommendation_rationale", "ranked_options", "next_scope", "unresolved_citations",
        })
        self.assertEqual(RECORD["schema_version"], 1)
        self.assertEqual(RECORD["leg"], 21)
        self.assertEqual(RECORD["fork_commit"], PIN)
        self.nonempty_text(RECORD["track_item"])
        self.nonempty_text(RECORD["interpretation"])
        invariants = RECORD["invariants"]
        self.assertIsInstance(invariants, list)
        self.assertGreaterEqual(len(invariants), 4)
        self.assertEqual(len({item["id"] for item in invariants}), len(invariants))
        self.assertEqual({item["category"] for item in invariants}, {
            "modal_launch", "cpml_update", "yee_material_sampling", "time_step_bound",
        })
        for item in invariants:
            with self.subTest(invariant=item["id"]):
                self.assertEqual(set(item), {
                    "id", "category", "current_invariant", "break_condition", "evidence_kind",
                    "citations", "repair_cost", "falsification_route",
                })
                for key in ("id", "current_invariant", "break_condition", "evidence_kind"):
                    self.nonempty_text(item[key])
                self.assertIsInstance(item["citations"], list)
                self.assertTrue(item["citations"])
                for citation in item["citations"]:
                    assert_citation(citation)
                repair = item["repair_cost"]
                self.assertEqual(set(repair), {"work_items", "surfaces", "validation", "estimate_basis"})
                self.text_list(repair["work_items"])
                self.text_list(repair["surfaces"])
                self.nonempty_text(repair["validation"])
                self.nonempty_text(repair["estimate_basis"])
                route = item["falsification_route"]
                self.assertEqual(set(route), {
                    "procedure", "observation_that_refutes_breakage", "limit",
                })
                for value in route.values():
                    self.nonempty_text(value)
        cross = RECORD["cross_cutting_cost"]
        self.assertEqual(set(cross), {"description", "citations", "required_work", "estimate_basis"})
        self.nonempty_text(cross["description"])
        self.nonempty_text(cross["estimate_basis"])
        self.assertIsInstance(cross["citations"], list)
        self.assertTrue(cross["citations"])
        for citation in cross["citations"]:
            assert_citation(citation)
        self.text_list(cross["required_work"])
        self.assertTrue(RECORD["dropped_invariants"])
        for dropped in RECORD["dropped_invariants"]:
            self.assertEqual(set(dropped), {"claim", "reason"})
            self.nonempty_text(dropped["claim"])
            self.nonempty_text(dropped["reason"])
        self.assertEqual(RECORD["unresolved_citations"], [])

    def test_every_citation_resolves_at_its_exact_line(self):
        citations = list(citations_in(RECORD))
        self.assertGreaterEqual(len(citations), len(RECORD["invariants"]))
        for citation in citations:
            with self.subTest(path=citation.get("path"), line=citation.get("line")):
                assert_citation(citation)
        print(f"\nResolved {len(citations)} citations against {PIN} Git objects.", flush=True)

    def test_required_modal_anchor_is_inside_an_invariant(self):
        self.assertIn(modal_anchor(), [c for item in RECORD["invariants"] for c in item["citations"]])

    def test_negative_control_kills_always_successful_resolver(self):
        # Publish the check on itself: an always-successful resolver must cause
        # the real fabrication test's assertRaises to fail, not just return False.
        result = unittest.TestResult()
        with patch.object(sys.modules[__name__], "assert_citation", return_value=None):
            FabricationContract("test_fabricated_wrong_line_is_rejected").run(result)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.failures), 1)
        self.assertIn("AssertionError not raised", result.failures[0][1])
        print("\nEXPECTED MUTATION FAILURE: " + result.failures[0][1].splitlines()[-1], flush=True)

    def test_resolver_rejects_empty_malformed_and_out_of_range_anchors(self):
        for malformed in (
            "torchfdtd/models.py:9999 must_contain nonsense",
            {"commit": PIN, "file": "torchfdtd/nope.py", "lineno": 1, "text": "x"},
        ):
            with self.subTest(malformed=malformed), self.assertRaisesRegex(
                AssertionError, "citation requires exactly commit, path, line, must_contain"
            ):
                assert_citation(malformed)
        for changed in (
            {"must_contain": ""}, {"line": 0}, {"line": True}, {"line": 1000000000},
            {"commit": "HEAD"}, {"path": "../torchfdtd/mode_branches.py"},
        ):
            citation = {**modal_anchor(), **changed}
            with self.subTest(changed=changed), self.assertRaises(AssertionError):
                assert_citation(citation)

    def test_benefit_provenance_and_conditional_arithmetic(self):
        benefit = RECORD["cost_benefit"]
        self.assertEqual(set(benefit), {
            "headline_claim", "geometric_arithmetic", "conditional_runtime_model",
            "benefit", "cost", "unknowns",
        })
        headline = benefit["headline_claim"]
        self.assertEqual(headline["text"], "<= 8x at 2x coarser spacing")
        self.nonempty_text(headline["origin"])
        self.nonempty_text(headline["interpretation"])
        if headline["provenance"] != "unverified":
            # A future sourced version must provide resolvable pinned evidence.
            self.assertEqual(headline["provenance"], "sourced")
            self.assertTrue(headline["citations"])
            for citation in headline["citations"]:
                assert_citation(citation)
        arithmetic = benefit["geometric_arithmetic"]
        self.assertEqual(arithmetic["provenance"], "arithmetic only")
        self.assertIs(arithmetic["equal_accuracy_established"], False)
        self.assertEqual(arithmetic["coarsening_per_active_axis"], 2)
        for dimension in (2, 3):
            self.assertEqual(arithmetic[f"cell_reduction_{dimension}d"], 2 ** dimension)
        self.nonempty_text(arithmetic["derivation"])
        model = benefit["conditional_runtime_model"]
        self.assertEqual(set(model), {
            "symbols", "time_step_ratio_at_twice_spacing", "ideal_runtime_speedup",
            "ideal_interior_memory_reduction", "provenance",
        })
        self.assertEqual(set(model["symbols"]), {"d", "q", "w", "m"})
        for value in model["symbols"].values():
            self.nonempty_text(value)
        for key, value in model.items():
            if key != "symbols":
                self.nonempty_text(value)
        self.nonempty_text(benefit["benefit"])
        self.nonempty_text(benefit["cost"])
        self.text_list(benefit["unknowns"])

    def test_single_recommendation_and_bounded_followup(self):
        self.assertIsInstance(RECORD["recommendation"], str)
        self.assertIn(RECORD["recommendation"], OPTIONS)
        self.nonempty_text(RECORD["recommendation_rationale"])
        options = RECORD["ranked_options"]
        self.assertEqual([item["rank"] for item in options], [1, 2, 3])
        self.assertEqual({item["option"] for item in options}, OPTIONS)
        self.assertEqual(options[0]["option"], RECORD["recommendation"])
        for item in options:
            self.nonempty_text(item["tradeoff"])
        following = RECORD["next_scope"]
        self.assertEqual(set(following), {"authorized_by_this_record", "deliverables", "decision_rule"})
        self.assertIs(following["authorized_by_this_record"], False)
        self.text_list(following["deliverables"])
        self.nonempty_text(following["decision_rule"])

    def test_no_implementation_authorization_run_or_expanded_write_set(self):
        self.assertIs(RECORD["implementation_authorized"], False)
        self.assertEqual(RECORD["acceptance_kind"], "new")
        self.assertEqual(RECORD["gate"], "docs only, no run")
        scope = RECORD["scope"]
        self.assertEqual(set(scope), {
            "fork_access", "simulation_run", "solver_import", "gpu_tokens", "ledger_rows", "write_set",
        })
        self.nonempty_text(scope["fork_access"])
        self.assertIs(scope["simulation_run"], False)
        self.assertIs(scope["solver_import"], False)
        self.assertEqual(scope["gpu_tokens"], 0)
        self.assertEqual(scope["ledger_rows"], 0)
        self.assertEqual(scope["write_set"], [
            "configs/leg21-o6-assessment.json", "checks/test_leg21_o6_assessment.py",
            "docs/leg21-2026-10-19.md",
        ])


if __name__ == "__main__":
    unittest.main()
