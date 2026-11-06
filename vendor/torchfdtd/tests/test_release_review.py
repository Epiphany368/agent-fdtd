"""docs/RELEASE_REVIEW.md keeps the open contract questions and the owner's decision on them, and claims no clearance (G9-03)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'docs' / 'RELEASE_REVIEW.md'
README = ROOT / 'README.md'
DECISION = '## Owner decision, 24 September 2026'
# The review records questions and decisions; it must never state a legal conclusion in the affirmative.
CLEARANCE_CLAIMS = ('is legally cleared', 'has been cleared', 'no legal risk', 'no infringement', 'guarantees that',
                    'is permitted by ansys')


def flat(text):
    return ' '.join(text.split())


def decision_section():
    text = REVIEW.read_text(encoding='utf-8')
    assert text.count(DECISION) == 1, 'the owner decision of 24 September 2026 must be recorded once'
    return text.split(DECISION, 1)[1].split('\n## ', 1)[0]


def test_open_contract_question_and_gates_stay_listed():
    text = REVIEW.read_text(encoding='utf-8')
    assert '## Contract issue that prevents clearance' in text and '## Outstanding release gates' in text
    assert 'remains open as a contractual question' in flat(text)


def test_owner_decision_is_recorded_with_its_limits():
    section = flat(decision_section())
    for phrase in ('interoperability features stay in the package', 'The README no longer describes them',
                   "owner's acceptance of the remaining risk", 'does not resolve the contractual question',
                   'no vendor has been contacted', 'nothing in this record is a legal opinion or a clearance'):
        assert phrase in section, phrase
    modules = re.findall(r'`(torchfdtd/[\w/]+\.py)`', section)
    assert modules, 'the decision names the modules it keeps'
    for module in modules:
        assert (ROOT / module).is_file(), f'{module} named by the decision is not in the package'


def test_review_makes_no_clearance_claim():
    lower = flat(REVIEW.read_text(encoding='utf-8')).lower()
    for phrase in CLEARANCE_CLAIMS:
        assert phrase not in lower, phrase


def test_readme_does_not_describe_the_interoperability_features_the_package_keeps():
    readme = README.read_text(encoding='utf-8')
    assert not re.search(r'Lumerical|\bFSP\b|\.fsp\b', readme), 'the README describes the interoperability features'
    assert 'not affiliated with Ansys' in readme
    for module in ('fsp.py', 'fsp_binary.py', 'fsp_native.py'):
        assert (ROOT / 'torchfdtd' / module).is_file(), module
