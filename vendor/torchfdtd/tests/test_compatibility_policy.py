"""G9-04: docs/COMPATIBILITY.md names every public export and states the format
version constants that the code carries; the changelog and the bug template exist."""
import re
import subprocess
import tempfile
import typing
from pathlib import Path

import numpy as np
import torch

import torchfdtd
from torchfdtd import Project, StreamedAdjointOptions
from torchfdtd.models import demo_project
from torchfdtd.mode_network_project import ModeNetworkConfig
from torchfdtd.solver import Simulation
from torchfdtd.streamed_restart import MARKER, journal_contract

ROOT = Path(__file__).resolve().parents[1]
COMPATIBILITY = (ROOT / 'docs' / 'COMPATIBILITY.md').read_text(encoding='utf-8')
CHANGELOG = (ROOT / 'docs' / 'CHANGELOG.md').read_text(encoding='utf-8')
TEMPLATE = (ROOT / '.github' / 'ISSUE_TEMPLATE' / 'bug_report.md').read_text(encoding='utf-8')


def section(text, heading):
    start = text.index(f'\n## {heading}\n')
    end = text.find('\n## ', start + 1)
    return text[start:end if end > 0 else None]


def table_rows(text):
    rows = [line for line in text.splitlines() if line.startswith('| ') and not line.startswith('| ---')]
    return [[cell.strip() for cell in row.strip('|').split(' | ')] for row in rows[1:]]


def test_every_public_name_is_documented_and_no_stale_name_remains():
    api = section(COMPATIBILITY, 'Public API surface')
    documented = set()
    for _, module, names in table_rows(api):
        assert module.startswith('`torchfdtd.') and module.endswith('`'), module
        documented.update(re.findall(r'`([A-Za-z_][A-Za-z0-9_]*)`', names))
    exported = set(torchfdtd.__all__)
    assert exported - documented == set(), sorted(exported - documented)
    assert documented - exported == set(), sorted(documented - exported)
    assert f'{len(exported)} names at this version' in api
    for name in exported:
        assert callable(getattr(torchfdtd, name)) or isinstance(getattr(torchfdtd, name), type), name


def test_format_version_constants_match_the_code():
    rows = {row[0]: row for row in table_rows(section(COMPATIBILITY, 'Persisted formats and their readers'))}
    field = Project.model_fields['schema_version']
    assert rows['Project JSON'][2] == f'`{field.default}`'
    assert typing.get_args(field.annotation) == (field.default,)
    assert rows['Streamed restart journal'][2] == f'`{MARKER}`'
    contract = journal_contract(demo_project(), torch.ones(2, 2, 2), StreamedAdjointOptions(), (0,))
    assert set(re.findall(r'`([a-z_0-9]+)`', rows['Streamed restart journal'][4])) >= set(contract)
    mode = ModeNetworkConfig.model_fields['version']
    assert rows['Mode-network configuration'][2] == f'`{mode.default}`'
    project = demo_project()
    project.region.backend = 'cpu'
    project.region.steps = 20
    with tempfile.TemporaryDirectory(prefix='compat-') as directory:
        path = Path(directory) / 'result.npz'
        Simulation(project).run().save(path)
        with np.load(path, allow_pickle=False) as archive:
            members = set(archive.files)
    documented = set(re.findall(r'`([A-Za-z_<>0-9]+)`', rows['Result NPZ'][2]))
    fixed = {name for name in documented if '<' not in name}
    assert fixed - {'endpoint_<name>'} <= members | {'project'}, sorted(fixed - members)
    patterns = [re.compile('^' + re.escape(name).replace('<k>', r'\d+').replace('<array>', r'[a-z_]+').replace('<name>', r'[A-Za-z_]+') + '$')
                for name in documented if '<' in name]
    unexplained = [m for m in members if m not in fixed and not any(p.match(m) for p in patterns)]
    assert unexplained == [], unexplained


def test_version_strings_agree_across_pyproject_server_and_document():
    version = re.search(r'^version = "([^"]+)"', (ROOT / 'pyproject.toml').read_text(encoding='utf-8'), re.M).group(1)
    server = (ROOT / 'torchfdtd' / 'server.py').read_text(encoding='utf-8')
    assert server.count(f"'{version}'") == 2, 'FastAPI title version and /api/health version'
    assert f'Version of record: **{version}**' in COMPATIBILITY
    assert f'\n## {version} ' in CHANGELOG


def test_changelog_cites_real_commits_and_carries_the_result_impact_section():
    assert '### Results change' in CHANGELOG and '### Security' in CHANGELOG
    cited = set(re.findall(r'\(([0-9a-f]{7}(?:, [0-9a-f]{7})*)\)', CHANGELOG))
    hashes = {h for group in cited for h in group.split(', ')}
    assert len(hashes) > 50
    known = subprocess.run(['git', 'log', '--format=%h'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    assert hashes <= set(known), sorted(hashes - set(known))


def test_severity_scale_and_bug_template_ask_for_platform_report_and_minimal_repro():
    severity = section(COMPATIBILITY, 'Numerical bug severity')
    assert [row[0] for row in table_rows(severity)] == ['S0', 'S1', 'S2', 'S3']
    assert '## Result-impact notices and rollback' in COMPATIBILITY
    for heading in ['## Platform report', '## Minimal reproduction', '## Expected and observed', '## Proposed severity']:
        assert heading in TEMPLATE, heading
    assert 'torchfdtd hardware' in TEMPLATE and '/api/health' in TEMPLATE and 'S0' in TEMPLATE and 'S3' in TEMPLATE
    assert TEMPLATE.startswith('---\nname: Bug report\n')
