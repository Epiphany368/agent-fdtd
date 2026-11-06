"""The README "Compared with Meep" block and docs/MEEP_COMPARISON.md are the renderer's output for the committed records.

No simulation runs here; everything is read from docs/validation/meep_comparison and the two rendered documents.
"""
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / 'docs' / 'validation' / 'meep_comparison'
EXAMPLES = ('microring', 'metalens', 'metagrating')


def load_renderer():
    spec = importlib.util.spec_from_file_location('render_meep_comparison', ROOT / 'scripts' / 'render_meep_comparison.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def renderer():
    for name in ('microring_comparison', 'metalens_comparison', 'metagrating_comparison'):
        if not (RECORDS / f'{name}.json').is_file():
            pytest.skip(f'{name}.json is not committed')
    return load_renderer()


@pytest.fixture(scope='module')
def records(renderer):
    return renderer.load_records()


@pytest.fixture(scope='module')
def readme_text():
    return (ROOT / 'README.md').read_bytes().decode('utf-8')


@pytest.fixture(scope='module')
def readme_block(renderer, readme_text):
    assert readme_text.count(renderer.START) == 1 and readme_text.count(renderer.END) == 1
    start = readme_text.index(renderer.START)
    end = readme_text.index(renderer.END) + len(renderer.END)
    return readme_text[start:end]


def test_readme_block_is_the_renderer_output(renderer, records, readme_block):
    assert readme_block == renderer.render_readme_block(records)


def test_readme_block_sits_after_the_speed_table(renderer, readme_text):
    faster = readme_text.index('## How much faster')
    modes = readme_text.index('## Execution modes')
    assert faster < readme_text.index(renderer.START) < readme_text.index(renderer.END) < modes


def test_readme_update_is_idempotent(renderer, records, readme_text):
    block = renderer.render_readme_block(records)
    once = renderer.update_readme(readme_text, block)
    assert once == readme_text
    assert renderer.update_readme(once, block) == once
    stripped = readme_text[:readme_text.index(renderer.START)].rstrip('\n') + '\n' + readme_text[readme_text.index(renderer.END) + len(renderer.END):].lstrip('\n')
    assert renderer.update_readme(stripped, block) == readme_text


def test_doc_page_is_the_renderer_output(renderer, records):
    committed = (ROOT / 'docs' / 'MEEP_COMPARISON.md').read_bytes().decode('utf-8')
    assert committed == renderer.render_doc(records)


def test_readme_lists_the_doc_page_and_every_example(readme_block, readme_text):
    assert '(docs/MEEP_COMPARISON.md)' in readme_block
    for example in EXAMPLES:
        assert f'(examples/meep_comparison/{example})' in readme_block
    documentation = readme_text[readme_text.index('## Documentation'):readme_text.index('## Verification')]
    assert 'docs/MEEP_COMPARISON.md' in documentation


def test_every_example_passes_or_its_failure_is_listed(records, readme_block):
    flags = {'microring': records['microring']['comparison']['all_passed'],
             'metalens 2D': records['metalens_2d']['comparison']['all_pass'],
             'metalens 3D': records['metalens_3d']['comparison']['all_pass'],
             'metagrating': records['metagrating']['comparison']['all_criteria_passed']}
    rows = [line for line in readme_block.splitlines() if line.startswith('| [')]
    assert len(rows) == len(flags)
    for row, (name, passed) in zip(rows, flags.items()):
        if passed:
            assert 'FAIL' not in row, name
        else:
            assert 'FAIL' in row, f'{name} fails a criterion but the README row does not say so'
    if all(flags.values()):
        assert 'FAIL' not in readme_block


def leaves(node, out):
    if isinstance(node, dict):
        for value in node.values():
            leaves(value, out)
    elif isinstance(node, list):
        if len(node) > 8 and all(isinstance(v, (int, float)) for v in node):
            return  # spectra, profiles and sample grids are not scalars a README row could quote
        for value in node:
            leaves(value, out)
    elif isinstance(node, bool):
        return
    elif isinstance(node, (int, float)):
        out['numbers'].append(float(node))
    elif isinstance(node, str):
        out['strings'].append(node)


def formats_of(token):
    """Every format that could have produced this token from a record value."""
    if re.fullmatch(r'\d+', token):
        return [lambda v: str(int(round(v))) if float(v).is_integer() else None]
    if re.search(r'e[+-]?\d+$', token):
        mantissa = token.split('e')[0]
        digits = len(mantissa.split('.')[1]) if '.' in mantissa else 0
        return [lambda v, d=digits: f'{v:.{d}e}']
    digits = len(token.split('.')[1]) if '.' in token else 0
    return [lambda v, d=digits: f'{v:.{d}f}', lambda v: repr(v), lambda v: str(v)]


NUMBER = re.compile(r'(?<![A-Za-z\d.])-?\d+(?:,\d{3})*(?:\.\d+)?(?:e[+-]?\d+)?(?![A-Za-z\d])')


def test_every_number_in_the_readme_block_comes_from_the_records(records, readme_block):
    pool = {'numbers': [], 'strings': []}
    for path in sorted(RECORDS.glob('*.json')):
        leaves(json.loads(path.read_text(encoding='utf-8')), pool)
    derived = set()
    for stem in ('microring', 'metalens_2d', 'metalens_3d', 'metagrating'):
        rec = records[stem]
        t, m = rec['torchfdtd_timing']['timing'], rec['meep_timing']['timing']
        derived.add(f'{float(m["stepping_seconds"]) / float(t["stepping_seconds"]):.1f}')
        cells = rec['torchfdtd']['grid']['cells']
        if isinstance(cells, list):
            n = 1
            for c in cells:
                n *= int(c)
            pool['numbers'].append(float(n))
    criteria = [records['microring']['comparison']['criteria'].values(), records['metagrating']['comparison']['metrics'].values(),
                records['metalens_2d']['comparison']['criteria_rows'], records['metalens_3d']['comparison']['criteria_rows']]
    for group in criteria:
        group = list(group)
        derived.add(str(len(group)))
        derived.add(str(sum(1 for item in group if item.get('passed', item.get('pass')))))
    body = [line for line in readme_block.splitlines() if not line.startswith('<!--')]   # the header row counts too
    text = '\n'.join(body)
    text = re.sub(r'\]\([^)]*\)', '', text)
    text = re.sub(r'`[^`]*`', '', text)
    missing = []
    for match in NUMBER.finditer(text):
        token = match.group(0).lstrip('-')
        plain = token.replace(',', '')
        if plain in derived:
            continue
        found = any(token in s for s in pool['strings'])
        if not found:
            for fmt in formats_of(plain):
                for value in pool['numbers']:
                    try:
                        if fmt(value) == plain:
                            found = True
                            break
                    except (ValueError, OverflowError):
                        continue
                if found:
                    break
        if not found:
            missing.append(token)
    assert not missing, f'numbers in the README block absent from the records: {missing}'


def _ranks_and_precisions(records):
    ranks = {str(records[stem]['meep_timing']['environment']['mpi_processes']) for stem in records}
    precisions = {records[stem][solver]['grid']['precision'] for stem in records for solver in ('torchfdtd', 'meep')}
    return ranks, precisions


def test_rank_and_precision_tokens_in_the_readme_block_come_from_the_records(records, readme_block):
    """Every "<n> ranks" (digits or a spelled-out word) and every float<nn> in the block, header row included, is a record value."""
    ranks, precisions = _ranks_and_precisions(records)
    assert set(re.findall(r'(\w+) ranks', readme_block)) <= ranks
    assert set(re.findall(r'float\d+', readme_block)) <= precisions
    assert 'when timed' not in readme_block


def test_readme_block_follows_the_records_for_precision_and_ranks(renderer, records):
    """Rendering altered records changes the precision and rank strings; a string fixed in the renderer would not."""
    import copy
    altered = copy.deepcopy(records)
    for stem in altered:
        altered[stem]['torchfdtd']['grid']['precision'] = 'float64'
        altered[stem]['meep']['grid']['precision'] = 'float32'
        altered[stem]['meep_timing']['environment']['mpi_processes'] = 12
    block = renderer.render_readme_block(altered)
    ranks, precisions = _ranks_and_precisions(altered)
    assert ranks == {'12'} and set(re.findall(r'(\w+) ranks', block)) == {'12'}
    assert re.search(r'TorchFDTD \([^)]*float64, fused CUDA kernels\)', block) and re.search(r'Meep [\d.]+ \(CPU, float32, MPI\)', block)
    assert set(re.findall(r'float\d+', block)) <= precisions
    assert '| Meep CPU stepping (s), 12 ranks |' in block
    original = renderer.render_readme_block(records)
    assert '12 ranks' not in original and '(CPU, float32, MPI)' not in original


@pytest.mark.parametrize('example', ['metagrating', 'metalens', 'microring'])
def test_example_scripts_import_torchfdtd_from_this_checkout_or_say_where_it_came_from(example, monkeypatch):
    """The three solver scripts put this checkout first on sys.path; an import from elsewhere exits with the path named."""
    import torchfdtd
    path = ROOT / 'examples' / 'meep_comparison' / example / f'torchfdtd_{example}.py'
    spec = importlib.util.spec_from_file_location(f'torchfdtd_{example}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.torchfdtd.__file__).resolve() == Path(torchfdtd.__file__).resolve()
    assert str(Path(torchfdtd.__file__).resolve()).startswith(str(ROOT))
    if example == 'microring':
        return   # asserts at import time, from the checkout it puts first on sys.path
    module.require_checkout_import()
    elsewhere = str(Path('/opt/venv/lib/site-packages/torchfdtd/__init__.py').resolve())
    monkeypatch.setattr(module.torchfdtd, '__file__', elsewhere)
    with pytest.raises(SystemExit) as info:
        module.require_checkout_import()
    assert elsewhere in str(info.value) and str(ROOT) in str(info.value) and 'cd ' in str(info.value)
