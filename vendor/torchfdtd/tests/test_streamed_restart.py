"""Durable restart journal: interrupted streamed runs resume and reproduce results.

Interruptions are injected after a chosen number of forward or transpose block
operations, and one case kills a real child process. Resumed runs must report
the block they continued from, reproduce the uninterrupted signals and gradient,
reject journals written for other inputs or another signal adjoint, and remove
their state records on completion.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

import torchfdtd
from torchfdtd import Project, StreamedSimulation, StreamedAdjointOptions
from torchfdtd.spacetime import SlabBlockOperator
from test_differentiable import project, gpu


def options(tmp_path, storage='host', device='cpu', **overrides):
    values = dict(device=device, slab_width=4, temporal_depth=3, checkpoints=1, local_checkpoints=1,
                  restart_directory=tmp_path / 'journal')
    if storage == 'disk':
        values.update(state_storage='disk', state_directory=tmp_path / 'scratch', disk_budget_bytes=64 * 1024 ** 2)
    values.update(overrides)
    return StreamedAdjointOptions(**values)


def scene():
    return project(steps=13)  # five blocks of depth three, the last one partial


class Interrupted(RuntimeError):
    pass


def interrupt(monkeypatch, method, allowed):
    original = getattr(SlabBlockOperator, method)
    calls = [0]

    def wrapper(self, *args, **kwargs):
        calls[0] += 1
        if calls[0] > allowed:
            raise Interrupted(f'interrupted {method} after {allowed} calls')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(SlabBlockOperator, method, wrapper)
    return calls


def reference(p, eps, storage, device, tmp_path):
    settings = options(tmp_path / 'reference', storage, device, restart_directory=None)
    result = StreamedSimulation(p, settings)(eps)
    gradient, = torch.autograd.grad(result.signals.square().sum(), eps)
    return result.signals.detach().clone(), gradient


def _same(actual, expected, device):
    if device == 'cuda':
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=0)
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('storage', ['host', 'disk'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('transposes', [1, 3])
def test_backward_resumes_after_interrupt(tmp_path, monkeypatch, storage, device, transposes):
    if device == 'cuda':
        gpu()
    p = scene()
    eps = torch.full(p.region.shape, 1.7, dtype=torch.float64, requires_grad=True)
    signals, gradient = reference(p, eps, storage, device, tmp_path)
    settings = options(tmp_path, storage, device)
    result = StreamedSimulation(p, settings)(eps)
    _same(result.signals.detach(), signals, device)
    history = result.report['restart_signal_history_bytes']
    assert history == signals.numel() * signals.element_size()
    state, parameter = result.report['state_bytes'], eps.numel() * eps.element_size()
    assert result.report['restart_reservation_bytes'] >= max(2 * (state + history), history + 2 * (state + parameter))
    calls = interrupt(monkeypatch, 'transpose', transposes)
    with pytest.raises(Interrupted):
        torch.autograd.grad(result.signals.square().sum(), eps)
    monkeypatch.undo()
    assert calls[0] == transposes + 1
    latest = json.loads((tmp_path / 'journal' / 'latest-backward.json').read_text())
    assert latest['kind'] == 'backward' and latest['block'] == 5 - transposes
    resumed = StreamedSimulation(p, settings)(eps)
    assert resumed.report['forward_resumed_from_block'] == 'complete'
    _same(resumed.signals.detach(), signals, device)
    later, = torch.autograd.grad(resumed.signals.square().sum(), eps)
    assert resumed.report['backward_resumed_from_block'] == 5 - transposes
    _same(later, gradient, device)
    journal = tmp_path / 'journal'
    assert (journal / 'complete.json').exists() and not list(journal.glob('latest-*.json'))
    assert not [d for d in journal.iterdir() if d.is_dir()]
    if storage == 'disk':
        assert not list((tmp_path / 'scratch').iterdir())


@pytest.mark.parametrize('storage', ['host', 'disk'])
@pytest.mark.parametrize('forwards', [1, 3])
def test_forward_resumes_after_interrupt(tmp_path, monkeypatch, storage, forwards):
    p = scene()
    eps = torch.full(p.region.shape, 1.7, dtype=torch.float64, requires_grad=True)
    signals, gradient = reference(p, eps, storage, 'cpu', tmp_path)
    settings = options(tmp_path, storage)
    interrupt(monkeypatch, 'forward', forwards)
    with pytest.raises(Interrupted):
        StreamedSimulation(p, settings)(eps)
    monkeypatch.undo()
    latest = json.loads((tmp_path / 'journal' / 'latest-forward.json').read_text())
    assert latest['kind'] == 'forward' and latest['block'] == forwards
    resumed = StreamedSimulation(p, settings)(eps)
    assert resumed.report['forward_resumed_from_block'] == forwards
    torch.testing.assert_close(resumed.signals.detach(), signals, rtol=0, atol=0)
    later, = torch.autograd.grad(resumed.signals.square().sum(), eps)
    assert 'backward_resumed_from_block' not in resumed.report
    torch.testing.assert_close(later, gradient, rtol=0, atol=0)
    assert (tmp_path / 'journal' / 'complete.json').exists()


def test_every_blocks_controls_record_count(tmp_path):
    p = scene()
    eps = torch.full(p.region.shape, 1.7, dtype=torch.float64, requires_grad=True)
    result = StreamedSimulation(p, options(tmp_path, restart_every_blocks=2))(eps)
    torch.autograd.grad(result.signals.square().sum(), eps)
    # Forward records after blocks two and four plus the completion record.
    # Backward records after blocks three and one.
    assert result.report['journal_records_written'] == dict(forward=3, backward=2)
    assert result.report['journal_seconds'] > 0


def test_contract_and_signal_adjoint_mismatch_are_rejected(tmp_path, monkeypatch):
    p = scene()
    eps = torch.full(p.region.shape, 1.7, dtype=torch.float64, requires_grad=True)
    settings = options(tmp_path)
    result = StreamedSimulation(p, settings)(eps)
    interrupt(monkeypatch, 'transpose', 2)
    with pytest.raises(Interrupted):
        torch.autograd.grad(result.signals.square().sum(), eps)
    monkeypatch.undo()
    other = torch.full(p.region.shape, 1.71, dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match='contract mismatch.*Differences: epsilon_sha256$'):
        StreamedSimulation(p, settings)(other)
    resumed = StreamedSimulation(p, settings)(eps)
    with pytest.raises(ValueError, match='different signal adjoint.*Differences: signal_bar_sha256$'):
        torch.autograd.grad(resumed.signals.abs().sum(), eps)
    # The journal is intact after the rejected attempt and the original loss resumes.
    later, = torch.autograd.grad(resumed.signals.square().sum(), eps)
    assert resumed.report['backward_resumed_from_block'] == 3
    assert torch.isfinite(later).all()


@pytest.mark.parametrize('name', ['boundaries.py', 'differentiable.py', 'waveforms.py', 'cuda_kernels.py'])
def test_changed_package_file_is_rejected_by_the_contract(tmp_path, name):
    """A dirty editable install: one file edited in place, no version or commit change."""
    from torchfdtd.streamed_restart import RestartJournal, journal_contract, runtime_sources
    package = Path(torchfdtd.__file__).resolve().parent
    copy = tmp_path / 'torchfdtd'
    shutil.copytree(package, copy, ignore=shutil.ignore_patterns('__pycache__', 'web'))
    p = scene()
    eps = torch.full(p.region.shape, 1.7, dtype=torch.float64)
    contract = journal_contract(p, eps, options(tmp_path), [0, 3, 6, 9, 12, 13])
    # The fingerprint is the content of the imported package files, and every
    # module that generates CUDA kernel source at run time is among them.
    assert contract['runtime_sha256'] == runtime_sources() == runtime_sources(copy)
    assert set(contract['runtime_sha256']) == {f.relative_to(package).as_posix() for f in package.rglob('*.py')}
    generators = {f.name for f in package.glob('*.py') if '__global__' in f.read_text(encoding='utf-8')}
    assert 'cuda_kernels.py' in generators and generators <= set(contract['runtime_sha256'])
    RestartJournal(tmp_path / 'journal', contract)
    with open(copy / name, 'a', encoding='utf-8') as handle:
        handle.write('\n# a comment appended between the crash and the resume\n')
    changed = dict(contract, runtime_sha256=runtime_sources(copy))
    with pytest.raises(ValueError, match=re.escape(f'Differences: runtime_sha256, runtime_sha256.{name}')):
        RestartJournal(tmp_path / 'journal', changed)
    # The unchanged tree still opens the same journal.
    RestartJournal(tmp_path / 'journal', contract)


@pytest.mark.parametrize('change,keys', [('epsilon', 'epsilon_sha256'), ('waveform', 'project_sha256'),
                                         ('steps', 'project_sha256, starts'), ('time_step', 'project_sha256'),
                                         ('courant', 'project_sha256')])
def test_changed_inputs_are_rejected_with_the_contract_key_named(tmp_path, monkeypatch, change, keys):
    p = scene()
    eps = torch.full(p.region.shape, 1.7, dtype=torch.float64, requires_grad=True)
    signals, gradient = reference(p, eps, 'host', 'cpu', tmp_path)
    settings = options(tmp_path)
    interrupt(monkeypatch, 'forward', 1)
    with pytest.raises(Interrupted):
        StreamedSimulation(p, settings)(eps)
    monkeypatch.undo()
    other, other_eps = p.model_copy(deep=True), eps
    if change == 'epsilon':
        other_eps = torch.full(p.region.shape, 1.71, dtype=torch.float64, requires_grad=True)
    elif change == 'waveform':
        other.sources[0].wavelength = 1.2
    elif change == 'steps':
        other.region.steps = 14
    elif change == 'time_step':
        other.region.time_step_override = .9 * p.region.time_step
    elif change == 'courant':
        other.region.courant_factor = .5
    with pytest.raises(ValueError, match='contract mismatch.*Differences: ' + re.escape(keys) + '$'):
        StreamedSimulation(other, settings)(other_eps)
    assert json.loads((tmp_path / 'journal' / 'latest-forward.json').read_text())['block'] == 1
    # Identical inputs in a fresh scene object still resume the journal.
    resumed = StreamedSimulation(p.model_copy(deep=True), settings)(eps)
    assert resumed.report['forward_resumed_from_block'] == 1
    torch.testing.assert_close(resumed.signals.detach(), signals, rtol=0, atol=0)
    later, = torch.autograd.grad(resumed.signals.square().sum(), eps)
    torch.testing.assert_close(later, gradient, rtol=0, atol=0)


def test_stamped_or_replaced_project_resumes_the_journal(tmp_path, monkeypatch):
    """A workbench save between the interruption and the resume changes labels, placement, revision and content hash only."""
    p = scene()
    eps = torch.full(p.region.shape, 1.7, dtype=torch.float64, requires_grad=True)
    signals, gradient = reference(p, eps, 'host', 'cpu', tmp_path)
    settings = options(tmp_path)
    interrupt(monkeypatch, 'forward', 1)
    with pytest.raises(Interrupted):
        StreamedSimulation(p, settings)(eps)
    monkeypatch.undo()
    saved = p.model_copy(deep=True)
    saved.name = 'renamed'
    saved.region.backend = 'auto'; saved.region.execution_mode = 'streamed_host'; saved.region.snapshot_interval = 7
    saved.region.tiling.size_um = 12; saved.region.field = 'Hy'; saved.region.slice_position = .1
    for item in saved.structures + saved.sources + saved.monitors:
        item.id = 'new-' + item.id; item.name = 'renamed'
    saved = Project.model_validate(saved.model_dump()).stamped(p.revision + 1)
    assert saved.content_matches() and saved.revision == p.revision + 1 and saved.content_sha256 != p.content_sha256
    resumed = StreamedSimulation(saved, settings)(eps)
    assert resumed.report['forward_resumed_from_block'] == 1
    torch.testing.assert_close(resumed.signals.detach(), signals, rtol=0, atol=0)
    later, = torch.autograd.grad(resumed.signals.square().sum(), eps)
    torch.testing.assert_close(later, gradient, rtol=0, atol=0)


def test_journal_and_banks_share_a_volume_by_identity_not_by_path(tmp_path, monkeypatch):
    from torchfdtd.streamed import _reservation, _volume
    scratch, journal = tmp_path / 'scratch', tmp_path / 'journal'
    # Directories that do not exist yet resolve to the volume of their nearest ancestor.
    assert _volume(journal) == _volume(scratch) == _volume(tmp_path) == _volume(tmp_path / 'a' / 'b' / 'c')
    if os.name == 'nt':
        drive = os.path.splitdrive(os.path.realpath(tmp_path))[0]
        assert _volume(drive.lower() + '/x') == _volume(drive.upper() + '/y') == drive.lower()
        assert _volume('Q:/journal') != _volume(tmp_path)
    elif _volume('/dev') != _volume(tmp_path):
        assert _volume('/dev/absent/journal') == _volume('/dev')
    p = scene()
    eps = torch.full(p.region.shape, 1.7, dtype=torch.float64)
    settings = options(tmp_path, 'disk')
    monkeypatch.setattr('torchfdtd.streamed.host_memory', lambda: dict(available_bytes=1024 ** 4))
    monkeypatch.setattr('torchfdtd.streamed.cuda_budget_limit', lambda device, gpu, budget: budget)
    monkeypatch.setattr('torchfdtd.state_store.disk_free', lambda _: 1024 ** 4)
    report = _reservation(p, eps, settings)
    banks, restart = report['disk_reservation_bytes'], report['restart_reservation_bytes']
    assert banks > 0 and restart > 0
    # One volume holds both: the journal check charges the sum.
    monkeypatch.setattr('torchfdtd.state_store.disk_free', lambda _: banks + restart)
    _reservation(p, eps, settings)
    monkeypatch.setattr('torchfdtd.state_store.disk_free', lambda _: banks + restart - 1)
    with pytest.raises(ValueError, match='Restart journal reservation'):
        _reservation(p, eps, settings)
    # Distinct volumes under one path prefix: only the journal is charged there.
    monkeypatch.setattr('torchfdtd.streamed._volume', lambda directory: str(directory))
    _reservation(p, eps, settings)
    monkeypatch.setattr('torchfdtd.state_store.disk_free', lambda _: restart - 1)
    with pytest.raises(ValueError, match='Restart journal reservation'):
        _reservation(p, eps, settings)


def test_reservation_covers_the_signal_history_of_coexisting_records(tmp_path, monkeypatch):
    from torchfdtd import Monitor
    from torchfdtd.streamed import _journal_bytes
    p = scene()
    p.region.steps = 250
    p.monitors = [Monitor(component='Ez', center=(.1 * i - .35, .1 * j - .15, 0)) for i in range(8) for j in range(4)]
    eps = torch.full(p.region.shape, 1.7, dtype=torch.float64, requires_grad=True)
    settings = options(tmp_path, temporal_depth=25, checkpoints=3, restart_every_blocks=2)
    # The journal is largest while a record is being written beside the one it
    # replaces: measure every byte under the journal (the `.tmp` directory and
    # its JSON included) when a record's meta.json is written and again at
    # every removal, after the rename and the pointer update.
    peaks = []
    from torchfdtd import streamed_restart
    original_rmtree, original_json = shutil.rmtree, streamed_restart._atomic_json

    def everything(root):
        return sum(f.stat().st_size for f in root.rglob('*') if f.is_file())

    def measured_rmtree(path, *args, **kwargs):
        peaks.append(everything(Path(path).parent))
        return original_rmtree(path, *args, **kwargs)

    def measured_json(path, payload):
        original_json(path, payload)
        if path.name == 'meta.json' and path.parent.name.endswith('.tmp'):
            peaks.append(everything(path.parent.parent))
    monkeypatch.setattr('torchfdtd.streamed_restart.shutil.rmtree', measured_rmtree)
    monkeypatch.setattr('torchfdtd.streamed_restart._atomic_json', measured_json)
    result = StreamedSimulation(p, settings)(eps)
    torch.autograd.grad(result.signals.square().sum(), eps)
    report = result.report
    assert report['journal_records_written'] == dict(forward=5, backward=4)
    assert len(peaks) >= 18
    state, history = report['state_bytes'], report['restart_signal_history_bytes']
    gradient = eps.numel() * eps.element_size()
    assert history == 250 * 32 * 8 == result.signals.numel() * result.signals.element_size()
    assert history > state
    # Two forward records with the history exceed the former two-state charge.
    assert max(peaks) > 2 * (state + gradient)
    arrays = max(2 * (state + history), history + 2 * (state + gradient))
    assert report['restart_reservation_bytes'] == arrays + report['restart_journal_metadata_bytes']
    # The array part is realised exactly; the metadata bound covers the JSON.
    assert arrays <= max(peaks) <= report['restart_reservation_bytes']
    assert max(peaks) - arrays <= report['restart_journal_metadata_bytes']


def test_stale_temporary_records_are_removed_only_inside_the_journal(tmp_path, monkeypatch):
    from torchfdtd.streamed import _journal_bytes
    from torchfdtd.streamed_restart import RestartJournal, journal_contract
    p = scene()
    eps = torch.full(p.region.shape, 1.7, dtype=torch.float64, requires_grad=True)
    settings = options(tmp_path)
    result = StreamedSimulation(p, settings)(eps)
    interrupt(monkeypatch, 'transpose', 2)
    with pytest.raises(Interrupted):
        torch.autograd.grad(result.signals.square().sum(), eps)
    monkeypatch.undo()
    journal = tmp_path / 'journal'
    # What a crash mid-write leaves behind: a `<name>.tmp` record directory
    # without a pointer and a `.tmp` file of an interrupted JSON replacement.
    stale = journal / 'backward-2.tmp'
    stale.mkdir()
    (stale / 'adjoint-0.bin').write_bytes(bytes(8192))
    (journal / 'latest-backward.json.tmp').write_bytes(b'{')
    outside = tmp_path / 'elsewhere.tmp'
    outside.mkdir()
    (outside / 'keep.bin').write_bytes(bytes(16))
    published = _journal_bytes(journal)
    assert published == sum(f.stat().st_size for f in journal.rglob('*') if f.is_file()) - 8192 - 1
    resumed = StreamedSimulation(p, settings)(eps)
    assert not stale.exists() and not (journal / 'latest-backward.json.tmp').exists()
    assert (journal / 'backward-3').is_dir() and outside.is_dir() and (outside / 'keep.bin').exists()
    later, = torch.autograd.grad(resumed.signals.square().sum(), eps)
    assert resumed.report['backward_resumed_from_block'] == 3 and torch.isfinite(later).all()
    # A pointer that names a record outside the journal is refused, not followed.
    fresh = tmp_path / 'fresh'
    contract = journal_contract(p, eps.detach(), settings, [0, 3, 6, 9, 12, 13])
    RestartJournal(fresh, contract)
    (fresh / 'latest-forward.json').write_text(json.dumps(dict(directory='../elsewhere.tmp', kind='forward', block=1)))
    with pytest.raises(ValueError, match='outside this journal directory'):
        RestartJournal(fresh, contract).load_forward(lambda: None, result.signals.detach())
    assert (outside / 'keep.bin').exists()


def test_restart_journal_space_is_checked(tmp_path, monkeypatch):
    p = scene()
    eps = torch.full(p.region.shape, 1.7, dtype=torch.float64, requires_grad=True)
    monkeypatch.setattr('torchfdtd.state_store.disk_free', lambda _: 1024)
    with pytest.raises(ValueError, match='Restart journal reservation'):
        StreamedSimulation(p, options(tmp_path))(eps)
    assert not (tmp_path / 'journal').exists()


def test_existing_journal_records_count_toward_the_reservation(tmp_path, monkeypatch):
    from torchfdtd.streamed import _reservation
    p = scene()
    eps = torch.full(p.region.shape, 1.7, dtype=torch.float64)
    settings = options(tmp_path)
    monkeypatch.setattr('torchfdtd.streamed.host_memory', lambda: dict(available_bytes=1024 ** 4))
    monkeypatch.setattr('torchfdtd.streamed.cuda_budget_limit', lambda device, gpu, budget: budget)
    fresh = _reservation(p, eps, settings)
    restart = fresh['restart_reservation_bytes']
    assert fresh['restart_journal_existing_bytes'] == 0
    # Exactly the reservation free: admitted with an empty journal.
    monkeypatch.setattr('torchfdtd.state_store.disk_free', lambda _: restart)
    _reservation(p, eps, settings)
    monkeypatch.setattr('torchfdtd.state_store.disk_free', lambda _: restart - 1)
    with pytest.raises(ValueError, match='Restart journal reservation'):
        _reservation(p, eps, settings)
    # A surviving record on disk already holds part of that space.
    record = tmp_path / 'journal' / 'backward-1'
    record.mkdir(parents=True)
    (record / 'adjoint-0.bin').write_bytes(bytes(4096))
    again = _reservation(p, eps, settings)
    assert again['restart_journal_existing_bytes'] == 4096
    monkeypatch.setattr('torchfdtd.state_store.disk_free', lambda _: restart - 4096)
    _reservation(p, eps, settings)


CHILD = '''
import sys, time, torch
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
from torchfdtd import StreamedSimulation, StreamedAdjointOptions
from torchfdtd.spacetime import SlabBlockOperator
from test_differentiable import project
original = SlabBlockOperator.transpose
def slow(self, *args, **kwargs):
    value = original(self, *args, **kwargs)
    time.sleep(0.4)
    return value
SlabBlockOperator.transpose = slow
p = project(steps=13)
eps = torch.full(p.region.shape, 1.7, dtype=torch.float64, requires_grad=True)
settings = StreamedAdjointOptions(device='cpu', slab_width=4, temporal_depth=3, checkpoints=1, local_checkpoints=1,
    state_storage='disk', state_directory=sys.argv[3], disk_budget_bytes=64*1024**2, restart_directory=sys.argv[4])
result = StreamedSimulation(p, settings)(eps)
torch.autograd.grad(result.signals.square().sum(), eps)
print('child finished without being killed', flush=True)
'''


def test_killed_process_is_resumed_by_a_new_process(tmp_path):
    p = scene()
    eps = torch.full(p.region.shape, 1.7, dtype=torch.float64, requires_grad=True)
    signals, gradient = reference(p, eps, 'disk', 'cpu', tmp_path)
    source = str(Path(torchfdtd.__file__).resolve().parents[1])
    tests = str(Path(__file__).resolve().parent)
    scratch, journal = tmp_path / 'scratch', tmp_path / 'journal'
    scratch.mkdir()
    child = subprocess.Popen([sys.executable, '-c', CHILD, source, tests, str(scratch), str(journal)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.time() + 120
    seen = None
    while time.time() < deadline and child.poll() is None:
        latest = journal / 'latest-backward.json'
        if latest.exists():
            try:
                seen = json.loads(latest.read_text())
            except (json.JSONDecodeError, OSError):
                seen = None
            if seen and seen['kind'] == 'backward' and seen['block'] <= 3:
                break
        time.sleep(0.05)
    assert child.poll() is None, child.stderr.read().decode('utf8', 'replace')[-2000:]
    child.kill()
    child.wait(timeout=60)
    assert child.returncode != 0
    # A killed process leaves its scratch banks behind; the journal survives.
    resumed = StreamedSimulation(p, options(tmp_path, 'disk'))(eps)
    assert resumed.report['forward_resumed_from_block'] == 'complete'
    later, = torch.autograd.grad(resumed.signals.square().sum(), eps)
    assert resumed.report['backward_resumed_from_block'] == seen['block']
    torch.testing.assert_close(resumed.signals.detach(), signals, rtol=0, atol=0)
    torch.testing.assert_close(later, gradient, rtol=0, atol=0)
    assert (journal / 'complete.json').exists()
