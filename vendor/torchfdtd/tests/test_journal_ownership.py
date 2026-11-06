"""Journal ownership, integrity and durability levels (G5-09).

One writer owns a journal at a time through `owner.json`: a second writer is
refused by name while the owner lives, a lock left by a dead process is taken
over and the previous run reported as `partial`, and a lock from another host
or an unreadable lock is refused. Every record file is checked for its
checksum, dtype, shape, array count and byte count. The terminal states
completed, cancelled, failed and partial are distinct and the records of an
unfinished run stay readable. Rename-into-place uses the platform primitive
(directory fsync on POSIX, write-through move on Windows); power-loss
durability is documented as a separate, unexercised level.
"""
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import torch

import torchfdtd
from torchfdtd import StreamedSimulation
from torchfdtd import streamed_restart
from torchfdtd.spacetime import SlabBlockOperator
from torchfdtd.streamed import StreamCancelled, _StreamedExecution
from torchfdtd.streamed_restart import (RestartJournal, RestartRecordDamaged, inspect_journal, journal_contract,
                                        MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, _validate_record, _read_array)
from test_streamed_restart import options, scene, Interrupted, interrupt

STARTS = [0, 3, 6, 9, 12, 13]


def state_count(p, eps):
    """E, H and every CPML psi segment of the test scene."""
    return len(_StreamedExecution().host(p, eps.detach(), None).state())


def contract_for(tmp_path, settings=None):
    p = scene()
    eps = torch.full(p.region.shape, 1.7, dtype=torch.float64)
    return p, eps, journal_contract(p, eps, settings or options(tmp_path), STARTS)


def interrupted_forward(tmp_path, monkeypatch, blocks=2):
    p = scene()
    eps = torch.full(p.region.shape, 1.7, dtype=torch.float64, requires_grad=True)
    settings = options(tmp_path)
    interrupt(monkeypatch, 'forward', blocks)
    with pytest.raises(Interrupted):
        StreamedSimulation(p, settings)(eps)
    monkeypatch.undo()
    return p, eps, settings


# ---- ownership -----------------------------------------------------------------------
HOLDER = '''
import json, sys, time
sys.path.insert(0, sys.argv[1])
from torchfdtd.streamed_restart import RestartJournal
journal = RestartJournal(sys.argv[2], json.loads(open(sys.argv[3], encoding='utf-8').read()))
print('holding', journal.pid, flush=True)
time.sleep(120)
'''


def test_second_writer_is_refused_while_the_owner_lives_and_admitted_after_it_dies(tmp_path):
    p, eps, contract = contract_for(tmp_path)
    (tmp_path / 'contract.json').write_text(json.dumps(contract), encoding='utf-8')
    journal = tmp_path / 'journal'
    child = subprocess.Popen([sys.executable, '-c', HOLDER, str(Path(torchfdtd.__file__).resolve().parents[1]),
                              str(journal), str(tmp_path / 'contract.json')], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    line = child.stdout.readline()
    assert line.startswith('holding'), child.stderr.read()[-2000:]
    holder_pid = int(line.split()[1])
    try:
        view = inspect_journal(journal)
        assert view['state'] == 'running' and view['owner']['alive'] is True and view['owner']['pid'] == holder_pid
        with pytest.raises(ValueError, match=rf'owned by run [0-9a-f]+ \(pid {holder_pid}\), which is still running'):
            RestartJournal(journal, contract)
        assert json.loads((journal / 'owner.json').read_text())['pid'] == holder_pid
    finally:
        child.kill()
        child.wait(timeout=60)
    # The owner is dead: its lock is stale, the run it left is partial.
    assert inspect_journal(journal)['state'] == 'partial'
    taken = RestartJournal(journal, contract)
    assert taken.previous_state == 'partial' and taken.state == 'running'
    assert json.loads((journal / 'owner.json').read_text())['run_id'] == taken.run_id
    # The same process cannot open a second writer on its own live journal either.
    with pytest.raises(ValueError, match=f'pid {os.getpid()}'):
        RestartJournal(journal, contract)
    taken.close()
    assert not (journal / 'owner.json').exists()
    again = RestartJournal(journal, contract)
    assert again.previous_state == 'partial'
    again.close()


def test_stale_lock_of_a_dead_process_is_taken_over(tmp_path):
    p, eps, contract = contract_for(tmp_path)
    journal = tmp_path / 'journal'
    journal.mkdir()
    dead = int(subprocess.run([sys.executable, '-c', 'import os;print(os.getpid())'], capture_output=True, text=True, check=True).stdout)
    (journal / 'owner.json').write_text(json.dumps(dict(marker=streamed_restart.MARKER, run_id='deadbeef', pid=dead,
                                                          host=streamed_restart.socket.gethostname(), started=0.)))
    opened = RestartJournal(journal, contract)
    assert opened.previous_state == 'partial'
    assert json.loads((journal / 'owner.json').read_text())['run_id'] == opened.run_id
    opened.close()


@pytest.mark.parametrize('case', ['other_host', 'unreadable', 'foreign'])
def test_locks_that_cannot_be_judged_are_refused_by_name(tmp_path, case):
    p, eps, contract = contract_for(tmp_path)
    journal = tmp_path / 'journal'
    journal.mkdir()
    if case == 'other_host':
        (journal / 'owner.json').write_text(json.dumps(dict(marker=streamed_restart.MARKER, run_id='abc', pid=os.getpid(), host='elsewhere', started=0.)))
        match = "owned by run abc on host 'elsewhere'; liveness cannot be checked across hosts"
    elif case == 'unreadable':
        (journal / 'owner.json').write_bytes(b'{')
        match = 'unreadable owner.json'
    else:
        (journal / 'owner.json').write_text(json.dumps(dict(marker='something-else', pid=os.getpid())))
        match = 'owner.json was not written by this journal'
    with pytest.raises(ValueError, match=match):
        RestartJournal(journal, contract)
    # Nothing was taken over or changed.
    assert not (journal / 'status.json').exists()


def test_ownership_is_taken_again_after_a_terminal_state(tmp_path):
    p, eps, contract = contract_for(tmp_path)
    journal = RestartJournal(tmp_path / 'journal', contract)
    first = journal.run_id
    journal.complete()
    assert journal.state == 'completed' and not (tmp_path / 'journal' / 'owner.json').exists()
    # A second backward on the same result publishes again: the run re-owns its journal.
    journal.record_signals(torch.zeros(13, 3, dtype=torch.float64))
    assert json.loads((tmp_path / 'journal' / 'owner.json').read_text())['run_id'] == first
    assert journal.state == 'running' and journal.previous_state is None
    journal.complete()
    assert inspect_journal(tmp_path / 'journal')['state'] == 'completed'


# ---- integrity of every file -----------------------------------------------------------
def tamper(meta_path, mutate):
    payload = json.loads(meta_path.read_text())
    mutate(payload)
    meta_path.write_text(json.dumps(payload))


@pytest.mark.parametrize('check', ['dtype', 'bytes', 'shape', 'count', 'missing', 'checksum', 'file', 'kind', 'marker', 'no_arrays'])
def test_every_record_file_is_checked(tmp_path, monkeypatch, check):
    p, eps, settings = interrupted_forward(tmp_path, monkeypatch, 3)
    journal = tmp_path / 'journal'
    record = journal / 'forward-3'
    meta = record / 'meta.json'
    reasons = dict(dtype='state-1.bin declares an unknown dtype \'float99\'',
                   bytes='state-1.bin declares 5 bytes, its shape and dtype need ',
                   shape='state-0.bin holds ',
                   count=f'array count {state_count(p, eps)} ({state_count(p, eps)+1} expected: {state_count(p, eps)} state arrays and the signals)',
                   missing='signals.bin is missing',
                   checksum='checksum mismatch in state-2.bin',
                   file='an array description lacks file, shape or dtype',
                   kind="meta.json declares kind 'backward', pointer expects 'forward'",
                   no_arrays='meta.json lists no arrays')
    if check == 'dtype':
        tamper(meta, lambda m: m['arrays'][1].update(dtype='float99'))
    elif check == 'bytes':
        tamper(meta, lambda m: m['arrays'][1].update(bytes=5))
    elif check == 'shape':
        def shrink(m):
            m['arrays'][0]['shape'][1] -= 1
            m['arrays'][0]['bytes'] = 8*int(torch.tensor(m['arrays'][0]['shape']).prod())
        tamper(meta, shrink)
    elif check == 'count':
        tamper(meta, lambda m: m['arrays'].pop(2))
    elif check == 'missing':
        (record / 'signals.bin').unlink()
    elif check == 'checksum':
        tamper(meta, lambda m: m['arrays'][2].update(sha256='0'*64))
    elif check == 'file':
        tamper(meta, lambda m: m['arrays'][1].pop('file'))
    elif check == 'kind':
        tamper(meta, lambda m: m.update(kind='backward'))
    elif check == 'marker':
        tamper(meta, lambda m: m.update(marker='other'))
    else:
        tamper(meta, lambda m: m.update(arrays=[]))
    if check == 'marker':
        with pytest.raises(ValueError, match='not written by this journal'):
            _validate_record(journal, 'forward-3', 'forward')
        return
    cheap = check not in ('count', 'checksum')
    view = inspect_journal(journal)
    assert view['records']['forward']['valid'] is (not cheap)
    if cheap:
        assert reasons[check] in view['records']['forward']['reason']
        with pytest.raises(RestartRecordDamaged) as info:
            _validate_record(journal, 'forward-3', 'forward')
        assert info.value.record == 'forward-3' and info.value.reason.startswith(reasons[check])
    elif check == 'checksum':
        description = json.loads(meta.read_text())['arrays'][2]
        with pytest.raises(RestartRecordDamaged, match=re.escape(reasons[check])):
            _read_array(record / 'state-2.bin', description, torch.empty(description['shape'], dtype=torch.float64))
    # The damaged record is rolled back to forward-2 with the same reason named in the report.
    resumed = StreamedSimulation(p, settings)(eps)
    rollback, = resumed.report['restart_rollbacks']
    assert rollback['record'] == 'forward-3' and rollback['resumed_from'] == 'forward-2' and reasons[check] in rollback['reason']
    assert resumed.report['forward_resumed_from_block'] == 2
    torch.autograd.grad(resumed.signals.square().sum(), eps)
    assert resumed.report['restart_state'] == 'completed'


def test_written_descriptions_carry_checksum_dtype_shape_count_and_bytes(tmp_path, monkeypatch):
    p, eps, settings = interrupted_forward(tmp_path, monkeypatch, 2)
    meta = json.loads((tmp_path / 'journal' / 'forward-2' / 'meta.json').read_text())
    from torchfdtd.streamed_restart import sha256_file
    n = state_count(p, eps)
    assert meta['marker'] == streamed_restart.MARKER and meta['kind'] == 'forward' and meta['block'] == 2
    assert len(meta['arrays']) == n+1 and [d['file'] for d in meta['arrays']] == [f'state-{i}.bin' for i in range(n)]+['signals.bin']
    for description in meta['arrays']:
        path = tmp_path / 'journal' / 'forward-2' / description['file']
        assert description['bytes'] == path.stat().st_size == 8*int(torch.tensor(description['shape']).prod())
        assert description['dtype'] == 'float64' and description['sha256'] == sha256_file(path)
    pointer = json.loads((tmp_path / 'journal' / 'latest-forward.json').read_text())
    assert pointer == dict(directory='forward-2', kind='forward', block=2, previous='forward-1')


# ---- terminal states -----------------------------------------------------------------
def test_terminal_states_are_distinct_and_partial_results_stay_readable(tmp_path, monkeypatch):
    p = scene()
    eps = torch.full(p.region.shape, 1.7, dtype=torch.float64, requires_grad=True)
    states = {}
    # completed
    settings = options(tmp_path / 'completed')
    result = StreamedSimulation(p, settings)(eps)
    torch.autograd.grad(result.signals.square().sum(), eps)
    states['completed'] = inspect_journal(tmp_path / 'completed' / 'journal')
    assert states['completed']['state'] == 'completed' and states['completed']['complete'] and states['completed']['records'] == {}
    # cancelled: the records at the boundary are valid and readable without owning the journal
    cancel = threading.Event()
    settings = options(tmp_path / 'cancelled')
    original = SlabBlockOperator.forward
    calls = [0]

    def wrapper(self, *args, **kwargs):
        calls[0] += 1
        if calls[0] == 2:
            cancel.set()
        return original(self, *args, **kwargs)
    monkeypatch.setattr(SlabBlockOperator, 'forward', wrapper)
    with pytest.raises(StreamCancelled):
        StreamedSimulation(p, settings, cancel=cancel)(eps)
    monkeypatch.undo()
    states['cancelled'] = inspect_journal(tmp_path / 'cancelled' / 'journal')
    assert states['cancelled']['state'] == 'cancelled' and states['cancelled']['status']['phase'] == 'forward'
    assert states['cancelled']['records']['forward'] == dict(record='forward-2', block=2, previous='forward-1',
                                                             bytes=states['cancelled']['records']['forward']['bytes'], complete=False, arrays=state_count(p, eps)+1, valid=True)
    contract = json.loads((tmp_path / 'cancelled' / 'journal' / 'contract.json').read_text())
    reader = RestartJournal(tmp_path / 'cancelled' / 'journal', contract)
    templates = _StreamedExecution().host(p, eps.detach(), None).state()
    block, state, signals = reader.load_forward(lambda: tuple(torch.zeros_like(s) for s in templates), torch.empty(13, 3, dtype=torch.float64))
    reader.close()
    assert block == 2 and torch.isfinite(signals[:6]).all() and float(signals[:6].abs().sum()) > 0
    # failed: the reason is named, the records survive
    settings = options(tmp_path / 'failed')
    result = StreamedSimulation(p, settings)(eps)
    interrupt(monkeypatch, 'transpose', 1)
    with pytest.raises(Interrupted):
        torch.autograd.grad(result.signals.square().sum(), eps)
    monkeypatch.undo()
    states['failed'] = inspect_journal(tmp_path / 'failed' / 'journal')
    assert states['failed']['state'] == 'failed' and states['failed']['status']['reason'] == 'Interrupted: interrupted transpose after 1 calls'
    assert states['failed']['records']['forward']['complete'] and states['failed']['records']['backward']['block'] == 4
    # partial: a running status whose owner is dead
    settings = options(tmp_path / 'partial')
    result = StreamedSimulation(p, settings)(eps)
    journal = tmp_path / 'partial' / 'journal'
    dead = int(subprocess.run([sys.executable, '-c', 'import os;print(os.getpid())'], capture_output=True, text=True, check=True).stdout)
    for name in ('owner.json', 'status.json'):
        payload = json.loads((journal / name).read_text())
        payload['pid'] = dead
        (journal / name).write_text(json.dumps(payload))
    states['partial'] = inspect_journal(journal)
    assert states['partial']['state'] == 'partial' and states['partial']['records']['forward']['complete']
    assert len({s['state'] for s in states.values()}) == 4


# ---- durability primitives -------------------------------------------------------------
def test_rename_into_place_uses_the_platform_durability_primitive(tmp_path, monkeypatch):
    p, eps, contract = contract_for(tmp_path)
    calls = []
    if os.name == 'nt':
        real = streamed_restart._move_file_ex

        def recording(source, target, flags):
            calls.append((Path(source).name, Path(target).name, flags))
            return real(source, target, flags)
        monkeypatch.setattr(streamed_restart, '_move_file_ex', recording)
        assert streamed_restart._fsync_directory(tmp_path) is None
    else:
        real_open, real_fsync = os.open, os.fsync
        directories = set()

        def recording_open(path, flags, *args, **kwargs):
            fd = real_open(path, flags, *args, **kwargs)
            if Path(path).is_dir():
                directories.add(fd)
            return fd

        def recording_fsync(fd):
            if fd in directories:
                calls.append(fd)
            return real_fsync(fd)
        monkeypatch.setattr(os, 'open', recording_open)
        monkeypatch.setattr(os, 'fsync', recording_fsync)
    journal = RestartJournal(tmp_path / 'journal', contract)
    journal.record_signals(torch.zeros(13, 3, dtype=torch.float64))
    journal.close()
    if os.name == 'nt':
        files = [c for c in calls if c[1] in ('contract.json', 'status.json', 'meta.json', 'latest-forward.json')]
        assert files and all(flags == MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH for _, _, flags in files)
        assert ('forward-complete.tmp', 'forward-complete', MOVEFILE_WRITE_THROUGH) in calls
    else:
        # contract, status, owner directory entry, meta, record rename, pointer: each followed by a directory fsync.
        assert len(calls) >= 6


def test_durability_levels_are_stated_separately(tmp_path):
    text = (Path(__file__).resolve().parents[1] / 'docs' / 'STREAMED_RESTART.md').read_text(encoding='utf-8')
    section = text[text.index('## Durability levels'):]
    assert 'process-kill consistency' in section and 'power-loss durability' in section
    assert 'not exercised' in section and 'MOVEFILE_WRITE_THROUGH' in section and 'fsync' in section
