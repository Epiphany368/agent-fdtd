"""The process I/O counters of torchfdtd.memory_accounting count every byte passed through read and write calls.

The scratch-disk fields of the memory report compare these deltas with the logical bytes the
bank files wrote and read (tests/test_memory_accounting_g5.py). Windows counts transfers per
call (GetProcessIoCounters); on Linux the matching counters are rchar and wchar of
/proc/self/io, while read_bytes and write_bytes count storage traffic only: a page rewritten
while still dirty is counted once and a read served by the page cache not at all.
"""
import pytest

from torchfdtd.memory_accounting import process_io


def test_process_io_counts_rewrites_and_cached_reads(tmp_path):
    block = bytes(range(256))*16  # one 4 KiB page
    with (tmp_path/'bank.bin').open('x+b', buffering=0) as handle:  # unbuffered, as StateStore opens its bank files
        start = process_io()
        if start is None:
            pytest.skip('no process I/O counters on this platform')
        for _ in range(64):
            handle.seek(0)
            assert handle.write(block) == len(block)
        middle = process_io()
        target = bytearray(len(block))
        for _ in range(64):
            handle.seek(0)
            assert handle.readinto(target) == len(block)
        end = process_io()
    assert middle['write_bytes']-start['write_bytes'] >= 64*len(block), start['instrument']
    assert end['read_bytes']-middle['read_bytes'] >= 64*len(block), start['instrument']
