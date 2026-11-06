"""In-process registry of live streamed reservations: admission, not a quota.

Scope: one Python process. `_reservation` compares a run against the free
memory it observes, so two runs admitted at the same moment could each see the
same free bytes and together exceed them. This registry closes that gap for
runs in the same process: a phase (forward or backward) holds its byte
reservation on every tier it uses for as long as the phase runs, and a new
phase is admitted only when the bytes already held on each of its tiers plus
its own reservation fit the budget the new phase declares. The check and the
registration happen under one lock, so two threads cannot both pass.

It is not a machine-wide guarantee (other processes are invisible), it does
not limit what an admitted run allocates afterwards, and it does not replace
the live free-memory checks; it only refuses the second of two runs that would
together exceed the declared budget.
"""
from pathlib import Path
import threading

import torch


class ReservationRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._held = {}
        self._next = 0

    def held_bytes(self, tier, key):
        with self._lock:
            return sum(self._held.get((tier, key), {}).values())

    def acquire(self, requests):
        """Admit every (tier, key, required_bytes, budget_bytes) request together, or none.

        Raises ValueError naming the first tier that would overflow.
        """
        requests = list(requests)
        with self._lock:
            for tier, key, required, budget in requests:
                held = sum(self._held.get((tier, key), {}).values())
                if held+required > budget:
                    raise ValueError(
                        f'Concurrent streamed reservation refused: the {tier} tier {key!r} already holds {held} bytes '
                        f'for runs in this process and this run reserves {required} bytes, together more than its '
                        f'budget of {budget} bytes. The registry admits per process, is not a quota and does not see '
                        'other processes.')
            token = self._next
            self._next += 1
            for tier, key, required, budget in requests:
                self._held.setdefault((tier, key), {})[token] = required
        return Lease(self, token, [(tier, key) for tier, key, _, _ in requests])

    def _release(self, token, keys):
        with self._lock:
            for key in keys:
                entries = self._held.get(key)
                if entries is not None:
                    entries.pop(token, None)
                    if not entries:
                        del self._held[key]


class Lease:
    def __init__(self, registry, token, keys):
        self.registry, self.token, self.keys = registry, token, keys
        self.released = False

    def release(self):
        if not self.released:
            self.released = True
            self.registry._release(self.token, self.keys)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False


REGISTRY = ReservationRegistry()


def streamed_requests(options, reservation):
    """The tier requests of one streamed phase: host DRAM, the CUDA device and the scratch volume."""
    requests = [('host', 'process', reservation['host_reservation_bytes'], options.host_budget_bytes)]
    device = torch.device(options.device)
    if device.type == 'cuda':
        index = device.index if device.index is not None else torch.cuda.current_device()
        requests.append(('gpu', f'cuda:{index}', reservation['gpu_reservation_bytes'], options.gpu_budget_bytes))
    if options.state_storage == 'disk':
        anchor = Path(options.state_directory).expanduser().resolve().anchor or str(Path(options.state_directory).expanduser().resolve())
        requests.append(('disk', anchor, reservation['disk_reservation_bytes'], options.disk_budget_bytes))
    return requests


def streamed_lease(options, reservation, registry=REGISTRY):
    return registry.acquire(streamed_requests(options, reservation))
