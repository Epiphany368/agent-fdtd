"""FA_KEEP_TRACE=1 (2026-09-30): the step's first forward of a (tile, polarization) unit is the recorded forward, and its autograd
graph is kept until the adjoint phase, which then runs only the reversible backward (the separate no-grad forward + recorded forward
re-run of the default path become one recorded forward). The boundary trace (T x 2 x Nx x Ny x 2 floats, ~20 GB for a 1040^2 window)
goes from a two-slot GPU ring straight into an exact-size pinned host buffer (DMA on a copy stream, finiteness checked on the GPU;
no pageable archive, no host memcpy, no host isfinite scan), so a kept unit holds only its fields/materials on the GPU (~56 B/cell).
A unit is kept only while the GPU keeps room for the next recorded solve (solver reservation / 0.8 + FA_KEEP_RESERVE_GIB); the
others take the default path. Same kernels and operation order as the default path, so fields and gradients do not change.
Binary-evaluation and --eval-only forwards stay no-grad.
FA_KEEP_MAX_UNITS=auto (2026-09-30, off unless set): no unit count cap; a kept unit's trace goes to pinned host RAM while this
rank's share of the container memory limit (cgroup memory.max, else MemAvailable; minus FA_KEEP_HOST_RESERVE_GIB, default 64,
split over LOCAL_WORLD_SIZE ranks, minus this process's other resident memory) has room, and to an exact-size GPU buffer when the
host share is used up and the GPU still admits the next recorded solve after holding the unit's fields and that buffer
(FA_KEEP_DEVICE=0 disables the GPU overflow). The trace values, copy order and checks are the same on both storages."""
import math, os
import torch
import torchfdtd.reversible_trace as _rt
import torchfdtd.reversible_cpml as _rc

ENABLED = os.environ.get("FA_KEEP_TRACE", "0") == "1"


# ---------------- exact-size pinned host buffers, reused across steps ----------------
class _Pool:
    def __init__(self): self.free = {}; self.count = 0; self.bytes = 0

    def get(self, numel, dtype):
        lst = self.free.setdefault((numel, dtype), [])
        if lst: return lst.pop()
        t = torch.empty(numel, dtype=dtype); ok = False
        try: ok = int(torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0)) == 0
        except Exception: ok = False
        if not ok: t = torch.empty(numel, dtype=dtype, pin_memory=True)         # (caching host allocator: may round the size up)
        self.count += 1; self.bytes += t.numel() * t.element_size()
        return t

    def put(self, t): self.free.setdefault((t.numel(), t.dtype), []).append(t)


POOL = _Pool()
PLACEMENT = {"next": "host"}                                               # set by Keeper.admit for the next DirectTrace ("host" | "device")


def _host_limit_bytes():
    """The container memory limit and current usage (cgroup v2, then v1), else the machine's MemTotal / MemAvailable."""
    for lim, cur in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                     ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            v = open(lim).read().strip()
            if v != "max" and int(v) < 1 << 60: return int(v), int(open(cur).read().strip())
        except (OSError, ValueError): pass
    info = {}
    with open("/proc/meminfo") as f:
        for line in f: k, v = line.split(":"); info[k] = int(v.split()[0]) * 1024
    return info["MemTotal"], info["MemTotal"] - info["MemAvailable"]


def _rss_bytes():
    with open("/proc/self/statm") as f: return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


class DirectTrace:
    """Drop-in for reversible_trace.AsyncBoundaryTrace (frame / commit / finish / reverse / close / archive)."""

    def __init__(self, shape, device, *, chunk_steps=32, dtype=torch.float32):
        device = torch.device(device)
        if device.index is None: device = torch.device("cuda", torch.cuda.current_device())
        self.shape, self.k, self.device, self.dtype = tuple(shape), int(chunk_steps), device, dtype
        self.compute = torch.cuda.current_stream(device); self.copy = torch.cuda.Stream(device=device)
        self.on_device = PLACEMENT["next"] == "device"; PLACEMENT["next"] = "host"
        self._flat = torch.empty(math.prod(self.shape), dtype=dtype, device=device) if self.on_device else POOL.get(math.prod(self.shape), dtype)
        self.archive = self._flat.view(self.shape); self.archive._fa_verified = False
        self.slots = [torch.empty((self.k, *self.shape[1:]), dtype=dtype, device=device) for _ in range(2)]
        self.done = [None, None]; self.ok = torch.ones((), dtype=torch.bool, device=device)
        self.next, self.state, self.released = 0, "writing", False

    def frame(self, n):
        if n != self.next or self.state != "writing": raise RuntimeError("DirectTrace: frames must be written once in ascending order")
        s = (n // self.k) % 2
        if n % self.k == 0 and self.done[s] is not None: self.compute.wait_event(self.done[s])   # slot's previous copy finished (GPU-side wait)
        return self.slots[s][n % self.k]

    def commit(self, n):
        self.next = n + 1
        if n % self.k == self.k - 1 or n == self.shape[0] - 1:
            start = (n // self.k) * self.k; cnt = n - start + 1; s = (n // self.k) % 2
            self.ok &= torch.isfinite(self.slots[s][:cnt]).all()
            ev = torch.cuda.Event(); ev.record(self.compute)
            with torch.cuda.stream(self.copy):
                self.copy.wait_event(ev); self.archive[start:start + cnt].copy_(self.slots[s][:cnt], non_blocking=True)
                d = torch.cuda.Event(); d.record(self.copy)
            self.done[s] = d

    def finish(self):
        if self.next != self.shape[0]: raise RuntimeError("DirectTrace: incomplete trace")
        self.copy.synchronize()
        if not bool(self.ok): raise RuntimeError("Recorded CPML boundary trace became nonfinite.")
        self.archive._fa_verified = True; self.state = "complete"; self.slots = None; self.done = [None, None]   # the GPU ring is freed here

    def reverse(self):
        if self.state != "complete": raise RuntimeError("DirectTrace: reverse needs a complete trace")
        return _Reader(self)

    def close(self):
        if not self.released:
            self.released = True
            try: self.copy.synchronize()
            except Exception: pass
            if self.on_device: self._flat = self.archive = None                  # back to the caching allocator
            else: POOL.put(self._flat)

    def __enter__(self): return self

    def __exit__(self, kind, error, tb): self.close()

    def __del__(self):
        try: self.close()
        except Exception: pass


class _Reader:
    def __init__(self, t):
        self.t = t; k = t.k; T = t.shape[0]
        self.slots = [torch.empty((k, *t.shape[1:]), dtype=t.dtype, device=t.device) for _ in range(2)]
        self.ev, self.entered, self.top = {}, set(), (T - 1) // k
        self._load(self.top)

    def _load(self, c):
        t = self.t; start = c * t.k; cnt = min(t.k, t.shape[0] - start); s = c % 2
        ev = torch.cuda.Event(); ev.record(t.compute)                     # the slot's previous chunk has been consumed
        with torch.cuda.stream(t.copy):
            t.copy.wait_event(ev); self.slots[s][:cnt].copy_(t.archive[start:start + cnt], non_blocking=True)
            h = torch.cuda.Event(); h.record(t.copy)
        self.ev[c] = h

    def __getitem__(self, n):
        c = n // self.t.k
        if c not in self.entered:
            self.entered.add(c); self.t.compute.wait_event(self.ev[c])
            if c - 1 >= 0 and (c - 1) not in self.ev: self._load(c - 1)
        return self.slots[c % 2][n - c * self.t.k]

    def __enter__(self): return self

    def __exit__(self, kind, error, tb):
        try: self.t.copy.synchronize()
        finally: self.slots = None; self.t.close()                        # the trace is not read again: back to the pool
        return False


_ORIG_REQUIRE_FINITE = None


def _require_finite(value, message, chunk):                               # the DirectTrace archive was checked on the GPU chunk by chunk
    if getattr(value, "_fa_verified", False): return
    return _ORIG_REQUIRE_FINITE(value, message, chunk)


def install():
    """Route async CPU traces through DirectTrace (only the keep models use trace_storage=cpu)."""
    global _ORIG_REQUIRE_FINITE
    if _ORIG_REQUIRE_FINITE is None:
        _ORIG_REQUIRE_FINITE = _rc._require_finite; _rc._require_finite = _require_finite
        _rt.AsyncBoundaryTrace = DirectTrace


# ---------------- kept units ----------------
class Keeper:
    """Holds (leaf, field a, field b, points) of the units solved with a live graph in this step's forward phase."""

    def __init__(self, device):
        mx = os.environ.get("FA_KEEP_MAX_UNITS", "1000000").strip().lower()
        self.auto = mx == "auto"
        self.device = device; self.held = {}; self.hold_bytes = 0; self.max_units = 1000000 if self.auto else int(mx)
        self.reserve = float(os.environ.get("FA_KEEP_RESERVE_GIB", "2")) * 2**30; self.kept = 0; self.declined = 0
        self.trace_bytes = 0; self.host_share = None; self.on_device = 0
        self.allow_device = os.environ.get("FA_KEEP_DEVICE", "1") == "1"

    def clear(self):
        self.held.clear(); self.kept = 0; self.declined = 0; self.on_device = 0

    def reservation(self, model, freq):
        """The solver's own reservation of a kept-path solve (cached per model; None if the solver refuses it right now)."""
        r = getattr(model, "_fa_keep_resv", None)
        if r is None:
            try: r = model._fa_keep_resv = int(model.plan(freq, device="cuda", material_components=3, block_size=32)["gpu_reservation_bytes"])
            except ValueError: return None
        reg = model.project.region; nx, ny = reg.shape[:2]
        self.trace_bytes = reg.steps * 2 * nx * ny * 2 * 4                     # (T, 2, Nx, Ny, 2) FP32, the DirectTrace archive
        return r

    def _host_room(self):
        """auto: whether one more trace fits this rank's host share (a free pool buffer of that size costs nothing)."""
        if POOL.free.get((self.trace_bytes // 4, torch.float32)): return True
        limit, used = _host_limit_bytes()
        if self.host_share is None:
            ranks = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1")))
            reserve = float(os.environ.get("FA_KEEP_HOST_RESERVE_GIB", "64")) * 2**30
            self.host_share = (limit - reserve) / ranks - max(0, _rss_bytes() - POOL.bytes)
        margin = 8 * 2**30                                                   # the whole container, all ranks: never closer than 8 GiB to the limit
        return POOL.bytes + self.trace_bytes <= self.host_share and limit - used >= self.trace_bytes + margin

    def admit(self, reservation_bytes, cells):
        """Keep the next unit if, after holding it, the GPU still admits a recorded solve (0.8 x free >= reservation).
        auto: its trace goes to host while the host share has room, else to the GPU if the GPU admits fields + trace + next solve."""
        PLACEMENT["next"] = "host"
        if reservation_bytes is None or len(self.held) >= self.max_units: self.declined += 1; return False
        torch.cuda.empty_cache()                                             # release what can be released, then count only the free memory: cache stuck in segments with live tensors is not room
        free, _ = torch.cuda.mem_get_info(self.device); unused = 0
        est = self.hold_bytes or 56 * cells
        if self.auto and not self._host_room():
            est += self.trace_bytes                                          # GPU overflow: the unit also holds its trace on the GPU
            ok = self.allow_device and (free + unused - est) * 0.8 >= reservation_bytes + self.reserve and (free + unused) * 0.8 >= reservation_bytes + self.trace_bytes
            if ok: PLACEMENT["next"] = "device"; self.on_device += 1
        else:
            ok = (free + unused - est) * 0.8 >= reservation_bytes + self.reserve and free * 0.8 + unused * 0.8 >= reservation_bytes
        self._last_device = ok and PLACEMENT["next"] == "device"
        if not ok: self.declined += 1
        return ok

    def store(self, unit, leaf, fa, fb, pts, bytes_before):
        self.held[unit] = (leaf, fa, fb, pts); self.kept += 1
        held = torch.cuda.memory_allocated(self.device) - bytes_before
        if getattr(self, "_last_device", False): held -= self.trace_bytes     # hold_bytes estimates the fields only; a GPU trace is added per admit
        self.hold_bytes = max(self.hold_bytes, held)

    def pop(self, unit): return self.held.pop(unit, None)
