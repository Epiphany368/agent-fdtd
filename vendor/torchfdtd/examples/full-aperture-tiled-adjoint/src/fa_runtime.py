"""Run-time plumbing shared by the FA drivers (2026-09-30): phase JSONL, background saves, stop control, optimizer offload, manifest.

Phase events: one JSON object per line with step, rank, phase, seconds,
overlap, cuda_synchronized; phase in setup, reference, forward, recorded_forward, adjoint, asm, optimizer, save, binary_evaluation,
nccl, idle, telemetry. Every rank appends to phase_events.rank<r>.jsonl; rank 0 merges them into phase_events.jsonl.
Saves are written to <name>.tmp and renamed, so a kill never leaves a truncated checkpoint.
"""
import hashlib, json, math, os, queue, shutil, subprocess, sys, threading, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PHASES = ("setup", "reference", "forward", "recorded_forward", "adjoint", "asm", "optimizer", "save", "binary_evaluation", "nccl", "idle", "telemetry")


def rank():
    return int(os.environ.get("FF_MULTI_RANK", os.environ.get("RANK", "0")))


def write_json(path, value):
    path = Path(path); tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1), encoding="utf-8"); os.replace(tmp, path)


# ---------------- phase events ----------------
class PhaseLog:
    def __init__(self, out, r):
        self.out = Path(out); self.rank = r; self.path = self.out / f"phase_events.rank{r}.jsonl"
        self.f = open(self.path, "a", encoding="utf-8", buffering=1)
        self._uuid = None

    def event(self, step, phase, seconds, *, synchronized=True, **extra):
        assert phase in PHASES, phase
        rec = dict(step=step, rank=self.rank, phase=phase, seconds=float(seconds), overlap=False, cuda_synchronized=bool(synchronized), t_unix=time.time(), **extra)
        self.f.write(json.dumps(rec) + "\n")

    def timed(self, step, phase, device=None, **extra):
        """with log.timed(k, 'forward', dev): ... -> one event, CUDA synchronized at the end when device is given."""
        log = self
        class _T:
            def __enter__(s): s.t = time.perf_counter(); return s
            def __exit__(s, *exc):
                if device is not None and torch.cuda.is_available(): torch.cuda.synchronize(device)
                s.seconds = time.perf_counter() - s.t
                if exc[0] is None: log.event(step, phase, s.seconds, synchronized=device is not None, **extra)
                return False
        return _T()

    def telemetry(self, step, device):
        rec = dict(step=step, rank=self.rank, phase="telemetry", overlap=False, t_unix=time.time())
        if torch.cuda.is_available():
            rec.update(allocated_bytes=torch.cuda.memory_allocated(device), peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(device),
                       peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(device))
            free, total = torch.cuda.mem_get_info(device); rec.update(free_bytes=free, total_bytes=total)
            torch.cuda.reset_peak_memory_stats(device)
            if os.environ.get("FA_NVSMI", "1") == "1" and shutil.which("nvidia-smi"):
                try:
                    if self._uuid is None: self._uuid = str(torch.cuda.get_device_properties(device).uuid)
                    q = subprocess.run(["nvidia-smi", f"--id=GPU-{self._uuid}" if not self._uuid.startswith("GPU-") else f"--id={self._uuid}",
                                        "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=10).stdout.strip().split(",")
                    rec.update(gpu_util_percent=float(q[0]), memory_util_percent=float(q[1]), nvsmi_memory_used_mib=float(q[2]), power_w=float(q[3]))
                except Exception:
                    pass
        self.f.write(json.dumps(rec) + "\n")

    def merge_ranks(self):
        """Rank 0: rewrite phase_events.jsonl from every rank's file (ranks in order, lines in order)."""
        if self.rank != 0: return
        self.f.flush(); lines = []
        for p in sorted(self.out.glob("phase_events.rank*.jsonl"), key=lambda p: int(p.name.split("rank")[1].split(".")[0])):
            lines += [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        tmp = self.out / "phase_events.jsonl.tmp"; tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"); os.replace(tmp, self.out / "phase_events.jsonl")


_LOGS = {}
def phase_log(out, r=None):
    r = rank() if r is None else r
    key = (str(Path(out).resolve()), r)
    if key not in _LOGS: _LOGS[key] = PhaseLog(out, r)
    return _LOGS[key]


# ---------------- background saves ----------------
class AsyncSaver:
    """One writer thread; files go to <name>.tmp and are renamed. The caller hands over host arrays it no longer mutates.
    Errors are raised in the caller at the next call. A bounded queue keeps the host copies of large checkpoints in check."""

    def __init__(self, depth=None):
        self.sync = os.environ.get("FA_SAVE_SYNC", "0") == "1"
        self.q = queue.Queue(maxsize=int(depth or os.environ.get("FA_SAVE_QUEUE", "6"))); self.err = None; self.t = None
        self.seconds = 0.0; self.bytes = 0

    def _start(self):
        if self.t is None and not self.sync:
            self.t = threading.Thread(target=self._run, name="fa-saver", daemon=True); self.t.start()

    def _run(self):
        while True:
            job = self.q.get()
            if job is None: self.q.task_done(); return
            try: self._do(*job)
            except BaseException as e: self.err = e
            finally: self.q.task_done()

    def _do(self, kind, path, payload, compressed):
        t = time.perf_counter(); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name + ".tmp")
        if kind == "npz":
            with open(tmp, "wb") as f: (np.savez_compressed if compressed else np.savez)(f, **payload)
        else:
            torch.save(payload, tmp)
        os.replace(tmp, path); self.seconds += time.perf_counter() - t; self.bytes += path.stat().st_size

    def _submit(self, job):
        if self.err is not None: e, self.err = self.err, None; raise RuntimeError(f"background save failed: {e!r}") from e
        if self.sync: self._do(*job); return
        self._start(); self.q.put(job)

    def save_npz(self, path, compressed=None, **arrays):
        if compressed is None: compressed = os.environ.get("FA_COMPRESS_STEPS", "0") == "1"
        self._submit(("npz", path, {k: (v if isinstance(v, np.ndarray) else np.asarray(v)) for k, v in arrays.items()}, bool(compressed)))

    def torch_save(self, obj, path):
        self._submit(("torch", path, obj, False))

    def close(self):
        if self.t is not None: self.q.put(None); self.q.join(); self.t = None
        if self.err is not None: e, self.err = self.err, None; raise RuntimeError(f"background save failed: {e!r}") from e


# ---------------- stop control ----------------
class StopControl:
    """Stop before a step when <out>/STOP (or FA_STOP_FILE) exists, or when FA_DEADLINE_UNIX is closer than the expected step
    time plus FA_DEADLINE_MARGIN_S (default 300 s for the last saves). Expected step = max of the last three steps (FA_FIRST_STEP_S before any)."""

    def __init__(self, out):
        self.files = [Path(out) / "STOP"] + ([Path(os.environ["FA_STOP_FILE"])] if os.environ.get("FA_STOP_FILE") else [])
        self.deadline = float(os.environ.get("FA_DEADLINE_UNIX", "0") or 0); self.margin = float(os.environ.get("FA_DEADLINE_MARGIN_S", "300"))
        self.first = float(os.environ.get("FA_FIRST_STEP_S", "0"))

    def check(self, step_seconds):
        for f in self.files:
            if f.exists(): return f"STOP file {f}"
        if self.deadline > 0:
            expect = max(step_seconds[-3:]) if step_seconds else self.first
            left = self.deadline - time.time()
            if left < expect + self.margin: return f"deadline in {left:.0f} s < expected step {expect:.0f} s + margin {self.margin:.0f} s"
        return None


# ---------------- optimizer offload (FA_OPT_OFFLOAD=1) ----------------
_PIN = {}
def offload(g, names=None):
    """Move the optimizer tensors of namespace g (theta, Adam moments; ns_run.py adds the design D / Dq) to pinned host buffers for
    the FDTD phases: on rank 0 of the non-symmetric run they are 4 x 1.6 GB, which would otherwise shrink the solver's admission budget.
    Names that alias one tensor share one buffer."""
    if os.environ.get("FA_OPT_OFFLOAD", "0") != "1": return
    done = {}
    for n in (names or g.get("FA_OFFLOAD_NAMES", ("theta", "opt_m", "opt_v"))):
        t = g.get(n)
        if not isinstance(t, torch.Tensor) or not t.is_cuda: continue
        if t.data_ptr() in done: g[n] = done[t.data_ptr()]; continue
        b = _PIN.get(n)
        if b is None or b.shape != t.shape or b.dtype != t.dtype: b = _PIN[n] = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
        b.copy_(t); done[t.data_ptr()] = b; g[n] = b
    torch.cuda.synchronize()


def reload(g, device, names=("theta", "opt_m", "opt_v")):
    if os.environ.get("FA_OPT_OFFLOAD", "0") != "1": return
    for n in names:
        if not g[n].is_cuda: g[n] = g[n].to(device)


# ---------------- objective diagnostics ----------------
def _jsonable(v):
    if isinstance(v, dict): return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [_jsonable(x) for x in v]
    if isinstance(v, (str, bool)) or v is None: return v
    if isinstance(v, np.ndarray): return v.tolist() if v.size <= 64 else f"array{list(v.shape)}"
    try: return float(v)
    except Exception: return repr(v)


def metrics(obj, T, arrays=False):
    """obj.metrics(T): the JSON-safe scalars (and with arrays=True also the numpy arrays). Diagnostics only: errors are recorded."""
    if not hasattr(obj, "metrics"): return (None, {}) if arrays else None
    try:
        with torch.no_grad(): m = obj.metrics(T) or {}
        sc, ar = (m.get("scalars", {}), m.get("arrays", {}) or {}) if isinstance(m, dict) and "scalars" in m else (m, {})
        sc = _jsonable(sc)
    except Exception as e:
        sc, ar = {"error": repr(e)}, {}
    return (sc, ar) if arrays else sc


# ---------------- manifest ----------------
def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    return h.hexdigest()


def environment():
    env = dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
               numpy=np.__version__, timing_cuda_synchronized=True)
    try:
        import cupy; env["cupy"] = cupy.__version__
    except Exception as e: env["cupy"] = f"unavailable: {e!r}"
    try:
        import scipy; env["scipy"] = scipy.__version__
    except Exception: pass
    try:
        import torchfdtd; env["torchfdtd"] = str(Path(torchfdtd.__file__).parent)
        from importlib import metadata; env["torchfdtd_version"] = metadata.version("torchfdtd")
    except Exception: pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0); env.update(gpu=p.name, gpu_memory_bytes=p.total_memory, compute_capability=f"{p.major}.{p.minor}",
                                                            device_count=torch.cuda.device_count())
    if shutil.which("nvidia-smi"):
        try: env["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.split()[0]
        except Exception: pass
    try: env["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
    except Exception: env["nccl"] = None
    return env


def write_manifest(out, *, geometry, objective, source_dir, world, extra=None):
    """run_manifest.json (schema_version 1) with resources plus the FA geometry/objective/env sections."""
    out = Path(out); src = Path(source_dir)
    sources = {p.name: sha256_file(p) for p in sorted(src.glob("*.py"))}
    now = datetime.now(timezone.utc).isoformat()
    dl = float(os.environ.get("FA_DEADLINE_UNIX", "0") or 0)
    m = dict(schema_version=1, run_id=os.environ.get("FA_RUN_ID", out.name), epoch=os.environ.get("FA_EPOCH", now),
             resource=dict(gpu_sku=os.environ.get("FA_GPU_SKU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                           gpu_count=world, started_utc=now,
                           hard_stop_deadline_utc=datetime.fromtimestamp(dl, timezone.utc).isoformat() if dl > 0 else None),
             geometry=geometry,
             precision=dict(field_dtype="float32", ASM_dtype="complex128", TF32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                            TF32_cudnn=bool(torch.backends.cudnn.allow_tf32), float32_matmul_precision=torch.get_float32_matmul_precision()),
             source_sha256=sources, environment=environment(), objective=objective,
             env={k: v for k, v in sorted(os.environ.items()) if k.startswith(("FF_", "EZ_", "EZC_", "FA_", "SM_", "CUDA_VISIBLE", "NCCL_", "OMP_"))
                  and "KEY" not in k and "TOKEN" not in k and "SECRET" not in k},
             target_steps=int(os.environ.get("FA_TARGET_STEPS", os.environ.get("EZC_OUTER", "60"))))
    if extra: m.update(extra)
    old = out / "run_manifest.json"
    if old.exists():                                                       # a resumed launch keeps the earlier epoch's manifest
        try: prev = json.loads(old.read_text(encoding="utf-8")).get("epoch", "previous")
        except Exception: prev = "previous"
        os.replace(old, out / f"run_manifest.{str(prev).replace(':', '')}.json")
    write_json(old, m)
    return m
