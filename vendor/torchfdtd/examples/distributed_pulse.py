"""Rank-local periodic standing wave. Launch with torchrun, never plain Python.

python -m torch.distributed.run --standalone --nproc-per-node=2 \
    -m examples.distributed_pulse --backend gloo --output scratch/dd.json
"""
import argparse
from datetime import timedelta
import json
import hashlib
import math
import os
from pathlib import Path
import socket
import subprocess
import time

import torch
import torch.distributed as dist

from torchfdtd import Region, Boundaries, BoundaryFace
from torchfdtd.domain_decomposition import DistributedYeeDomain


def aggregate_reports(rows, *, shape, steps, backend, precision):
    """Pure compact-record aggregation; no global fields are accepted."""
    if not rows or sorted(row['rank'] for row in rows) != list(range(len(rows))):
        raise ValueError('Expected exactly one compact record per rank.')
    cells = math.prod(shape)
    if sum(row['owned_cells'] for row in rows) != cells:
        raise ValueError('Owned cell counts do not cover the global grid.')
    identities = [row['device_uuid'] for row in rows]
    if backend == 'nccl' and (any(not value for value in identities) or len(set(identities)) != len(rows)):
        raise ValueError('NCCL requires a distinct verified physical GPU UUID on every rank.')
    forward = max(row['forward_seconds'] for row in rows)
    backward = max(row['backward_seconds'] for row in rows)
    unit = torch.finfo(getattr(torch, precision)).eps
    rtol, atol = min(1e-3, 32*unit*steps), cells*min(1e-4, 16*unit*steps)
    checks = []
    for row in rows:
        for observed, expected in (('objective', 'oracle_objective'),
                ('uniform_epsilon_gradient', 'oracle_uniform_epsilon_gradient')):
            actual, reference = row[observed], row[expected]
            tolerance = atol+rtol*abs(reference)
            checks.append(dict(rank=row['rank'], quantity=observed, absolute_error=abs(actual-reference),
                tolerance=tolerance, passed=math.isfinite(actual) and math.isfinite(reference)
                and abs(actual-reference) <= tolerance))
    timing_valid = all(math.isfinite(row[key]) and row[key] > 0
        for row in rows for key in ('forward_seconds', 'backward_seconds'))
    passed = timing_valid and all(check['passed'] for check in checks)
    return dict(accuracy_passed=passed, accuracy_checks=checks,
        accuracy_tolerance=dict(rtol=rtol, atol=atol,
            definition='rtol=min(1e-3,32*eps*steps); atol=global_cells*min(1e-4,16*eps*steps)'),
        schema='torchfdtd.distributed_pulse.v1', backend=backend, world_size=len(rows),
        physical_gpu_count=len(set(identities)) if backend == 'nccl' else 0,
        precision=precision, global_shape=list(shape), global_cells=cells, steps=steps,
        forward_seconds_max_rank=forward, backward_seconds_max_rank=backward,
        forward_gcups=cells*steps/forward/1e9 if passed else None,
        gcups_definition='global_cells * forward_steps / max_rank_forward_seconds / 1e9; excludes backward',
        max_rank_cuda_peak_allocated_bytes=max(row['cuda_peak_allocated_bytes'] for row in rows),
        max_rank_cuda_peak_reserved_bytes=max(row['cuda_peak_reserved_bytes'] for row in rows),
        max_rank_workspace_reservation_bytes=max(row['workspace_reservation_bytes'] for row in rows),
        cpu_tensor_peak_measured=False, hardware_scaling_verified=False, ranks=rows)


def atomic_json(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    def finite_json(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {key: finite_json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite_json(item) for item in value]
        return value

    try:
        with temporary.open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(finite_json(result), stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def collective_write_json(path, result):
    """Share rank-zero write status through the default CPU control group."""
    error = None
    if dist.get_rank() == 0:
        try:
            atomic_json(path, result)
        except (OSError, ValueError, TypeError) as failure:
            error = f'{type(failure).__name__}: {failure}'
    statuses = [None]*dist.get_world_size()
    dist.all_gather_object(statuses, error)
    failures = [value for value in statuses if value is not None]
    if failures:
        raise RuntimeError('Collective result write failed: '+'; '.join(failures))


def prepare_data_group(backend, local_rank):
    """Validate hardware over the already initialized CPU Gloo control group."""
    rank, world = dist.get_rank(), dist.get_world_size()
    identity = dict(rank=rank, hostname=socket.gethostname(), device='cpu', device_uuid=None)
    error = None
    device = torch.device('cpu')
    try:
        if backend == 'nccl':
            if not torch.cuda.is_available():
                raise ValueError('CUDA is unavailable in this process or PyTorch build.')
            device = torch.device(f'cuda:{int(local_rank)}')
            torch.cuda.set_device(device)
            properties = torch.cuda.get_device_properties(device)
            cuda_uuid = getattr(properties, 'uuid', None)
            if cuda_uuid is None:
                raise ValueError('CUDA did not expose a device UUID.')
            normalize = lambda value: value.strip().lower().removeprefix('gpu-')
            query = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                capture_output=True, text=True, check=True, timeout=10)
            physical = {normalize(line) for line in query.stdout.splitlines() if line.strip()}
            uuid = normalize(str(cuda_uuid))
            if uuid not in physical:
                raise ValueError('CUDA UUID does not identify a verified physical GPU; MIG is unsupported.')
            identity.update(device=str(device), device_name=properties.name, device_uuid=uuid)
    except (ValueError, TypeError, RuntimeError, AssertionError, AttributeError, OSError, subprocess.SubprocessError) as failure:
        error = str(failure)
    records = [None]*world
    dist.all_gather_object(records, dict(identity=identity, error=error))
    if any(record['error'] is not None for record in records):
        raise ValueError(f'Collective CUDA device admission failed: {records}')
    if backend == 'nccl':
        uuids = [record['identity']['device_uuid'] for record in records]
        if len(set(uuids)) != world:
            raise ValueError('Each CUDA rank must have a distinct physical GPU UUID.')
        group = dist.new_group(backend='nccl', timeout=timedelta(seconds=120))
    else:
        group = None
    return device, identity, group


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('gloo', 'nccl'), default='gloo')
    parser.add_argument('--shape', nargs=3, type=int, default=(24, 12, 12))
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--precision', choices=('float32', 'float64'), default='float32')
    parser.add_argument('--epsilon', type=float, default=2.25)
    parser.add_argument('--checkpoints', type=int, default=2)
    parser.add_argument('--max-replayed-steps', type=int, default=100000)
    parser.add_argument('--rank-budget-mib', type=int, default=512)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--output', default='scratch/distributed-pulse.json')
    args = parser.parse_args(argv)
    if min(args.shape) < 5 or args.steps < 10 or args.warmup < 0 or args.repetitions < 1:
        parser.error('shape axes >= 5, steps >= 10, warmup >= 0 and repetitions >= 1 are required.')
    if not math.isfinite(args.epsilon) or args.epsilon <= 1:
        parser.error('epsilon must be finite and > 1.')
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        parser.error('Use torchrun to initialize the distributed environment.')
    # CPU control traffic must precede CUDA binding and NCCL communication.
    dist.init_process_group('gloo', timeout=timedelta(seconds=120))
    data_group = None
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        configurations = [None]*world
        dist.all_gather_object(configurations, vars(args))
        if any(value != configurations[0] for value in configurations):
            raise ValueError('All ranks must use identical launcher arguments.')
        device, identity, data_group = prepare_data_group(args.backend, os.environ.get('LOCAL_RANK', '0'))
        faces = {axis+side: BoundaryFace(kind='periodic') for axis in 'xyz' for side in ('_min', '_max')}
        region = Region(dimension='3d', size=tuple(n*.1 for n in args.shape), mesh=.1,
            steps=args.steps, precision=args.precision, pml_cells=3, boundaries=Boundaries(**faces))
        domain = DistributedYeeDomain(region, group=data_group, device=device, checkpoints=args.checkpoints,
            rank_budget_bytes=args.rank_budget_mib*1024**2, max_replayed_steps=args.max_replayed_steps)
        # Only local coordinates and state. Ez(x) has zero discrete divergence.
        e, h = domain.zero_state()
        angle = 2*math.pi*torch.arange(domain.ownership.begin, domain.ownership.end,
            dtype=domain.dtype, device=device)/region.shape[0]
        e[..., 2] = angle.sin()[:, None, None]
        epsilon = torch.full(domain.shape, args.epsilon, dtype=domain.dtype, device=device, requires_grad=True)
        durations = []

        def synchronize():
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            dist.barrier(group=data_group)

        for repetition in range(args.warmup+args.repetitions):
            epsilon.grad = None
            synchronize()
            if repetition == args.warmup and device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            begin = time.perf_counter()
            final_e, final_h = domain.propagate(e, h, epsilon, steps=args.steps)
            loss = domain.global_sum(final_e.square().sum()+.4*final_h.square().sum())
            synchronize()
            forward_end = time.perf_counter()
            loss.backward()
            synchronize()
            backward_end = time.perf_counter()
            if repetition >= args.warmup:
                durations.append((forward_end-begin, backward_end-forward_end))
            if repetition+1 < args.warmup+args.repetitions:
                del final_e, final_h, loss
        # Constant-memory scalar complex recurrence and its exact tangent.
        phase = complex(math.cos(2*math.pi/region.shape[0]), math.sin(2*math.pi/region.shape[0]))
        amplitude_e, amplitude_h, tangent_e, tangent_h = 1+0j, 0j, 0j, 0j
        for _ in range(args.steps):
            tangent_e = (tangent_e+domain.courant*(1-1/phase)*tangent_h/args.epsilon
                -domain.courant*(1-1/phase)*amplitude_h/args.epsilon**2)
            amplitude_e += domain.courant*(1-1/phase)*amplitude_h/args.epsilon
            amplitude_h += domain.courant*(phase-1)*amplitude_e
            tangent_h += domain.courant*(phase-1)*tangent_e
        oracle = math.prod(region.shape)/2*(abs(amplitude_e)**2+.4*abs(amplitude_h)**2)
        oracle_gradient = math.prod(region.shape)*(amplitude_e.conjugate()*tangent_e
            +.4*amplitude_h.conjugate()*tangent_h).real
        gradient = epsilon.grad.sum().detach()
        dist.all_reduce(gradient, group=data_group)
        row = dict(identity, owned_cells=math.prod(domain.shape), owned_x=[domain.ownership.begin, domain.ownership.end],
            measured_repetitions=[dict(forward_seconds=f, backward_seconds=b) for f,b in durations],
            forward_seconds=sum(v[0] for v in durations)/len(durations),
            backward_seconds=sum(v[1] for v in durations)/len(durations),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0,
            workspace_reservation_bytes=domain.last_report['rank_reservation_bytes'],
            checkpoint_capacity=args.checkpoints, peak_checkpoints=domain.last_report['peak_checkpoints'],
            replayed_steps=domain.last_report['replayed_steps'], halo_rows=1,
            communicated_bytes_total=domain.last_report['communicated_bytes'],
            objective=float(loss.detach()), oracle_objective=oracle,
            uniform_epsilon_gradient=float(gradient), oracle_uniform_epsilon_gradient=oracle_gradient)
        rows = [None]*world
        dist.all_gather_object(rows, row)
        result = aggregate_reports(rows, shape=region.shape, steps=args.steps, backend=args.backend, precision=args.precision)
        source_root = Path(__file__).resolve().parents[1]
        source_paths = ('examples/distributed_pulse.py', 'torchfdtd/domain_decomposition.py')
        result.update(warmup=args.warmup, repetitions=args.repetitions,
            torch_version=torch.__version__, cuda_version=torch.version.cuda,
            source_sha256={name: hashlib.sha256((source_root/name).read_bytes()).hexdigest() for name in source_paths},
            timing_scope='propagation, objective reduction and rank synchronization; backward includes replay and halo communication')
        collective_write_json(args.output, result)
        if rank == 0:
            print(json.dumps({key: result[key] for key in ('world_size','global_cells','forward_gcups','accuracy_passed','hardware_scaling_verified')}))
        if not result['accuracy_passed']:
            raise RuntimeError('Fourier objective/material-gradient accuracy gate failed; failed JSON retained, throughput withheld.')
    finally:
        if data_group is not None:
            dist.destroy_process_group(data_group)
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
