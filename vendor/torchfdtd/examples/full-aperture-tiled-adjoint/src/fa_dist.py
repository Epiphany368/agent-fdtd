"""Collectives for the FA drivers (2026-09-30): NCCL on the pod; with gloo (Windows / CPU checks) CUDA tensors are staged through the host."""
import torch
import torch.distributed as dist


def _gloo():
    return dist.get_backend() == "gloo"


def broadcast(t, src=0):
    if not dist.is_initialized(): return t                                 # single process
    if _gloo() and t.is_cuda:
        h = t.cpu(); dist.broadcast(h, src); t.copy_(h); return t
    dist.broadcast(t, src); return t


def all_reduce(t):
    if not dist.is_initialized(): return t
    if _gloo() and t.is_cuda:
        h = t.cpu(); dist.all_reduce(h); t.copy_(h); return t
    dist.all_reduce(t); return t


def send(t, dst):
    dist.send(t.cpu() if _gloo() else t, dst=dst)


def recv(t, src):
    if _gloo() and t.is_cuda:
        h = torch.empty(t.shape, dtype=t.dtype); dist.recv(h, src=src); t.copy_(h); return t
    dist.recv(t, src=src); return t


def gather_units(units, own, local, rank, device):
    """Rank 0 gets every unit's tuple of float32 tensors, in canonical unit order; each owner sends its own units in that order
    (NCCL point-to-point order per rank pair keeps the matching). Other ranks return {}."""
    out = {}
    for u in units:
        r = own[u]
        if rank == 0:
            if r == 0: out[u] = local.pop(u); continue
            h = recv(torch.empty(8, dtype=torch.int64, device=device), r); parts = []
            for j in range(int(h[0])):
                sh = recv(torch.empty(int(h[1 + j]), dtype=torch.int64, device=device), r)
                parts.append(recv(torch.empty(tuple(sh.tolist()), dtype=torch.float32, device=device), r))
            out[u] = tuple(parts)
        elif r == rank:
            parts = local.pop(u); h = torch.zeros(8, dtype=torch.int64, device=device); h[0] = len(parts)
            for j, t in enumerate(parts): h[1 + j] = t.dim()
            send(h, 0)
            for t in parts:
                assert t.dtype == torch.float32
                send(torch.tensor(t.shape, dtype=torch.int64, device=device), 0); send(t.contiguous(), 0)
    return out
