"""Memory admission of the full-autograd reference oracles.

Full autograd keeps the saved tensors of every timestep until backward, so the
retained graph grows with cells times steps. It is admitted by an estimate of
that graph against 80 percent of free CUDA memory or of available host memory,
the rules of the resident solvers, and an optional caller cap. The tensor part
counts per step one restart state (fields, CPML memories, stored PMC faces and
pole banks) at the field element size, the reciprocal and scaled permittivity
at every material sample, and for ADE the oscillator coefficients a, d, 4d
and k at every oscillator sample. The graph nodes and their tensor metadata
live in host memory on either device; each source term adds about six nodes
per step. The constants are upper bounds of the peaks measured on CPU and
CUDA through forward and backward, recorded in
docs/validation/oracle_graph_memory.json by benchmarks/oracle_graph_memory.py.
"""
import math
import sys

import torch

from .boundaries import BoundaryDescription, material_shape
from .cuda_memory import cuda_budget_limit, cuda_mem_info
from .memory_profile import host_memory

# Retained tensor bytes per step in units of the per-step tensor count above.
# Tensor ADE adds three field components per operator application of one step.
GRAPH_TENSOR_FACTORS={'yee':2,'ade':2.5,'tensor_ade':9}
# Per-step tensor bytes independent of the grid: small saved tensors and the
# 512-byte rounding of small CUDA allocations.
GRAPH_SMALL_TENSOR_BYTES=32*1024
# Host bytes per step for graph nodes and tensor metadata (tensor ADE: per
# operator application), and per source term per step.
GRAPH_NODE_BYTES={'yee':1024**2,'ade':1024**2,'tensor_ade':3*1024**2//2}
GRAPH_SOURCE_TERM_BYTES=32*1024
# Steps' worth of graph added for the initial state, inputs and the backward.
GRAPH_EXTRA_STEPS=4
# Host memory of the first call: autograd engine and thread pools, and on CUDA
# the kernel modules and libraries loaded on first use.
GRAPH_FIXED_HOST_BYTES={'cpu':16*1024**2,'cuda':1024**3}


def oracle_state_bytes(region,pole_count=0):
    """One restart state: E, H, CPML memories, stored faces and pole banks."""
    boundary=BoundaryDescription(region)
    n=math.prod(region.shape)
    cpml=sum(math.prod(s['shape']) for group in boundary.cpml.values() for s in group)
    faces=sum(math.prod(shape) for blocks in boundary.pmc_blocks.values() for _,_,shape in blocks)
    electric_faces=sum(math.prod(shape) for _,_,shape in boundary.pmc_blocks['E'])
    item=(8 if region.precision=='float64' else 4)*(2 if region.complex_fields else 1)
    return (6*n+cpml+faces+pole_count*(6*n+2*electric_faces))*item


def oracle_source_terms(project):
    """Injected source terms per step: polarization components, doubled for one-way sources."""
    return sum(len(project.resolved_source(s).polarization_components)*(2 if s.injection=='oneway' else 1)
               for s in project.sources if s.enabled)


def oracle_graph_bytes(region,kind,*,pole_count=0,iterations=0,material_elements=None,parameter_elements=0,source_terms=1):
    """Estimated per-run graph tensor bytes and graph node bytes, before fixed allowances.

    kind is 'yee' (dielectric, modal and source oracles), 'ade' or 'tensor_ade';
    iterations is the fixed Neumann length of the tensor ADE step.
    material_elements is the permittivity (epsilon_inf) element count, three
    components per material sample when omitted; parameter_elements is the
    element count of the largest ADE oscillator parameter (strength, omega0 or
    gamma, pole axis included); source_terms counts the injected terms.
    """
    steps=region.steps+GRAPH_EXTRA_STEPS
    real=8 if region.precision=='float64' else 4
    unit=oracle_state_bytes(region,pole_count)
    applies=1
    if kind=='tensor_ade':
        applies=2*pole_count+1+iterations*(1+pole_count)
        unit+=3*math.prod(region.shape)*real*(2 if region.complex_fields else 1)*applies
    else:
        if material_elements is None:material_elements=3*math.prod(material_shape(region))
        unit+=2*material_elements*real+(4*parameter_elements*real if kind=='ade' else 0)
    return dict(tensor_bytes=math.ceil(steps*(GRAPH_TENSOR_FACTORS[kind]*unit+GRAPH_SMALL_TENSOR_BYTES)),
                node_bytes=steps*(GRAPH_NODE_BYTES[kind]*applies+GRAPH_SOURCE_TERM_BYTES*source_terms))


def oracle_requirements(estimate,device):
    """Bytes admitted on the oracle's device (graph) and on the host.

    On CPU the tensors and nodes share host memory. On CUDA the tensors, plus
    the cuBLAS workspaces the resident spectral reservation also counts, are
    on the device and the nodes on the host. Under the Windows WDDM driver the
    process also commits host memory for its device allocations (measured:
    the host peak grew with the device graph plus the nodes), so there the
    device graph counts against host memory too.
    """
    device=torch.device(device)
    if device.type=='cuda':
        from .adjoint_memory import _spectral_library_reservation
        graph=estimate['tensor_bytes']+_spectral_library_reservation(device)
        host=estimate['node_bytes']+GRAPH_FIXED_HOST_BYTES['cuda']+(graph if sys.platform=='win32' else 0)
        return dict(graph=graph,host=host)
    host=estimate['tensor_bytes']+estimate['node_bytes']+GRAPH_FIXED_HOST_BYTES['cpu']
    return dict(graph=host,host=host)


def check_oracle_dtype(epsilon,region):
    """The oracle runs at the permittivity's dtype; the estimate uses the project precision."""
    if not isinstance(epsilon,torch.Tensor) or epsilon.dtype not in (torch.float32,torch.float64):
        raise ValueError('epsilon must be a real float32 or float64 torch Tensor.')
    if (epsilon.dtype==torch.float64)!=(region.precision=='float64'):
        raise ValueError('epsilon dtype must match the project precision.')


def admit_oracle(estimate,device,graph_budget_bytes=None):
    """Refuse a reference graph above device or host memory, or above the caller cap.

    graph_budget_bytes caps the graph estimate on the oracle's device.
    """
    if graph_budget_bytes is not None and (isinstance(graph_budget_bytes,bool) or not isinstance(graph_budget_bytes,int) or graph_budget_bytes<=0):
        raise ValueError('graph_budget_bytes must be a positive integer or None.')
    device=torch.device(device)
    cuda=device.type=='cuda'
    required=oracle_requirements(estimate,device)
    def refuse(what,value,limit,memory):
        raise ValueError(f'Full-autograd reference needs an estimated {value:,} bytes of {what}; the limit is '
                         f'{limit:,} bytes ({memory}). Reduce the grid or steps, or use the checkpointed adjoint; '
                         'graph_budget_bytes sets an explicit cap on the graph estimate.')
    if graph_budget_bytes is not None and required['graph']>graph_budget_bytes:
        refuse('retained graph',required['graph'],graph_budget_bytes,f'graph_budget_bytes={graph_budget_bytes:,}')
    if cuda:
        limit=cuda_budget_limit(device,required['graph'])
        if required['graph']>limit:
            refuse('retained graph on CUDA',required['graph'],limit,
                   f'80% of {cuda_mem_info(device)[0]:,} bytes of free CUDA memory')
    available=host_memory()['available_bytes']
    if available is not None and required['host']>int(available*.8):
        refuse('host memory for the graph nodes' if cuda else 'retained graph',required['host'],int(available*.8),
               f'80% of {available:,} bytes of available host memory')
    return dict(oracle_graph_bytes=required['graph'],oracle_host_bytes=required['host'])
