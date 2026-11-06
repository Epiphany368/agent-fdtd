"""Metadata-only conservative admission for the scoped reversible adjoint.

The caller validates reversible physics before this planner. No field, waveform,
inverse coefficient, terminal copy or CUDA tensor is constructed here.
"""
import math

import torch


def _source_preparation(project):
    """Bound host FP64 pulse arithmetic and sampled-table conversion scratch."""
    active = [project.resolved_source(s) for s in project.sources if s.enabled]
    if not active:
        return 0
    terms = sum(len(s.polarization_components) for s in active)
    samples = sum(len(s.signal.time_s) for s in active if s.pulse == 'sampled')
    # source_time_signal retains times/x/u/envelope/carrier, with additional
    # Gaussian taper/chirp temporaries. Reserve sixteen FP64 timestep arrays,
    # four per prepared polarization term, and sampled-table conversion space.
    return project.region.steps*(128+32*terms)+64*samples


def _reversible_reservation(project, options, device):
    """Return the complete admitted allocation plan, with terminal-only storage.

    options provides gpu_budget_bytes, host_budget_bytes, resident_budget_bytes.
    Host budget means TOTAL host reservation, including CPU working tensors.
    Resident budget bounds the active device (total host for CPU). Incoming
    caller material storage is retained but not a new allocation. The caller
    initially requires contiguous scalar FP32 epsilon; the inherited worst-case
    diagonal coefficient/copy allowance is deliberately not reduced.
    """
    from .adjoint_memory import _resident_reservation
    from .differentiable import AdjointOptions
    from .memory_profile import host_memory
    from .cuda_memory import cuda_budget_limit

    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Reversible reservation requires CPU or CUDA.')
    budgets = {}
    for name in ('gpu_budget_bytes', 'host_budget_bytes', 'resident_budget_bytes'):
        value = getattr(options, name, None)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise ValueError(f'{name} must be a positive integer byte budget.')
        budgets[name] = value
    internal = AdjointOptions(checkpoints=1, storage='device',
        backward_kernel='fused' if device.type == 'cuda' else 'torch',
        resident_budget_bytes=budgets['resident_budget_bytes'])
    # Preserve explicit resident admission for budgeted/large regions. The base
    # may reject its own subtotal early; final checks below recheck the complete
    # reservation after diagnostics and source preparation. Host/GPU budgets
    # are applied only to those complete totals. No solver state is allocated.
    base = _resident_reservation(project, internal, device)
    cells = math.prod(project.region.shape)
    chunk = min(65536, 3*cells)
    diagnostic_buffers = 2*chunk*8
    diagnostic_rounding = 4096  # Scalar reductions and small allocation quanta.
    diagnostics = diagnostic_buffers+diagnostic_rounding
    preparation = _source_preparation(project)
    active = base['memory_reservation_bytes']+diagnostics
    host = base['host_reservation_bytes']+preparation+(diagnostics if device.type == 'cpu' else 0)
    if device.type == 'cpu':
        active = host
    if budgets['host_budget_bytes'] is not None and host > budgets['host_budget_bytes']:
        raise ValueError('Reversible total host reservation exceeds the explicit host byte budget.')
    if budgets['resident_budget_bytes'] is not None and active > budgets['resident_budget_bytes']:
        raise ValueError('Reversible total resident reservation exceeds the explicit resident byte budget.')
    available = host_memory()['available_bytes']
    if available is not None and host > int(.8*available):
        raise ValueError('Reversible total host reservation exceeds available host memory.')
    if device.type == 'cuda' and active > cuda_budget_limit(device, active, budgets['gpu_budget_bytes']):
        raise ValueError('Reversible complete reservation exceeds the CUDA byte budget.')

    parts = dict(base['workspace_components_bytes'])
    parts['reconstruction_diagnostics'] = diagnostics
    return dict(base,
        memory_reservation_bytes=active, host_reservation_bytes=host,
        gpu_reservation_bytes=active if device.type == 'cuda' else 0,
        workspace_reservation_bytes=base['workspace_reservation_bytes']+diagnostics,
        workspace_components_bytes=parts,
        workspace_model='reversible_conservative_'+base['workspace_model'],
        inherited_workspace_model=base['workspace_model'],
        inherited_workspace_note='Retains base replay/conversion and worst-case diagonal coefficient allowances; no additional scalar epsilon copy is assumed.',
        terminal_state_bytes=base['device_checkpoint_reservation_bytes'],
        terminal_state_count=1, checkpoint_states=0,
        device_checkpoint_reservation_bytes=0, host_checkpoint_reservation_bytes=0,
        disk_checkpoint_reservation_bytes=0, device_staging_reservation_bytes=0,
        diagnostic_chunk_elements=chunk, diagnostic_buffer_bytes=diagnostic_buffers,
        diagnostic_rounding_bytes=diagnostic_rounding,
        diagnostic_reservation_bytes=diagnostics,
        source_preparation_host_bytes=preparation,
        retained_caller_epsilon_bytes=cells*4,
        contiguous_seed_reservation_bytes=base['output_history_bytes'],
        inverse_extra_coefficient_tensor_bytes=0,
        checkpoint_replays=0, strategy='reversible',
        budgeted_resident=budgets['resident_budget_bytes'] is not None,
        state_storage='one immutable terminal E/H pair plus reusable primal and adjoint fields; histories remain O(steps*(sources+monitors))')
