"""Metadata-only work and logical state-file traffic for streamed dielectric Yee.

No solver, material values, tensor allocation, state files, device admission, or
performance probes are used. Logical buffered-file bytes are not physical SSD or
PCIe transactions. Computation counts are cell-steps, not FLOPs or wall time.
"""
from functools import lru_cache
import math


def _positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ValueError(f'{name} must be a {"nonnegative" if zero else "positive"} integer.')
    return value


def _split(length, slots):
    """Same binomial split as differentiable._split, using logarithmic search."""
    low, high = 0, 1
    while math.comb(slots + high + 1, slots + 1) < length:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if math.comb(slots + middle + 1, slots + 1) < length:
            low = middle + 1
        else:
            high = middle
    return min(length - 1, math.comb(slots + low, slots + 1))


@lru_cache(maxsize=4096)
def _replay_summary(length, slots):
    """Aggregate replay advances and checkpoint saves without an event list.

    The left recursion is a loop. Right recursion reduces slots, so stack depth
    is bounded by the supported checkpoint capacity, not the time duration.
    """
    advances = saves = 0
    while length > 1:
        if slots == 0:
            advances += length * (length - 1) // 2
            break
        middle = _split(length, slots)
        right_advances, right_saves = _replay_summary(length - middle, slots - 1)
        advances += middle + right_advances
        saves += 1 + right_saves
        length = middle
    return advances, saves


def _zero_replay_calls(length, slots):
    """Nonempty forward replays whose first input is the implicit zero bank."""
    count = 0
    while length > 1:
        if slots == 0:
            return count + length - 1
        count += 1
        length = _split(length, slots)
    return count


def _segments(region, shape, item):
    """CPML derivative-domain intervals, with multiplicity two curl terms.

    Reproduces BoundaryDescription's active target intervals without preparing
    coefficients or psi tensors. Segment coordinates refer to derivative rows.
    """
    result = []
    for forward in (False, True):
        for axis, n in enumerate(shape):
            if n == 1:
                continue
            for side, face in enumerate(region.boundaries.pair(axis)):
                if face.kind != 'pml':
                    continue
                layers = region.pml_layers(axis, side)
                base = 0 if side == 0 else n - layers
                active_start, active_stop = (0, n - 1) if forward else (1, n)
                lo, hi = max(base, active_start), min(base + layers, active_stop)
                if lo >= hi:
                    continue
                dims = list(shape)
                dims[axis] = hi - lo
                result.append(dict(axis=axis, forward=forward, side=side,
                                   begin=lo - active_start, end=hi - active_start,
                                   rows=dims[0], row_bytes=math.prod(dims[1:]) * item,
                                   multiplicity=2))
    return result


def _depth_work(shape, width, depth, wrap_x, item, segments, local_slots):
    """Count row coverage without constructing per-cell/per-row index arrays."""
    nx, ny, nz = shape
    field_row_bytes = 6 * ny * nz * item
    gathered = owned = cells = 0
    # At most twelve interval counters regardless of domain dimensions.
    segment_owned = [0] * len(segments)
    for lo in range(0, nx, width):
        hi = min(nx, lo + width)
        begin, end = lo - depth, hi + depth
        if not wrap_x:
            begin, end = max(0, begin), min(nx, end)
        extent = end - begin
        cells += extent * ny * nz
        gathered += extent * field_row_bytes
        owned += (hi - lo) * field_row_bytes
        for i, segment in enumerate(segments):
            if segment['axis'] == 0:
                # Nonperiodic X CPML is indexed by indices[:-1]. Electric
                # derivatives target row+1; magnetic derivatives target row.
                first = max(begin, segment['begin'])
                stop = min(end - 1, segment['end'])
                take = max(0, stop - first)
                shift = 0 if segment['forward'] else 1
                count = max(0, min(stop, hi - shift) - max(first, lo - shift))
            else:
                # Repeated periodic windings are separate copied rows, even
                # when the same global index occurs several times in a tile.
                take, count = extent, hi - lo
            row_bytes = segment['row_bytes'] * segment['multiplicity']
            gathered += take * row_bytes
            owned += count * row_bytes
            segment_owned[i] += count
    if any(count != segment['rows'] for count, segment in zip(segment_owned, segments)):
        raise ValueError('Unsupported CPML ownership: slab destinations do not cover every state row exactly once.')
    replays, saves = _replay_summary(depth, min(local_slots, depth - 1))
    return dict(depth=depth, extended_cells=cells, gathered_state_bytes=gathered,
                owned_state_bytes=owned, local_replay_steps_per_tile=replays,
                local_checkpoint_saves_per_tile=saves,
                halo_cell_visits=cells - math.prod(shape))


def estimate_streamed_work(project, options, diagonal=False):
    """Return exact current-schedule work and logical StateStore byte counts.

    Inputs describe dense scalar/diagonal dielectric epsilon, not a geometry,
    density, tensor, endpoint, ADE, or trainable-source carrier. No values are
    sampled. Admission and physics validation remain the caller/runtime's job.
    Host storage has zero file-I/O counts; ``state_traffic_if_disk`` reports the
    same algorithm's logical file bytes if it were explicitly disk-backed.
    """
    from .models import Project
    from .streamed import StreamedAdjointOptions

    if not isinstance(project, Project) or type(options) is not StreamedAdjointOptions:
        raise ValueError('Work estimates require a Project and ordinary StreamedAdjointOptions; specialized geometry/density/ADE carriers are unsupported.')
    if not isinstance(diagonal, bool):
        raise ValueError('diagonal must be boolean; only dense scalar or diagonal epsilon is supported.')
    region = project.region
    if region.interface_method != 'staircase':
        raise ValueError('Subpixel/tensor constitutive work is unsupported; use staircase dielectric coefficients.')
    active = {obj.material for obj in project.structures if obj.enabled}
    if any(material.name in active and (material.model != 'dielectric' or material.oscillators)
           for material in project.materials):
        raise ValueError('Tensor and dispersive ADE material work estimates are unsupported.')
    if any(source.enabled and source.kind == 'tfsf' for source in project.sources):
        raise ValueError('TFSF incident-state work estimates are unsupported.')
    if region.run_control.auto_shutoff:
        raise ValueError('Work estimates require a fixed number of timesteps.')
    if any(face.kind in ('pmc', 'symmetric') for axis in range(3)
           for face in region.boundaries.pair(axis)):
        raise ValueError('PMC/symmetric faces are not implemented by the streamed work estimator; StreamedSimulation executes them directly.')
    if region.precision not in ('float32', 'float64'):
        raise ValueError('Work estimates require float32 or float64 precision.')
    shape = tuple(_positive(n, 'grid dimension') for n in region.shape)
    steps = _positive(region.steps, 'steps')
    width = _positive(options.slab_width, 'slab_width')
    depth = min(_positive(options.temporal_depth, 'temporal_depth'), steps)
    slots = _positive(options.checkpoints, 'checkpoints', zero=True)
    local_slots = _positive(options.local_checkpoints, 'local_checkpoints', zero=True)
    if slots > 32 or local_slots > 32:
        raise ValueError('Checkpoint capacities above 32 are unsupported.')
    if options.state_storage not in ('host', 'disk'):
        raise ValueError('State storage must be host or disk.')
    real_item = 4 if region.precision == 'float32' else 8
    item = real_item * (2 if region.complex_fields else 1)
    wrap_x = shape[0] > 1 and region.boundaries.pair(0)[0].kind in ('periodic', 'bloch')
    if wrap_x and any(face.kind == 'pml' for face in region.boundaries.pair(0)):
        raise ValueError('Mixed periodic-X and X-CPML metadata is unsupported.')
    segments = _segments(region, shape, item)
    state = 6 * math.prod(shape) * item + sum(
        s['rows'] * s['row_bytes'] * s['multiplicity'] for s in segments)
    blocks = (steps + depth - 1) // depth
    last_depth = steps - (blocks - 1) * depth
    summaries = {}
    for d in {depth, last_depth}:
        summaries[d] = _depth_work(shape, width, d, wrap_x, item, segments, local_slots)
        if summaries[d]['owned_state_bytes'] != state:
            raise ValueError('Unsupported state-row coverage in streamed slabs.')
    full, last = summaries[depth], summaries[last_depth]
    total_gather = (blocks - 1) * full['gathered_state_bytes'] + last['gathered_state_bytes']
    primal_reads = total_gather - full['gathered_state_bytes']
    replayed, checkpoint_saves = _replay_summary(blocks, slots)
    zero_calls = _zero_replay_calls(blocks, slots)
    replay_reads = (replayed - zero_calls) * full['gathered_state_bytes']
    traffic = dict(
        forward_state_read_bytes=primal_reads,
        forward_state_write_bytes=blocks * state,
        backward_state_read_bytes=replay_reads + primal_reads + (blocks - 1) * state + total_gather - blocks * state,
        backward_state_write_bytes=replayed * state + total_gather)
    traffic['total_state_io_bytes'] = sum(traffic.values())
    tile_count = (shape[0] + width - 1) // width
    forward_cells = (blocks - 1) * depth * full['extended_cells'] + last_depth * last['extended_cells']
    local_replay_cells = (blocks - 1) * full['local_replay_steps_per_tile'] * full['extended_cells'] + last['local_replay_steps_per_tile'] * last['extended_cells']
    local_steps = tile_count * ((blocks - 1) * full['local_replay_steps_per_tile'] + last['local_replay_steps_per_tile'])
    global_replay_cells = replayed * depth * full['extended_cells']
    block_work = []
    for d in sorted(summaries):
        count = blocks if d == depth == last_depth else (blocks - 1 if d == depth else 1)
        block_work.append(dict(count=count, **summaries[d]))
    disk_traffic = traffic if options.state_storage == 'disk' else {name: 0 for name in traffic}
    return dict(
        schema_version=1, material_representation='dense_diagonal_epsilon' if diagonal else 'dense_scalar_epsilon',
        grid_shape=list(shape), material_elements=math.prod(shape) * (3 if diagonal else 1),
        real_item_bytes=real_item, field_item_bytes=item, state_storage=options.state_storage,
        state_bytes=state, cpml_state_bytes=state - 6 * math.prod(shape) * item,
        blocks=blocks, temporal_depth=depth, final_block_depth=last_depth,
        tile_count=tile_count, forward_tile_visits=blocks * tile_count,
        backward_replay_tile_visits=replayed * tile_count, backward_transpose_tile_visits=blocks * tile_count,
        max_halo_cells_per_side=depth, block_work=block_work,
        global_replayed_blocks=replayed, global_replayed_steps=replayed * depth,
        zero_initial_replay_calls=zero_calls, global_checkpoint_saves=checkpoint_saves,
        local_replayed_steps=local_steps, forward_cell_steps=forward_cells,
        global_replay_cell_steps=global_replay_cells, local_replay_cell_steps=local_replay_cells,
        adjoint_cell_steps=forward_cells,
        total_cell_steps=2 * forward_cells + global_replay_cells + local_replay_cells,
        epsilon_reduction_passes=blocks, state_traffic_if_disk=dict(traffic), **disk_traffic,
        scope='Exact logical state-bank traffic and current Yee replay cell-steps. No material values, geometry synthesis, source/observation kernels, packing, checkpoint-copy work, device admission, physical I/O, timing or optimality claim.')
