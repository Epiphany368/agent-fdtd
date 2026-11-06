"""Dispersive subpixel interfaces: the averaging tensor of a node cell with dispersive laminate branches.

A node cell spans WINDOW mesh steps per axis around a Yee node. When it holds a
fraction f of one dispersive medium eps_m(w) and nondispersive media elsewhere,
with C and B the cell averages of eps and 1/eps over the nondispersive part and
n the unit normal of the dispersive surface, the constitutive map E = Z D is

    Z(w) = n n^T (B + f/eps_m(w)) + (I - n n^T)/(C + f eps_m(w))

(the averaging tensor of Farjadpour et al., Opt. Lett. 31, 2972, 2006 and Kottke,
Farjadpour and Johnson, Phys. Rev. E 77, 036611, 2008, with the dispersive
response split into its normal and tangential laminate branches as by Deinega
and Valuev, Opt. Lett. 32, 3429, 2007). The eight edge triplets of the node,
each with weight 1/8 as in the coupled operator of torchfdtd.subpixel, give the
edge of axis a and side s the share 1/2 [(Z Dbar)_a + s Z_aa delta_a], with Dbar
and delta the half sum and half difference of D on the two edges of axis a.
Every branch is a passive medium driven by D: the normal one is the dispersive
medium itself in series with B, the tangential one the parallel laminate
C + f eps_m, both advanced by the trapezoidal ADE update, so each node share is a
passive impedance whose high-frequency value is at most one. A node cell wholly
inside the medium gives its edges 1/2 D/eps_m; an edge between two such cells
keeps the Yee ADE of the material. Nondispersive cells next to a dispersive cell
use the static tensor n n^T B + (I - n n^T)/C, without coupling between a
D-driven edge and an E-updated one. The state of a D-driven sample is its D; its
E is assigned from D and the branches every step, never incremented, so no
offset between E and the E of its D can persist. See docs/SUBPIXEL_INTERFACES.md.
"""
from dataclasses import dataclass
import time

import numpy as np
import torch

from .geometry import object_bounds
from .subpixel_geometry import DielectricGeometry

ORDINARY, MIXED, WHOLE, NEXT = 0, 1, 2, 3
# Averaging window of a node cell, in mesh steps per axis: two steps was the most
# accurate on development Drude spheres of 3.6 to 11 cells per radius (window 1 to 3
# compared, docs/SUBPIXEL_INTERFACES.md); wider windows oversmooth small spheres.
WINDOW = 2.


def branch_coefficients(epsilon, oscillators, dt, n):
    """(eps, a, d, k) of a D-driven trapezoidal ADE bank of n samples; a, d and k have the poles on the first axis."""
    w0, strength, gamma = (np.stack([np.broadcast_to(np.asarray(o[j], dtype=float), (n,)) for o in oscillators]) for j in range(3))
    den = 1+.5*gamma*dt+.25*(w0*dt)**2
    return np.broadcast_to(np.asarray(epsilon, dtype=float), (n,)).copy(), .5*(w0*dt)**2, den, strength*dt*dt/(4*den)


class Branch:
    """eps x + sum_j P_j = d with P_j'' + g_j P_j' + w_j^2 P_j = s_j x: MaterialADE's update for a known d."""
    def __init__(self, grid, epsilon, oscillators, shape):
        eps, a, d, k = branch_coefficients(epsilon, oscillators, grid.time_step, shape[-1])
        self.torch = grid.is_torch
        self.x = grid._zeros(shape)
        self.P = grid._zeros((len(oscillators),)+tuple(shape)); self.Q = grid._zeros((len(oscillators),)+tuple(shape))
        grid.memory_states.extend((self.x, self.P, self.Q))
        lead = (1,)*(len(shape)-1)
        self.eps = grid._coefficient(eps.reshape(lead+eps.shape))
        self.a, self.d, self.k = (grid._coefficient(v.reshape(v.shape[:1]+lead+v.shape[1:])) for v in (a, d, k))
        self.k_sum = self.k.sum(0)
        self.host = eps, oscillators

    def advance(self, d):
        """Step to the new level of d and return x; eps x + sum_j P_j = d holds after every step."""
        response = (self.Q-self.a*self.P)/self.d
        new = (d-self.P.sum(0)-response.sum(0)-self.k_sum*self.x)/(self.eps+self.k_sum)
        delta = response+self.k*(new+self.x)
        self.P += delta
        self.Q *= -1
        self.Q += 2*delta
        if self.torch:
            self.x.copy_(new)
        else:
            self.x[...] = new
        return self.x

    def energy_terms(self, grid, scale):
        """(array, weight) pairs of scale (eps x^2 + sum_j (Q_j^2/dt^2 + w_j^2 P_j^2)/s_j), weights shaped like the arrays."""
        eps, oscillators = self.host
        shaped = lambda v: grid._coefficient(np.broadcast_to(np.asarray(v, dtype=float)*scale, self.x.shape).copy())
        terms = [(self.x, shaped(eps))]
        for j, (w0, strength, _) in enumerate(oscillators):
            strength = np.asarray(strength, dtype=float)
            inverse = np.divide(1, strength, out=np.zeros_like(strength), where=strength > 0)
            terms += [(self.Q[j], shaped(inverse/grid.time_step**2)), (self.P[j], shaped(np.asarray(w0)**2*inverse))]
        return terms


@dataclass
class DispersiveTensor:
    """Node kinds, the static tensors of NEXT nodes, and the D-driven samples, nodes and edges.

    ``kind`` holds one of ORDINARY, MIXED, WHOLE, NEXT per node. ``driven`` marks
    the samples whose field comes from the dispersive state (an end in a MIXED or
    WHOLE cell, not both WHOLE), and ``driven_owners`` gives the dispersive
    material of each, in sample order; ``standard`` samples lie between two WHOLE
    cells and keep the material ADE. ``active`` marks the nodes whose triplets
    change (every kind but ORDINARY and WHOLE cells away from the surface).
    ``groups`` has one entry per dispersive material."""
    node_shape: tuple
    kind: np.ndarray
    active: np.ndarray
    next_ids: np.ndarray
    next_tensors: np.ndarray
    driven: np.ndarray
    driven_owners: np.ndarray
    standard: np.ndarray
    owners: np.ndarray
    instantaneous: np.ndarray
    groups: list
    metadata: dict

    def static_tensors(self, ids):
        return self.next_tensors[np.searchsorted(self.next_ids, ids)]


def _node_points(region, coords, dim):
    return np.column_stack([region.mesh_nodes[a][coords[:, a]] if a < dim else np.zeros(len(coords)) for a in range(3)])


def _edge(region, coords, axis, side, dim, periodic):
    """Flat sample index, validity and Bloch phase of the edge of ``axis`` on ``side`` (0 upper, 1 lower) of each node."""
    shape = region.shape
    edge = coords.copy()
    if axis < dim and side:
        edge[:, axis] -= 1
    if axis >= dim:
        edge[:, axis] = 0
    good = np.ones(len(edge), bool); phase = np.ones(len(edge), complex)
    for b in range(3):
        if b in periodic:
            turns = np.floor_divide(edge[:, b], shape[b]); phase *= np.exp(1j*turns*region.bloch_phase[b]); edge[:, b] %= shape[b]
        else:
            good &= (edge[:, b] >= 0) & (edge[:, b] < shape[b]); edge[:, b] = np.clip(edge[:, b], 0, shape[b]-1)
    return np.ravel_multi_index(edge.T, shape)*3+axis, good, phase


def prepare_dispersive(project, ownership):
    """The dispersive tensor of every node cell that meets or touches a dispersive object, or None."""
    r = project.region; materials = project.materials
    active = {s.material for s in project.structures if s.enabled}
    dispersive = [i for i, m in enumerate(materials) if m.oscillators and m.name in active]
    if not dispersive:
        return None
    started = time.perf_counter()
    index = {m.name: i for i, m in enumerate(materials)}
    marks = DielectricGeometry(project, lambda m: 0. if m is None or not m.oscillators else 1.+dispersive.index(index[m.name]))
    media = DielectricGeometry(project, lambda m: 0. if m is None else 1.+index[m.name])
    plain = DielectricGeometry(project)
    background = r.background_index**2
    table = np.zeros((len(materials)+1, len(dispersive)+2)); table[0, -2:] = background, 1/background
    for i, m in enumerate(materials):
        if i in dispersive:
            table[i+1, dispersive.index(i)] = 1
        elif m.name in active:
            table[i+1, -2:] = m.instantaneous_epsilon, 1/m.instantaneous_epsilon
    dim, periodic = marks.dim, marks.periodic
    shape = np.array(r.shape); steps = np.array([v[1]-v[0] for v in r.mesh_nodes])
    node_shape = tuple(int(shape[a]) if a in periodic or a >= dim else int(shape[a]+1) for a in range(3))
    count = int(np.prod(node_shape)); tolerance = 1e-12
    window = steps*WINDOW
    radius = np.linalg.norm(window[:dim])/2*(1+1e-12)
    fraction = np.zeros(count, np.float32); material = np.full(count, -1, np.int16)
    cells = {}
    for start in range(0, count, 32768):
        ids = np.arange(start, min(start+32768, count))
        points = _node_points(r, np.stack(np.unravel_index(ids, node_shape), 1), dim)
        normal, near = marks.normals_and_candidates(points, radius)
        label = marks.epsilon(points).astype(int)
        inside = ~near & (label > 0)
        fraction[ids[inside]] = 1; material[ids[inside]] = label[inside]-1
        if np.any(near):
            values = media.volume_average(points[near], window, r.subpixel_quadrature, normal[near], table)
            parts = values[:, :len(dispersive)]; present = parts > tolerance
            if np.any(present.sum(axis=1) > 1):
                raise ValueError(f'Dispersive subpixel interfaces allow one dispersive material per node cell; '
                                 f'{int(np.count_nonzero(present.sum(axis=1) > 1))} node cells mix two. Separate the objects by at least one averaging window ({WINDOW:g} cells) or choose staircase interfaces.')
            total = parts.sum(axis=1); hit = total > tolerance
            fraction[ids[near][hit]] = np.minimum(total[hit], 1); material[ids[near][hit]] = np.argmax(parts[hit], axis=1)
            for k in np.flatnonzero(hit):
                cells[int(ids[near][k])] = (values[k, -2], values[k, -1], normal[near][k], min(total[k], 1.))
    kind = np.where(fraction >= 1-tolerance, WHOLE, np.where(fraction > 0, MIXED, ORDINARY)).astype(np.int8)
    grid = kind.reshape(node_shape)
    touching = np.zeros(node_shape, bool); boundary = np.zeros(node_shape, bool)
    for a in range(dim):
        for shift in (1, -1):
            for source, target in ((grid > 0, touching), (grid != WHOLE, boundary)):
                rolled = np.roll(source, shift, axis=a)
                if a not in periodic:
                    edge = [slice(None)]*3; edge[a] = slice(0, 1) if shift == 1 else slice(-1, None); rolled[tuple(edge)] = False
                target |= rolled
    kind[(touching.reshape(-1)) & (kind == ORDINARY)] = NEXT
    boundary = boundary.reshape(-1) & (kind == WHOLE)
    # Static tensors of the nondispersive cells next to a dispersive cell.
    next_ids = np.flatnonzero(kind == NEXT); next_tensors = np.zeros((len(next_ids), 3, 3))
    for start in range(0, len(next_ids), 32768):
        ids = next_ids[start:start+32768]
        points = _node_points(r, np.stack(np.unravel_index(ids, node_shape), 1), dim)
        normal = plain.normals_and_candidates(points, radius)[0]
        values = media.volume_average(points, window, r.subpixel_quadrature, normal, table)
        projector = normal[:, :, None]*normal[:, None, :]
        next_tensors[start:start+len(ids)] = projector*values[:, -1, None, None]+(np.eye(3)-projector)/values[:, -2, None, None]
    # Edges of MIXED nodes and of WHOLE nodes on the surface: the candidates for D-driven samples.
    size = int(np.prod(shape))*3
    driven = np.zeros(size, bool); standard = []; owners = []
    wholes = {}
    for ids in (np.flatnonzero(kind == MIXED), np.flatnonzero(boundary)):
        coords = np.stack(np.unravel_index(ids, node_shape), 1)
        for axis in range(3):
            for side in range(2):
                sample, good, _ = _edge(r, coords, axis, side, dim, periodic)
                sample = sample[good]
                ends = np.stack(np.unravel_index(sample//3, r.shape), 1)
                lower = np.ravel_multi_index(np.where(np.arange(3) < dim, ends, 0).T, node_shape)
                upper_coords = ends.copy()
                if axis < dim:
                    upper_coords[:, axis] += 1
                    if axis in periodic:
                        upper_coords[:, axis] %= shape[axis]
                upper_coords[:, dim:] = 0
                upper = np.ravel_multi_index(upper_coords.T, node_shape)
                both = (kind[lower] == WHOLE) & (kind[upper] == WHOLE)
                driven[sample[~both]] = True
                standard.append(sample[both]); owners.append(material[lower][both])
                for s, lo, hi in zip(sample[~both], lower[~both], upper[~both]):
                    wholes[int(s)] = (lo, hi)
    standard = np.concatenate(standard) if standard else np.empty(0, np.int64)
    owners = np.array([dispersive[k] for k in np.concatenate(owners)], np.int32) if len(standard) else np.empty(0, np.int32)
    standard, first = np.unique(standard, return_index=True); owners = owners[first]
    instantaneous = np.zeros(size)
    present = np.full(size, -1, np.int32)
    groups = []
    for k, i in enumerate(dispersive):
        m = materials[i]
        # Half-dispersive edges: each WHOLE end contributes 1/2 D/eps_m.
        half_index, half_weight = [], []
        for s, (lo, hi) in wholes.items():
            weight = .5*((kind[lo] == WHOLE and material[lo] == k)+(kind[hi] == WHOLE and material[hi] == k))
            if weight:
                half_index.append(s); half_weight.append(weight)
        half_index = np.array(half_index, np.int64); half_weight = np.array(half_weight)
        np.add.at(instantaneous, half_index, half_weight/m.epsilon_inf)
        present[half_index] = i
        ids = np.flatnonzero((kind == MIXED) & (material == k))
        coords = np.stack(np.unravel_index(ids, node_shape), 1)
        edges = np.zeros((3, 2, len(ids)), np.int64); phases = np.ones((3, 2, len(ids)), complex)
        for axis in range(3):
            for side in range(2):
                sample, good, phase = _edge(r, coords, axis, side, dim, periodic)
                if not np.all(good):
                    raise ValueError(f'Dispersive subpixel interfaces need at least one averaging window ({WINDOW:g} cells) between a dispersive surface '
                                     'and a nonperiodic grid boundary. Enlarge the region or choose staircase interfaces.')
                edges[axis, side] = sample; phases[axis, side] = phase
        present[edges.reshape(-1)] = i
        f = np.array([cells[int(n)][3] for n in ids])
        C = np.array([cells[int(n)][0] for n in ids]); B = np.array([cells[int(n)][1] for n in ids])
        normal = np.array([cells[int(n)][2] for n in ids]).reshape(-1, 3)
        # High-frequency share of each edge: 1/2 Z_aa at the instantaneous permittivity.
        z = normal**2*(B+f/m.epsilon_inf)[:, None]+(1-normal**2)/(C+f*m.epsilon_inf)[:, None]
        for axis in range(3):
            for side in range(2):
                np.add.at(instantaneous, edges[axis, side], .5*z[:, axis])
        groups.append(dict(material=i, epsilon_inf=m.epsilon_inf, oscillators=[tuple(map(float, o)) for o in m.oscillators],
                           half_index=half_index, half_weight=half_weight, edges=edges,
                           phases=phases if r.complex_fields else None, fraction=f, arithmetic=C, harmonic=B, normal=normal))
    metadata = dict(method='node_tensor_dispersive_laminate', materials=[materials[i].name for i in dispersive],
                    mixed_nodes=int(np.count_nonzero(kind == MIXED)), whole_surface_nodes=int(np.count_nonzero(boundary)),
                    next_nodes=int(len(next_ids)), driven_samples=int(np.count_nonzero(driven)), standard_surface_samples=int(len(standard)),
                    device_bytes=state_bytes(r, [(len(g['fraction']), len(g['half_index']), len(g['oscillators'])) for g in groups],
                                             int(np.count_nonzero(driven)), size//3),
                    preparation_seconds=time.perf_counter()-started)
    active = (kind == MIXED) | (kind == NEXT) | boundary
    return DispersiveTensor(node_shape, kind, active, next_ids, next_tensors, driven, present[driven], standard, owners, instantaneous,
                            groups, metadata)


def state_bytes(region, groups, driven, cells):
    """Device bytes of DispersiveState and its update: per group (MIXED nodes, half-dispersive edges, poles), the
    driven samples and the full-volume energy weights of the state norm."""
    real = 8 if region.precision == 'float64' else 4; field = real*(2 if region.complex_fields else 1)
    total = 3*real*cells+driven*(field+real+8)
    for nodes, halves, poles in groups:
        # Ten node branches: state x, P, Q and energy weights per sample, coefficients eps, k_sum, a, d, k per
        # branch; edges, phases and node geometry; the temporaries of one update.
        node = (10*field*(1+2*poles)+10*real*(1+2*poles)+4*real*(2+3*poles)+6*8+8*real+(6*field if region.complex_fields else 0)
                +field*(30+6*(1+poles)))
        total += nodes*node+halves*(field*(1+2*poles)+real*(1+2*poles)+real*(2+3*poles)+8+real)
    return total


def estimated_bytes(project):
    """An upper bound of state_bytes before planning: every node within one window of the bounding box of a
    dispersive object counted as a MIXED node with the most poles, with three driven samples each."""
    r = project.region
    poles = {m.name: len(m.oscillators) for m in project.materials if m.oscillators}
    dim = 2 if r.dimension == '2d' else 3
    steps = r.axis_steps
    nodes, most = 0, 0
    for s in project.structures:
        if not s.enabled or s.material not in poles:
            continue
        center, size = object_bounds(s)
        count = 1
        for a in range(dim):
            lo = max(center[a]-size[a]/2-WINDOW*steps[a], -r.size[a]/2); hi = min(center[a]+size[a]/2+WINDOW*steps[a], r.size[a]/2)
            count *= max(0, int(np.floor((hi-lo)/steps[a]))+2)
        nodes += count; most = max(most, poles[s.material])
    if not most:
        return 0
    nodes = min(nodes, int(np.prod([n+1 for n in r.shape[:dim]])))
    return state_bytes(r, [(nodes, 0, most)], 3*nodes, int(np.prod(r.shape)))


def refuse_driven_writers(tensor, project, terms, epsilon, ownership):
    """Refuse every E write on a D-driven sample, whose E is assigned from D each step: soft E source terms, and
    the TFSF and one-way neighbourhoods, checked with the driven samples marked by their material."""
    from .injection import validate_oneway_materials
    from .tfsf import validate_tfsf_materials
    present = np.array(ownership, copy=True)
    present.reshape(-1)[np.flatnonzero(tensor.driven)] = tensor.driven_owners
    validate_oneway_materials(project, epsilon, present)
    validate_tfsf_materials(project, epsilon, present)
    driven = tensor.driven.reshape(tuple(project.region.shape)+(3,))
    for field, loc, *_ in terms:
        if field[0] == 'E' and np.any(driven[tuple(loc)+('xyz'.index(field[1].lower()),)]):
            raise ValueError(f'A soft {field} source overlaps node cells cut by a dispersive surface (dispersive subpixel interfaces). '
                             'Move it at least one cell away from the surface or choose staircase interfaces.')


class DispersiveState:
    """D at the D-driven samples and the branches of their E: half-dispersive edges and MIXED node tensors, per material.

    ``plan`` is the SubpixelPlan: its update diagonal and its sparse rows at driven samples give the static
    share of E, and those rows couple only driven samples."""
    def __init__(self, grid, tensor, plan):
        self.torch = grid.is_torch
        device = grid.E.device if self.torch else None
        index = (lambda a: torch.as_tensor(a, device=device, dtype=torch.long)) if self.torch else (lambda a: np.asarray(a, np.int64))
        field = (lambda a: torch.as_tensor(a, device=device, dtype=grid.E.dtype)) if self.torch else (lambda a: np.asarray(a).astype(grid.E.dtype))
        self.grid = grid
        samples = np.flatnonzero(tensor.driven)
        position = lambda a: np.searchsorted(samples, a)
        self.samples = index(samples)
        self.D = grid._zeros((len(samples),))
        grid.memory_states.append(self.D)
        self.static = grid._coefficient(plan.update_diagonal.reshape(-1)[samples])
        rows = tensor.driven[plan.rows]
        columns, values = plan.columns[rows], plan.values[rows]
        coupled = np.minimum(position(columns), max(len(samples)-1, 0))
        if len(samples) and np.any(samples[coupled][values != 0] != columns[values != 0]):
            raise RuntimeError('A D-driven sample couples statically to an E-updated sample.')
        self.rows, self.columns, self.values = index(position(plan.rows[rows])), index(coupled), field(values)
        self.groups = []
        self.scales = []
        for g in tensor.groups:
            e, poles = g['epsilon_inf'], g['oscillators']
            n = len(g['fraction']); f, C, B = g['fraction'], g['arithmetic'], g['harmonic']
            parallel = [(w0, f*s, gamma) for w0, s, gamma in poles]
            group = dict(half_index=index(position(g['half_index'])), half_weight=grid._coefficient(g['half_weight']),
                         half=Branch(grid, e, poles, (len(g['half_index']),)), nodes=n)
            if n:
                group.update(edges=[[index(position(g['edges'][a, s])) for s in range(2)] for a in range(3)],
                             phases=None if g['phases'] is None else [[field(g['phases'][a, s]) for s in range(2)] for a in range(3)],
                             fraction=grid._coefficient(f), harmonic=grid._coefficient(B),
                             normal=[grid._coefficient(g['normal'][:, a]) for a in range(3)],
                             square=[grid._coefficient(g['normal'][:, a]**2) for a in range(3)],
                             across=Branch(grid, e, poles, (n,)), along=Branch(grid, C+f*e, parallel, (3, n)),
                             across_diagonal=Branch(grid, e, poles, (3, n)), along_diagonal=Branch(grid, C+f*e, parallel, (3, n)))
            self.groups.append(group)
            # Node energy D^T Z D + sum_a Z_aa delta_a^2: each branch weighed by its share of Z.
            square = g['normal'].T**2
            self.scales.append(dict(half=g['half_weight'], across=f, along=np.ones((3, n)), across_diagonal=square*f, along_diagonal=1-square))

    def apply(self, curl):
        """Advance D by courant * curl(H) and assign E at the driven samples from D and the branches."""
        g = self.grid
        flat = curl.reshape(-1)
        if self.torch:
            self.D += g.courant_number*flat.index_select(0, self.samples)
        else:
            self.D += g.courant_number*flat[self.samples]
        D = self.D
        take = (lambda i: D.index_select(0, i)) if self.torch else (lambda i: D[i])
        stack = torch.stack if self.torch else np.stack
        out = self.static*D
        if len(self.rows):
            self._add(out, self.rows, (self.values*D[self.columns]).sum(1))
        for group in self.groups:
            if len(group['half_index']):
                self._add(out, group['half_index'], group['half_weight']*group['half'].advance(take(group['half_index'])))
            if not group['nodes']:
                continue
            edges, phases = group['edges'], group['phases']
            local = [[take(edges[a][s])*(1 if phases is None else phases[a][s]) for s in range(2)] for a in range(3)]
            mean = stack([(local[a][0]+local[a][1])/2 for a in range(3)]); half = stack([(local[a][0]-local[a][1])/2 for a in range(3)])
            n = group['normal']
            normal_d = n[0]*mean[0]+n[1]*mean[1]+n[2]*mean[2]
            across = group['fraction']*group['across'].advance(normal_d)+group['harmonic']*normal_d
            along = group['along'].advance(stack([mean[a]-n[a]*normal_d for a in range(3)]))
            across_diagonal = group['across_diagonal'].advance(half); along_diagonal = group['along_diagonal'].advance(half)
            for a in range(3):
                share = n[a]*across+along[a]
                difference = group['square'][a]*(group['fraction']*across_diagonal[a]+group['harmonic']*half[a])+(1-group['square'][a])*along_diagonal[a]
                for s, sign in ((0, 1), (1, -1)):
                    value = (share+sign*difference)/2
                    self._add(out, edges[a][s], value if phases is None else value*phases[a][s].conj())
        if self.torch:
            g.E.reshape(-1).index_copy_(0, self.samples, out)
        else:
            g.E.reshape(-1)[self.samples] = out

    def _add(self, field, index, values):
        if self.torch:
            field.index_add_(0, index, values)
        else:
            np.add.at(field, index, values)

    def energy_terms(self):
        terms = []
        for group, scales in zip(self.groups, self.scales):
            for name in ('half', 'across', 'along', 'across_diagonal', 'along_diagonal'):
                if name in group:
                    terms += group[name].energy_terms(self.grid, scales[name])
        return terms
