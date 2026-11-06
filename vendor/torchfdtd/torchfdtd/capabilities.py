"""Capability registry: the admitted and rejected combinations of the solver contract.

The registry is data. AXES lists the nine contract axes and their values. Every
combination of one value per axis is realized as one small Project by
``example_project`` (the scene recipe below) and executed through one entry
point per execution mode (the LANES). RULES is the ordered list of rejections
the code performs, each with the code path that raises and the exact message
prefix; ``verdict`` walks the rules in code order, so the first matching rule
is the error a caller sees, and a combination that matches no rule is admitted
through the first matching lane. ``tests/test_capability_pairs.py`` builds a
pairwise covering array over the axes, runs every admitted case and asserts
every rejected case raises the registry's error from the registry's code path
before any field allocation. The tables in docs/CAPABILITIES.md and the
``combinations`` block of ``/api/capabilities`` are rendered from this module by
scripts/build_capability_tables.py. docs/FEATURE_CHECKLIST.md is a different
inventory (Lumerical properties) and is only linked from here.

Scene recipe (one Project per combination):

* Region 2.4 x 2.4 um transverse and 2.4 um along the propagation axis d
  (z in 3D, y in 2D), mesh 0.1 um, 3 PML cells, 12 steps, Yee material
  sampling, staircase interfaces. ``graded`` refines the mesh around the
  structure (mesh_max 0.2 um, 6 points per wavelength); ``explicit`` freezes
  those graded nodes as fixed coordinates.
* The material sits in one block of 0.45 x 0.45 x 0.45 um at the centre; mode
  scenes (mode source or mode-port monitor) stretch it into a straight guide
  along d instead.
  ``dielectric`` is a constant index 2; ``dispersive_ade`` a two-pole
  Drude/Lorentz multipole material; ``anisotropic_tensor`` a symmetric tensor
  material; ``tensor_ade`` the same tensor material whose Lorentz pole is
  passed to the tensor-ADE API as explicit pole tensors; ``pec`` an ideal
  conductor material, which the schema does not offer.
* The boundary kind applies to every active face. Directional scenes (sheet,
  one-way, TFSF, mode and tiled-sheet sources, plane, mode-port and
  radiation-box monitors, tiled execution) carry CPML on the two d faces and
  the boundary kind on the transverse faces. CPML faces use alpha = 0 so that the PMC
  endpoint profile contract can accept them. Bloch faces carry a phase of
  0.3 rad on their axis.
* Sources are Gaussian, 1 um carrier, two cycles. ``point`` is an Ez point
  source; ``sheet`` a soft plane through the interior; ``plane_oneway`` a
  one-way plane over the whole transverse cell; ``tfsf`` a 0.8 um box;
  ``mode`` the soft plane template of a fixed-eigenmode launch;
  ``tiled_sheet`` a soft plane extended through the lateral PML.
* Monitors: ``point`` one Ez probe; ``plane_dft`` one d-normal frequency plane
  over the transverse interior (one cell short of PEC/PMC walls);
  ``mode_port`` two opposing fixed mode ports (no project monitors);
  ``radiation_box`` six frequency planes on a closed 0.8 um box, projected
  to the far field after the run.
* ``precision`` is Region.precision; ``cpu`` is backend cpu, ``cuda_torch`` is
  backend cuda with the Torch kernels, ``cuda_fused`` backend cuda with the
  fused CuPy kernels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import math
import re

AXES = {
    'dimension': ('2d', '3d'),
    'mesh': ('uniform', 'graded', 'explicit'),
    'material': ('dielectric', 'dispersive_ade', 'anisotropic_tensor', 'tensor_ade', 'pec'),
    'boundary': ('cpml', 'periodic', 'bloch', 'pec', 'antisymmetric', 'pmc', 'symmetric'),
    'source': ('point', 'sheet', 'plane_oneway', 'tfsf', 'mode', 'tiled_sheet'),
    'monitor': ('point', 'plane_dft', 'mode_port', 'radiation_box'),
    'execution': ('forward', 'checkpointed_adjoint', 'reversible_adjoint', 'streamed', 'streamed_adjoint',
                  'tensor_batch', 'tiled'),
    'precision': ('float32', 'float64'),
    'backend': ('cpu', 'cuda_torch', 'cuda_fused'),
}

AXIS_LABELS = {
    'dimension': 'Dimension', 'mesh': 'Mesh', 'material': 'Material', 'boundary': 'Boundaries',
    'source': 'Source', 'monitor': 'Monitor', 'execution': 'Execution', 'precision': 'Precision', 'backend': 'Backend',
}

VALUE_LABELS = {
    'dimension': {'2d': '2D', '3d': '3D'},
    'mesh': {'uniform': 'uniform', 'graded': 'graded', 'explicit': 'explicit nodes'},
    'material': {'dielectric': 'dielectric', 'dispersive_ade': 'dispersive ADE', 'anisotropic_tensor': 'anisotropic tensor',
                 'tensor_ade': 'tensor ADE', 'pec': 'PEC material'},
    'boundary': {'cpml': 'CPML', 'periodic': 'periodic', 'bloch': 'Bloch', 'pec': 'PEC', 'antisymmetric': 'antisymmetric',
                 'pmc': 'PMC', 'symmetric': 'symmetric'},
    'source': {'point': 'point', 'sheet': 'sheet', 'plane_oneway': 'plane one-way', 'tfsf': 'TFSF', 'mode': 'mode',
               'tiled_sheet': 'tiled sheet'},
    'monitor': {'point': 'point', 'plane_dft': 'plane DFT', 'mode_port': 'mode port', 'radiation_box': 'radiation box'},
    'execution': {'forward': 'forward', 'checkpointed_adjoint': 'checkpointed adjoint',
                  'reversible_adjoint': 'reversible adjoint', 'streamed': 'streamed', 'streamed_adjoint': 'streamed adjoint',
                  'tensor_batch': 'tensor batch', 'tiled': 'tiled'},
    'precision': {'float32': 'float32', 'float64': 'float64'},
    'backend': {'cpu': 'CPU', 'cuda_torch': 'CUDA Torch', 'cuda_fused': 'CUDA fused'},
}

DIRECTIONAL_SOURCES = frozenset({'sheet', 'plane_oneway', 'tfsf', 'mode', 'tiled_sheet'})
DIRECTIONAL_MONITORS = frozenset({'plane_dft', 'mode_port', 'radiation_box'})
WALLS = frozenset({'pec', 'antisymmetric', 'pmc', 'symmetric'})
TENSOR_MATERIALS = frozenset({'anisotropic_tensor', 'tensor_ade'})
STREAMED_EXECUTIONS = frozenset({'streamed', 'streamed_adjoint'})

WAVELENGTH_UM = 1.0
MESH_UM = 0.1
PML_CELLS = 3
STEPS = 12
BLOCH_PHASE = 0.3
TRANSVERSE_UM = 2.4
LONGITUDINAL_UM = 2.4
BLOCK_UM = 0.45
BOX_UM = 0.8
PORT_UM = 0.3
PORT_SOURCE_UM = 0.6
TILE_UM = 0.9            # tests/test_capability_pairs.py plan_tiles arguments
TILE_OVERLAP_UM = 0.2
TENSOR = (2.4, 2.1, 2.7, 0.15, 0.0, 0.1)
POLE = dict(resonance_rad_s=2.5e15, strength_rad_s_squared=1.5e30, damping_rad_s=3e14)
DRUDE = dict(resonance_rad_s=0.0, strength_rad_s_squared=8e29, damping_rad_s=4e14)


def combinations():
    """Every combination of one value per axis, in axis order."""
    names = list(AXES)
    for values in product(*(AXES[name] for name in names)):
        yield dict(zip(names, values))


def combination_count():
    return math.prod(len(v) for v in AXES.values())


def config_id(config):
    return '/'.join(config[name] for name in AXES)


def validate_config(config):
    if set(config) != set(AXES):
        raise ValueError(f'A configuration needs exactly the axes {tuple(AXES)}.')
    for name, value in config.items():
        if value not in AXES[name]:
            raise ValueError(f'{name}={value!r} is not one of {AXES[name]}.')


def propagation_axis(config):
    return 2 if config['dimension'] == '3d' else 1


def active_axes(config):
    return (0, 1, 2) if config['dimension'] == '3d' else (0, 1)


def transverse_axes(config):
    d = propagation_axis(config)
    return tuple(a for a in active_axes(config) if a != d)


def is_directional(config):
    return (config['source'] in DIRECTIONAL_SOURCES or config['monitor'] in DIRECTIONAL_MONITORS
            or config['execution'] == 'tiled')


def face_kinds(config):
    """Boundary kind of every face, x_min ... z_max, as the recipe assigns them."""
    kind = {'cpml': 'pml'}.get(config['boundary'], config['boundary'])
    d = propagation_axis(config)
    kinds = {}
    for axis, name in enumerate('xyz'):
        for side in ('min', 'max'):
            if axis not in active_axes(config):
                kinds[f'{name}_{side}'] = 'pml'
            elif axis == d and is_directional(config):
                kinds[f'{name}_{side}'] = 'pml'
            else:
                kinds[f'{name}_{side}'] = kind
    return kinds


def transverse_kind(config):
    """The kind actually placed on the transverse faces (the boundary value, as 'pml' for cpml)."""
    return {'cpml': 'pml'}.get(config['boundary'], config['boundary'])


def has_cpml(config):
    return 'pml' in set(face_kinds(config).values())


@dataclass(frozen=True)
class Rule:
    """One rejection: the first matching rule in RULES is the error the code raises."""
    name: str
    when: dict
    stage: str            # schema | entry | run | post
    code_path: str        # torchfdtd/<module>.py::<function or Class.method>
    message: str          # exact prefix of str(exception)
    exception: str = 'ValueError'
    note: str = ''
    executions: tuple = ()  # empty means every execution
    format: dict = field(default_factory=dict)   # values of the message placeholders the code interpolates

    def matches(self, config):
        return all(config[axis] in values for axis, values in self.when.items())

    def expected_message(self, config):
        """The message prefix for one combination: {source} and {normal} come from the recipe."""
        values = dict(self.format, source=_source_name(config), normal='xyz'[propagation_axis(config)])
        return re.sub(r'\{(\w+)\}', lambda m: values[m.group(1)], self.message)


@dataclass(frozen=True)
class Lane:
    """One admitted entry point: the first matching lane executes an admitted combination."""
    name: str
    when: dict
    code_path: str
    test: str
    description: str
    note: str = ''

    def matches(self, config):
        return all(config[axis] in values for axis, values in self.when.items())


@dataclass(frozen=True)
class Verdict:
    status: str                 # admitted | rejected
    name: str                   # rule or lane name
    code_path: str
    message: str = ''
    exception: str = ''
    stage: str = ''
    test: str = ''
    note: str = ''

    def as_dict(self):
        return dict(status=self.status, name=self.name, code_path=self.code_path, message=self.message,
                    exception=self.exception, stage=self.stage, test=self.test, note=self.note)


def _s(*values):
    return frozenset(values)


# The rules below are ordered as the code checks them: the Project schema
# (torchfdtd/models.py and the validators it calls), then the constructor of
# the lane's entry point, then its run-time admission, then post-processing.
# `executions` narrows a rule to the lanes whose entry point performs the check.
RULES: list = []
LANES: list = []


def rule(name, when, stage, code_path, message, **extra):
    RULES.append(Rule(name, {k: frozenset(v) for k, v in when.items()}, stage, code_path, message, **extra))


def lane(name, when, code_path, test, description, **extra):
    LANES.append(Lane(name, {k: frozenset(v) for k, v in when.items()}, code_path, test, description, **extra))


def verdict(config):
    validate_config(config)
    for item in RULES:
        if item.executions and config['execution'] not in item.executions:
            continue
        if item.matches(config):
            return Verdict('rejected', item.name, item.code_path, item.message, item.exception, item.stage, note=item.note)
    for item in LANES:
        if item.matches(config):
            return Verdict('admitted', item.name, item.code_path, test=item.test, note=item.note, stage='lane')
    raise LookupError(f'No lane admits {config_id(config)}; the registry is incomplete.')


def registry_summary():
    """Counts over the full product: admitted and rejected combinations, per rule and lane."""
    counts = {'admitted': 0, 'rejected': 0}
    by_name = {}
    for config in combinations():
        v = verdict(config)
        counts[v.status] += 1
        by_name[v.name] = by_name.get(v.name, 0)+1
    return dict(total=combination_count(), **counts, by_name=by_name,
                rules=len(RULES), lanes=len(LANES),
                unused=[r.name for r in RULES if r.name not in by_name]+[l.name for l in LANES if l.name not in by_name])


def pair_tables(limit=4):
    """For every axis pair and value pair: admitted when any full combination is admitted.

    A cell counts the admitted combinations containing its two values over all of them and
    lists the most frequent lanes (admitted) and rules (rejected), at most `limit` each. Its
    `reason` is the rule that explains the pair: the most frequent rule whose condition (its
    `when` axes, and the execution axis when it is restricted to executions) names both axes,
    else one axis, else any.
    """
    by_name = {r.name: r for r in RULES}
    names = list(AXES)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i+1:]]
    cells = {(a, b, va, vb): [0, 0, {}, {}] for a, b in pairs for va in AXES[a] for vb in AXES[b]}
    for config in combinations():
        v = verdict(config)
        for a, b in pairs:
            cell = cells[a, b, config[a], config[b]]
            cell[1] += 1
            if v.status == 'admitted':
                cell[0] += 1
                cell[2][v.name] = cell[2].get(v.name, 0)+1
            else:
                cell[3][v.name] = cell[3].get(v.name, 0)+1
    tables = {}
    for a, b in pairs:
        table = {}
        for va in AXES[a]:
            for vb in AXES[b]:
                admitted, total, lanes, rules = cells[a, b, va, vb]
                def specificity(item):
                    rule = by_name[item[0]]
                    axes = set(rule.when)|({'execution'} if rule.executions else set())
                    return (a in axes)+(b in axes), item[1]
                reason = max(rules.items(), key=specificity)[0] if rules and not admitted else None
                table[f'{va}|{vb}'] = dict(status='admitted' if admitted else 'rejected', admitted=admitted, total=total,
                                          reason=reason,
                                          lanes=dict(sorted(lanes.items(), key=lambda kv: -kv[1])[:limit]),
                                          rules=dict(sorted(rules.items(), key=lambda kv: -kv[1])[:limit]))
        tables[f'{a}|{b}'] = table
    return tables


# Incidence definition of a plane source (docs/CONVENTIONS.md section 3, docs/SOURCES.md). The
# solver realizes one definition: a soft sheet on a Bloch cell carries the spatial phase
# exp(+i phi (x - x0) / L), so k_parallel = phi / L is fixed and the incidence angle
# asin(k_parallel / (n k0)) varies across the band. A fixed-angle broadband source (the
# vendor's BFAST) has no code path; the source preview and the workbench refuse it with the
# message below instead of drawing it, and the feature inventory rows named here stay 'missing'.
INCIDENCE = {
    'normal': dict(status='admitted', code_path='torchfdtd/solver.py::source_profile',
                   description='No Bloch phase on a transverse axis: k_parallel = 0 at every frequency.'),
    'fixed_k_parallel': dict(status='admitted', code_path='torchfdtd/solver.py::source_profile',
                             description='Bloch phase phi on a transverse axis of length L: k_parallel = phi / L is fixed and '
                                         'the angle asin(k_parallel / (n k0)) varies across the band; frequencies with '
                                         'k_parallel > n k0 are evanescent.'),
    'fixed_angle': dict(status='rejected', code_path='torchfdtd/source_preview.py::preview_source', exception='ValueError',
                        feature_inventory=('source.angle', 'boundary.bfast'),
                        message='Fixed-angle broadband injection is not implemented: a Bloch cell fixes k_parallel, so the '
                                'incidence angle varies across the band (feature inventory source.angle and boundary.bfast '
                                'are missing). Preview the fixed-k_parallel source instead.'),
}


def registry_json():
    rules = [dict(name=r.name, when={k: sorted(v) for k, v in r.when.items()}, stage=r.stage, code_path=r.code_path,
                  message=r.message, format=r.format, exception=r.exception, executions=list(r.executions), note=r.note)
             for r in RULES]
    lanes = [dict(name=l.name, when={k: sorted(v) for k, v in l.when.items()}, code_path=l.code_path, test=l.test,
                  description=l.description, note=l.note) for l in LANES]
    summary = registry_summary()
    return dict(schema_version=1, kind='capability_registry', axes={k: list(v) for k, v in AXES.items()},
                axis_labels=AXIS_LABELS, value_labels=VALUE_LABELS, recipe=__doc__.split('Scene recipe (one Project per combination):', 1)[1].strip(),
                summary={k: v for k, v in summary.items() if k != 'by_name'}, rules=rules, lanes=lanes,
                pairs=pair_tables(), incidence=INCIDENCE,
                related=dict(feature_checklist='docs/FEATURE_CHECKLIST.md', tables='docs/CAPABILITIES.md',
                             tests='tests/test_capability_pairs.py'))



# ---- Stage 1: combinations no entry point can express (the test verifies the schema field or signature) ----
# These precede the schema because the caller never builds a Project for them.
rule('mode_source_unavailable', dict(source=['mode']), 'unavailable', 'torchfdtd/models.py::Source.kind',
     'The Project schema offers point, plane and TFSF sources; fixed-eigenmode launches exist only as inputs of '
     'ModeInjectedPlaneSimulation and ModeNetwork.', exception='Unavailable',
     executions=('reversible_adjoint', 'tensor_batch', 'tiled'))
rule('mode_port_unavailable', dict(monitor=['mode_port']), 'unavailable', 'torchfdtd/models.py::Monitor.kind',
     'The Project schema offers point and field monitors; mode ports exist only as ModeNetwork inputs.',
     exception='Unavailable', executions=('reversible_adjoint', 'tensor_batch', 'tiled'))
rule('tensor_ade_unavailable', dict(material=['tensor_ade']), 'unavailable', 'torchfdtd/models.py::Material.model',
     'The Project schema offers no tensor ADE material; tensor poles exist only as TensorDispersiveSimulation inputs.',
     exception='Unavailable', executions=('reversible_adjoint', 'tensor_batch', 'tiled'))
rule('tensor_ade_streamed_unavailable', dict(material=['tensor_ade']), 'unavailable',
     'torchfdtd/streamed_tensor.py::StreamedTensorSimulation.forward',
     'The streamed tensor path takes node tensors only; it has no pole inputs.', exception='Unavailable',
     executions=('streamed', 'streamed_adjoint'))


# ---- Stage 2: the Project schema (torchfdtd/models.py and the validators it calls) ----
T = ('anisotropic_tensor', 'tensor_ade')
NOT_CPML = ('periodic', 'bloch', 'pec', 'antisymmetric', 'pmc', 'symmetric')
rule('material_pec_schema', dict(material=['pec']), 'schema', 'torchfdtd/models.py::Material',
     "Input should be 'dielectric', 'tensor', 'drude', 'lorentz' or 'multipole'", exception='ValidationError',
     note='No ideal-conductor material model exists; PEC enters as a boundary face only.')
rule('tensor_project_2d', dict(material=T, dimension=['2d']), 'schema', 'torchfdtd/tensor_project.py::validate_tensor_project',
     'Native tensor materials require FP32 material on uniform 3D grids.', exception='ValidationError')
rule('tensor_project_fp64', dict(material=T, precision=['float64']), 'schema', 'torchfdtd/tensor_project.py::validate_tensor_project',
     'Native tensor materials require FP32 material on uniform 3D grids.', exception='ValidationError',
     note='TensorDielectricSimulation also takes FP64 node tensors on a project that declares no tensor material (diagnostics).')
rule('tensor_project_mesh', dict(material=T, mesh=['graded', 'explicit']), 'schema', 'torchfdtd/tensor_project.py::validate_tensor_project',
     'Native tensor materials require FP32 material on uniform 3D grids.', exception='ValidationError')
rule('tensor_project_faces', dict(material=T, boundary=['pec', 'antisymmetric', 'pmc', 'symmetric']), 'schema',
     'torchfdtd/tensor_project.py::validate_tensor_project',
     'Native tensor faces must be periodic/Bloch or isotropic-exterior PML.', exception='ValidationError',
     note='The explicit node-tensor APIs accept PEC walls on a project that declares no tensor material.')
rule('tensor_project_sources', dict(material=T, source=['sheet', 'plane_oneway', 'tfsf', 'mode', 'tiled_sheet']), 'schema',
     'torchfdtd/tensor_project.py::validate_tensor_project',
     'Native tensor sources must be point soft electric field increments.', exception='ValidationError')
rule('tensor_project_monitors', dict(material=T, monitor=['plane_dft', 'mode_port', 'radiation_box']), 'schema',
     'torchfdtd/tensor_project.py::validate_tensor_project',
     'Native tensor requires point monitors at every timestep.', exception='ValidationError')
rule('pmc_2d_schema', dict(boundary=['pmc', 'symmetric'], dimension=['2d']), 'schema',
     'torchfdtd/endpoint_native.py::validate_pmc_project',
     'PMC/symmetric faces require a 3D region; two-dimensional PMC walls are not implemented by any path.',
     exception='ValidationError')
rule('tfsf_needs_pml_everywhere', dict(source=['tfsf'], boundary=NOT_CPML), 'schema', 'torchfdtd/tfsf.py::tfsf_plan',
     '{source}: the current isolated TFSF box requires PML on every active face.', exception='ValidationError')
rule('oneway_needs_periodic_transverse', dict(source=['plane_oneway'], boundary=['cpml', 'bloch', 'pec', 'antisymmetric', 'pmc', 'symmetric']),
     'schema', 'torchfdtd/injection.py::oneway_plan',
     '{source}: normal-incidence planes require periodic transverse boundaries (zero Bloch phase).', exception='ValidationError')
rule('sheet_writes_pec_wall', dict(source=['mode', 'tiled_sheet'], boundary=['pec', 'antisymmetric'], dimension=['3d']), 'schema',
     'torchfdtd/models.py::Project.valid_scene',
     '{source}: source writes a constrained PEC/antisymmetric wall component', exception='ValidationError',
     note='A sheet spanning the whole cell reaches the tangential Ex on the lower y PEC wall; in 2D Ex is normal to the x walls.')
rule('radiation_box_2d_schema', dict(monitor=['radiation_box'], dimension=['2d']), 'schema', 'torchfdtd/models.py::Project.valid_scene',
     'A 2D flux monitor must be x-normal or y-normal.', exception='ValidationError')

# ---- Stage 3: entry points (constructors), in the order each one checks ----
RESIDENT = ('forward', 'checkpointed_adjoint')
STREAMED = ('streamed', 'streamed_adjoint')
DIFF = ('forward', 'checkpointed_adjoint', 'streamed', 'streamed_adjoint')
# Mode networks (mode_port monitor) and modal launches (mode source)
rule('mode_network_streamed', dict(monitor=['mode_port']), 'entry', 'torchfdtd/mode_network.py::ModeNetwork.__init__',
     'Mode networks require resident AdjointOptions, not streaming.', executions=STREAMED)
MODAL = dict(source=['mode'])
PORT = dict(monitor=['mode_port'])
for target, when in (('port', PORT), ('modal', MODAL)):
    periodic_side = dict(when, boundary=['periodic', 'bloch', 'pec', 'antisymmetric', 'pmc', 'symmetric'])
    rule(f'{target}_periodic_launch_region', dict(periodic_side, dimension=['2d']), 'entry',
         'torchfdtd/mode_injection.py::prepare_modal_launch', 'Modal launch requires a real uniform 3D Yee-sampled region.',
         executions=DIFF)
    rule(f'{target}_periodic_launch_mesh', dict(periodic_side, mesh=['graded', 'explicit']), 'entry',
         'torchfdtd/mode_injection.py::prepare_modal_launch', 'Modal launch requires a real uniform 3D Yee-sampled region.',
         executions=DIFF)
    rule(f'{target}_periodic_launch_bloch', dict(periodic_side, boundary=['bloch']), 'entry',
         'torchfdtd/mode_injection.py::prepare_modal_launch', 'Modal launch requires a real uniform 3D Yee-sampled region.',
         executions=DIFF, note='Bloch faces make the fields complex; the modal launch is real.')
    rule(f'{target}_periodic_launch_materials', dict(periodic_side, material=['dispersive_ade']), 'entry',
         'torchfdtd/mode_injection.py::prepare_modal_launch', 'Only real nondispersive staircase materials are supported.',
         executions=DIFF)
    if target == 'port':
        rule('port_periodic_launch_template', dict(periodic_side, source=['point', 'plane_oneway', 'tfsf']), 'entry',
             'torchfdtd/mode_injection.py::prepare_modal_launch', 'Use one soft Gaussian plane source with cycle-based timing.',
             executions=DIFF)
    rule(f'{target}_periodic_launch_transverse', dict(periodic_side, boundary=['pec', 'antisymmetric', 'pmc', 'symmetric']), 'entry',
         'torchfdtd/injection.py::oneway_plan',
         '{source}: normal-incidence planes require periodic transverse boundaries (zero Bloch phase).', executions=DIFF)
    open_side = dict(when, boundary=['cpml'])
    if target == 'port':
        rule('port_open_launch_tfsf_template', dict(open_side, source=['tfsf']), 'schema',
             'torchfdtd/tfsf.py::tfsf_plan', '{source}: leave at least two interior cells between each TFSF face and PML.',
             exception='ValidationError', executions=DIFF,
             note='ModeNetwork moves the timing source to the port source plane and revalidates the copy (open_mode_injection._open_geometry) before checking its kind.')
    rule(f'{target}_open_launch_region', dict(open_side, dimension=['2d']), 'entry',
         'torchfdtd/open_mode_injection.py::_open_geometry', 'Open modal launch requires real FP32 uniform 3D staircase Yee sampling.',
         executions=DIFF)
    rule(f'{target}_open_launch_mesh', dict(open_side, mesh=['graded', 'explicit']), 'entry',
         'torchfdtd/open_mode_injection.py::_open_geometry', 'Open modal launch requires real FP32 uniform 3D staircase Yee sampling.',
         executions=DIFF)
    rule(f'{target}_open_launch_fp64', dict(open_side, precision=['float64']), 'entry',
         'torchfdtd/open_mode_injection.py::_open_geometry', 'Open modal launch requires real FP32 uniform 3D staircase Yee sampling.',
         executions=DIFF)
    rule(f'{target}_open_launch_materials', dict(open_side, material=['dispersive_ade']), 'entry',
         'torchfdtd/open_mode_injection.py::_open_geometry', 'Open modal launch requires nondispersive isotropic materials.',
         executions=DIFF)
    if target == 'port':
        rule('port_open_launch_template', dict(open_side, source=['point']), 'entry',
             'torchfdtd/open_mode_injection.py::_open_geometry',
             'Use a soft Gaussian plane timing template with at least two carrier cycles.', executions=DIFF)
    if target == 'modal':
        rule('modal_open_detectors', dict(open_side, monitor=['radiation_box']), 'entry',
             'torchfdtd/open_mode_injection.py::OpenModalLaunch.detector_mode',
             'Validated modal detectors must lie on longitudinal E-node planes.', executions=DIFF,
             note='ModeInjectedPlaneSimulation validates every plane against the open launch; the box faces normal to the transverse axes are not d-node planes.')
rule('modal_planes_need_field_monitors', dict(source=['mode'], monitor=['point']), 'entry',
     'torchfdtd/adjoint_planes.py::DifferentiablePlaneSimulation.__init__',
     'DifferentiablePlaneSimulation requires enabled field monitors only.', executions=DIFF)



# Tensor ADE API (TensorDispersiveSimulation on a tensor project with explicit poles)
rule('tensor_ade_fused_backward', dict(material=['tensor_ade'], backend=['cuda_fused']), 'entry',
     'torchfdtd/anisotropy.py::TensorDielectricSimulation.__init__', 'Full-tensor fused kernels are not validated.',
     executions=RESIDENT, note='TensorProject (anisotropic_tensor) always uses the Torch transpose and takes no kernel choice.')

# Native forward dispatch: Simulation.run -> run_endpoint for PMC/symmetric faces
PMC = ('pmc', 'symmetric')
rule('endpoint_fp64', dict(boundary=PMC, precision=['float64']), 'entry', 'torchfdtd/endpoint_native.py::validate_endpoint_project',
     'PMC native dispatch requires fixed uniform/explicit real FP32 3D meshes.', executions=('forward',))
rule('endpoint_graded', dict(boundary=PMC, mesh=['graded']), 'entry', 'torchfdtd/endpoint_native.py::validate_endpoint_project',
     'PMC native dispatch requires fixed uniform/explicit real FP32 3D meshes.', executions=('forward',))
rule('endpoint_ade', dict(boundary=PMC, material=['dispersive_ade']), 'entry', 'torchfdtd/endpoint_native.py::validate_endpoint_project',
     'PMC native dispatch does not support ADE/dispersive materials.', executions=('forward',))
rule('endpoint_sources', dict(boundary=PMC, source=['sheet', 'plane_oneway', 'tfsf', 'tiled_sheet']), 'entry',
     'torchfdtd/endpoint_native.py::validate_endpoint_project',
     'PMC native dispatch accepts enabled point soft electric sources only.', executions=('forward',))
rule('endpoint_monitors', dict(boundary=PMC, monitor=['plane_dft', 'radiation_box']), 'entry',
     'torchfdtd/endpoint_native.py::validate_endpoint_project',
     'PMC native dispatch accepts enabled point E/H monitors at every timestep only.', executions=('forward',))

# Native forward on the Yee grid (fused kernel choice)
rule('forward_fused_complex', dict(backend=['cuda_fused'], boundary=['bloch'], material=['dielectric', 'dispersive_ade', 'pec']),
     'entry', 'torchfdtd/solver.py::Simulation._run',
     'The fused CUDA kernel currently supports real fields. Select cuda_kernel="torch" for Bloch fields.', executions=('forward',),
     note='Refused before the grid is allocated; FusedYeeCUDA repeats the refusal. Tensor materials are dispatched to '
          'run_tensor before this check and ignore the kernel choice.')

# Differentiable APIs: constructors
rule('differentiable_tfsf', dict(source=['tfsf']), 'entry', 'torchfdtd/differentiable.py::DifferentiableSimulation.__init__',
     'Live TFSF incident-state derivatives are not implemented yet.', executions=('checkpointed_adjoint',)+STREAMED)
rule('dispersive_pmc_fused', dict(material=['dispersive_ade'], boundary=PMC, backend=['cuda_fused'], monitor=['point']), 'entry',
     'torchfdtd/dispersive_adjoint.py::DispersiveSimulation.__init__',
     'The fused CUDA ADE kernels do not implement stored PMC/symmetric faces. Use cuda_kernel="torch" and backward_kernel="torch".',
     executions=('checkpointed_adjoint',))
rule('dispersive_oneway', dict(material=['dispersive_ade'], source=['plane_oneway'], monitor=['point']), 'entry',
     'torchfdtd/dispersive_adjoint.py::DispersiveSimulation.__init__',
     'Dispersive differentiation currently requires soft source injection.', executions=('checkpointed_adjoint',))
rule('streamed_dispersive_oneway', dict(material=['dispersive_ade'], source=['plane_oneway'], monitor=['point']), 'entry',
     'torchfdtd/streamed_dispersive.py::StreamedDispersiveSimulation.__init__',
     'Streamed ADE differentiation currently requires soft source injection.', executions=STREAMED)
rule('planes_pmc', dict(material=['dielectric'], monitor=['plane_dft', 'radiation_box'], boundary=PMC), 'entry',
     'torchfdtd/boundaries.py::reject_pmc_faces',
     'PMC/symmetric faces are not implemented by {path}. Use DifferentiableSimulation, StreamedSimulation, '
     'run_tensor_batch or the endpoint Simulation dispatch.', format=dict(path='DifferentiablePlaneSimulation'), executions=DIFF)
rule('dispersive_planes_pmc', dict(material=['dispersive_ade'], monitor=['plane_dft', 'radiation_box'], boundary=PMC), 'entry',
     'torchfdtd/boundaries.py::reject_pmc_faces',
     'PMC/symmetric faces are not implemented by {path}. Use DifferentiableSimulation, StreamedSimulation, '
     'run_tensor_batch or the endpoint Simulation dispatch.', format=dict(path='DispersivePlaneSimulation'), executions=DIFF)
rule('dispersive_planes_oneway', dict(material=['dispersive_ade'], source=['plane_oneway'], monitor=['plane_dft', 'radiation_box']), 'entry',
     'torchfdtd/dispersive_adjoint.py::DispersiveSimulation.__init__',
     'Dispersive differentiation currently requires soft source injection.', executions=('checkpointed_adjoint',))
rule('streamed_dispersive_planes_oneway', dict(material=['dispersive_ade'], source=['plane_oneway'], monitor=['plane_dft', 'radiation_box']),
     'entry', 'torchfdtd/streamed_dispersive.py::StreamedDispersiveSimulation.__init__',
     'Streamed ADE differentiation currently requires soft source injection.', executions=STREAMED)

# Differentiable APIs: run-time admission
rule('adjoint_pmc_fused_backward', dict(boundary=PMC, backend=['cuda_fused']), 'run',
     'torchfdtd/differentiable.py::DifferentiableSimulation._run',
     'The fused CUDA backward kernel does not implement PMC/symmetric faces. Use backward_kernel="auto" or "torch".',
     executions=('checkpointed_adjoint',))
rule('pmc_sheet_reaches_wall', dict(boundary=PMC, source=['tiled_sheet']), 'run', 'torchfdtd/differentiable.py::_System.__init__',
     '{source}: only point sources may address a stored upper PMC/symmetric face; plane sources must end below the wall.',
     executions=DIFF, note='Checked from the prepared source terms before the Yee system allocates its fields.')

# Reversible adjoints (ReversibleSimulation for closed periodic cells, ReversibleCPMLSimulation with CPML on d)
REV = ('reversible_adjoint',)
CLOSED = dict(source=['point'], monitor=['point'])
rule('reversible_pmc', dict(CLOSED, boundary=PMC), 'entry', 'torchfdtd/boundaries.py::reject_pmc_faces',
     'PMC/symmetric faces are not implemented by {path}. Use DifferentiableSimulation, StreamedSimulation, '
     'run_tensor_batch or the endpoint Simulation dispatch.', format=dict(path='ReversibleSimulation'), executions=REV)
rule('reversible_region', dict(CLOSED, boundary=['periodic', 'bloch', 'pec', 'antisymmetric'], dimension=['2d']), 'entry',
     'torchfdtd/reversible.py::_validate_project',
     'ReversibleSimulation requires uniform 3D FP32 Yee staircase sampling and fixed resident steps.', executions=REV)
rule('reversible_fp64', dict(CLOSED, boundary=['periodic', 'bloch', 'pec', 'antisymmetric'], precision=['float64']), 'entry',
     'torchfdtd/reversible.py::_validate_project',
     'ReversibleSimulation requires uniform 3D FP32 Yee staircase sampling and fixed resident steps.', executions=REV)
rule('reversible_mesh', dict(CLOSED, boundary=['periodic', 'bloch', 'pec', 'antisymmetric'], mesh=['graded', 'explicit']), 'entry',
     'torchfdtd/reversible.py::_validate_project',
     'ReversibleSimulation requires uniform 3D FP32 Yee staircase sampling and fixed resident steps.', executions=REV)
rule('reversible_faces', dict(CLOSED, boundary=['bloch', 'pec', 'antisymmetric']), 'entry', 'torchfdtd/reversible.py::_validate_project',
     'ReversibleSimulation requires all six boundaries to be periodic, without Bloch phase.', executions=REV)
rule('reversible_materials', dict(CLOSED, boundary=['periodic'], material=['anisotropic_tensor']), 'entry',
     'torchfdtd/reversible.py::_validate_project',
     'ReversibleSimulation supports only nondispersive dielectric material declarations.', executions=REV)
OPEN = dict(boundary=['cpml', 'periodic', 'bloch', 'pec', 'antisymmetric'])
rule('reversible_cpml_pmc_point', dict(boundary=PMC, monitor=['point']), 'entry', 'torchfdtd/boundaries.py::reject_pmc_faces',
     'PMC/symmetric faces are not implemented by {path}. Use DifferentiableSimulation, StreamedSimulation, '
     'run_tensor_batch or the endpoint Simulation dispatch.', format=dict(path='ReversibleCPMLSimulation'), executions=REV)
rule('reversible_cpml_pmc_planes', dict(boundary=PMC, monitor=['plane_dft', 'radiation_box']), 'entry',
     'torchfdtd/boundaries.py::reject_pmc_faces',
     'PMC/symmetric faces are not implemented by {path}. Use DifferentiableSimulation, StreamedSimulation, '
     'run_tensor_batch or the endpoint Simulation dispatch.', format=dict(path='ReversibleCPMLPlaneSimulation'), executions=REV)
rule('reversible_cpml_region', dict(OPEN, dimension=['2d']), 'entry', 'torchfdtd/reversible_cpml.py::_validate_project',
     'ReversibleCPMLSimulation requires uniform 3D FP32 Yee staircase sampling and fixed resident steps.', executions=REV)
rule('reversible_cpml_fp64', dict(OPEN, precision=['float64']), 'entry', 'torchfdtd/reversible_cpml.py::_validate_project',
     'ReversibleCPMLSimulation requires uniform 3D FP32 Yee staircase sampling and fixed resident steps.', executions=REV)
rule('reversible_cpml_mesh', dict(OPEN, mesh=['graded', 'explicit']), 'entry', 'torchfdtd/reversible_cpml.py::_validate_project',
     'ReversibleCPMLSimulation requires uniform 3D FP32 Yee staircase sampling and fixed resident steps.', executions=REV)
rule('reversible_cpml_faces', dict(boundary=['cpml', 'pec', 'antisymmetric']), 'entry', 'torchfdtd/reversible_cpml.py::_validate_project',
     'ReversibleCPMLSimulation requires periodic or Bloch x/y and CPML on both z faces.', executions=REV)
rule('reversible_cpml_sources', dict(boundary=['periodic', 'bloch'], source=['plane_oneway', 'tfsf']), 'entry',
     'torchfdtd/reversible_cpml.py::_validate_project',
     'ReversibleCPMLSimulation supports fixed soft electric point or z-normal plane sources only.', executions=REV)
rule('reversible_lorentz_bloch', dict(boundary=['bloch'], material=['dispersive_ade']), 'entry',
     'torchfdtd/reversible_lorentz.py::resolve_material',
     'Reversible Lorentz density mixtures currently require real fields;', executions=REV)

# Tensor batch (run_tensor_batch)
TB = ('tensor_batch',)
rule('tensor_batch_cpu', dict(backend=['cpu']), 'entry', 'torchfdtd/tensor_batch.py::run_tensor_batch',
     'Tensor batch cannot execute a CPU project. Set backend="cuda" or "auto" explicitly.', executions=TB)
rule('tensor_batch_tensor_material', dict(material=['anisotropic_tensor']), 'entry', 'torchfdtd/tensor_batch.py::run_tensor_batch',
     'Tensor batch does not implement tensor materials; run the native tensor solver per project.', executions=TB)
rule('tensor_batch_complex', dict(boundary=['bloch']), 'entry', 'torchfdtd/tensor_batch.py::run_tensor_batch',
     'Tensor batch currently requires real fields. Use BatchRunner for complex Bloch fields.', executions=TB)
rule('tensor_batch_pmc_planes', dict(boundary=PMC, monitor=['plane_dft', 'radiation_box']), 'entry',
     'torchfdtd/tensor_batch.py::_validate_pmc_case',
     'Tensor batch does not implement field monitors with PMC/symmetric faces; use point monitors.', executions=TB)
rule('tensor_batch_pmc_sheet_reaches_wall', dict(boundary=PMC, source=['tiled_sheet']), 'entry', 'torchfdtd/tensor_batch.py::_validate_pmc_case',
     '{source}: only point sources may address a stored upper PMC/symmetric face; plane sources must end below the wall.', executions=TB,
     note='Checked from the prepared source terms before the cohort grids are allocated; FusedBatchIO repeats the refusal.')

# Tiled execution (plan_tiles)
TI = ('tiled',)
rule('tiled_mesh', dict(mesh=['graded', 'explicit']), 'entry', 'torchfdtd/tiled.py::plan_tiles',
     'Tiling requires a uniform mesh with one spacing on every axis.', executions=TI)
rule('tiled_complex', dict(boundary=['bloch']), 'entry', 'torchfdtd/tiled.py::plan_tiles',
     'Tiling requires real fields; a Bloch phase describes a periodic cell, not a finite device.', executions=TI)
rule('tiled_faces', dict(boundary=['periodic', 'pec', 'antisymmetric', 'pmc', 'symmetric']), 'entry', 'torchfdtd/tiled.py::plan_tiles',
     'Tiling requires CPML on the lateral and normal faces of the global project; each tile adds its own CPML at the cuts.',
     executions=TI)
rule('tiled_sources', dict(source=['point', 'plane_oneway', 'tfsf']), 'entry', 'torchfdtd/tiled.py::plan_tiles',
     '{source}: tiling requires soft plane sources normal to {normal}; point, one-way and TFSF sources are not partitioned.',
     executions=TI)
rule('tiled_sheet_span', dict(source=['sheet']), 'entry', 'torchfdtd/tiled.py::plan_tiles',
     '{source}: the sheet must span the whole non-PML lateral extent so that every tile sees the same incidence.', executions=TI)
rule('tiled_monitor', dict(monitor=['point', 'radiation_box']), 'entry', 'torchfdtd/tiled.py::plan_tiles',
     'Tiling requires exactly one enabled field monitor normal to {normal}: the output plane.', executions=TI)

# ---- Stage 4: post-processing of stored results ----
NATIVE_SOURCES = ('point', 'sheet', 'plane_oneway', 'tfsf', 'tiled_sheet')
rule('radiation_box_mesh', dict(monitor=['radiation_box'], source=NATIVE_SOURCES, mesh=['graded', 'explicit']), 'post',
     'torchfdtd/radiation_box.py::_axis_workspace_bytes',
     'Stored far fields require isolated uniform 3D geometry.', executions=('forward', 'tensor_batch'),
     note='The stored-result adapter checks the native mesh after the run; the differentiable project_farfield has no such check.')
rule('radiation_box_faces', dict(monitor=['radiation_box'], source=NATIVE_SOURCES, boundary=NOT_CPML), 'post',
     'torchfdtd/radiation_box.py::_geometry',
     'Stored far fields require isolated uniform 3D geometry with PML on all six outer faces.', executions=('forward', 'tensor_batch'),
     note='Checked after the run on the stored planes; the differentiable project_farfield accepts any six planes.')
rule('radiation_box_sheet_support', dict(monitor=['radiation_box'], source=['sheet', 'tiled_sheet']), 'post',
     'torchfdtd/radiation_box.py::_sources',
     'Total-field source support must be enclosed and clear of the measurement faces.', executions=('forward', 'tensor_batch'),
     note='A soft sheet outside the box is a total-field source; the closed-box transform needs it enclosed.')
rule('radiation_box_tfsf_support', dict(monitor=['radiation_box'], source=['tfsf']), 'post', 'torchfdtd/radiation_box.py::_sources',
     'TFSF box faces cross or approach the measurement faces; enclose the TFSF box with one cell', executions=('forward', 'tensor_batch'))

# ---- Lanes: the admitted entry points, first match wins ----
lane('forward.mode_network', dict(execution=['forward'], monitor=['mode_port']), 'torchfdtd/mode_network.py::ModeNetwork.__call__',
     'tests/test_capability_pairs.py::test_pairwise', 'Two-port complex S matrix of the fixed mode ports without gradients '
     '(prepare_modal_launch for periodic/Bloch transverse faces, prepare_open_modal_launch for CPML).')
lane('forward.modal_planes', dict(execution=['forward'], source=['mode']), 'torchfdtd/mode_injection.py::ModeInjectedPlaneSimulation.forward',
     'tests/test_capability_pairs.py::test_pairwise', 'Fixed-eigenmode sheet launch with spectral plane observers, no gradient.')
lane('forward.tensor_ade', dict(execution=['forward'], material=['tensor_ade']),
     'torchfdtd/anisotropy_dispersive.py::TensorDispersiveSimulation.forward',
     'tests/test_capability_pairs.py::test_pairwise', 'Full-tensor trapezoidal ADE on node tensors, no gradient.')
lane('forward.tensor', dict(execution=['forward'], material=['anisotropic_tensor']), 'torchfdtd/tensor_native.py::run_tensor',
     'tests/test_capability_pairs.py::test_pairwise', 'Native tensor solver dispatched by Simulation.run (Torch operations).',
     note='cuda_kernel="fused" is not honoured: the summary warns that tensor execution uses Torch operations.')
lane('forward.endpoint', dict(execution=['forward'], boundary=['pmc', 'symmetric']), 'torchfdtd/endpoint_native.py::run_endpoint',
     'tests/test_capability_pairs.py::test_pairwise', 'Exact-endpoint PEC/PMC/CPML solver dispatched by Simulation.run.')
lane('forward.grid', dict(execution=['forward']), 'torchfdtd/solver.py::Simulation.run',
     'tests/test_capability_pairs.py::test_pairwise', 'Native Yee/CPML forward on the fdtd grid; radiation boxes project the stored planes '
     'through native_radiation_box.')
lane('checkpointed.mode_network', dict(execution=['checkpointed_adjoint'], monitor=['mode_port']),
     'torchfdtd/mode_network.py::ModeNetwork.__call__', 'tests/test_capability_pairs.py::test_pairwise',
     'Interior material VJP of the two-port S matrix.')
lane('checkpointed.modal_planes', dict(execution=['checkpointed_adjoint'], source=['mode']),
     'torchfdtd/mode_injection.py::ModeInjectedPlaneSimulation.forward', 'tests/test_capability_pairs.py::test_pairwise',
     'Checkpointed adjoint through the fixed modal sheets.')
lane('checkpointed.tensor_ade', dict(execution=['checkpointed_adjoint'], material=['tensor_ade']),
     'torchfdtd/anisotropy_dispersive.py::TensorDispersiveSimulation.forward', 'tests/test_capability_pairs.py::test_pairwise',
     'Checkpointed derivatives of node epsilon_inf and tensor poles.')
lane('checkpointed.tensor', dict(execution=['checkpointed_adjoint'], material=['anisotropic_tensor']),
     'torchfdtd/tensor_project.py::TensorProject.__call__', 'tests/test_capability_pairs.py::test_pairwise',
     'Project-rasterized node tensors through TensorDielectricSimulation with the fixed isotropic CPML collar.')
lane('checkpointed.dispersive_planes', dict(execution=['checkpointed_adjoint'], material=['dispersive_ade'], monitor=['plane_dft', 'radiation_box']),
     'torchfdtd/dispersive_adjoint.py::DispersivePlaneSimulation.forward', 'tests/test_capability_pairs.py::test_pairwise',
     'ADE material VJPs of spectral planes; radiation boxes project through project_farfield.')
lane('checkpointed.dispersive', dict(execution=['checkpointed_adjoint'], material=['dispersive_ade']),
     'torchfdtd/dispersive_adjoint.py::DispersiveSimulation.forward', 'tests/test_capability_pairs.py::test_pairwise',
     'Checkpointed derivatives of epsilon_inf and the passive poles.')
lane('checkpointed.planes', dict(execution=['checkpointed_adjoint'], monitor=['plane_dft', 'radiation_box']),
     'torchfdtd/adjoint_planes.py::DifferentiablePlaneSimulation.forward', 'tests/test_capability_pairs.py::test_pairwise',
     'Checkpointed adjoint of fixed spectral planes; radiation boxes project through project_farfield.')
lane('checkpointed.point', dict(execution=['checkpointed_adjoint']), 'torchfdtd/differentiable.py::DifferentiableSimulation.forward',
     'tests/test_capability_pairs.py::test_pairwise', 'Discrete Yee/CPML adjoint with bounded checkpoints and point observations.')
lane('streamed.modal_planes', dict(execution=STREAMED, source=['mode']),
     'torchfdtd/mode_injection.py::ModeInjectedPlaneSimulation.forward', 'tests/test_capability_pairs.py::test_pairwise',
     'Streamed X-slab execution of the fixed modal sheets and plane observers.')
lane('streamed.tensor', dict(execution=STREAMED, material=['anisotropic_tensor']),
     'torchfdtd/streamed_tensor.py::StreamedTensorSimulation.forward', 'tests/test_capability_pairs.py::test_pairwise',
     'Streamed node-tensor slabs with the geometric CPML criterion.')
lane('streamed.dispersive_planes', dict(execution=STREAMED, material=['dispersive_ade'], monitor=['plane_dft', 'radiation_box']),
     'torchfdtd/dispersive_adjoint.py::DispersivePlaneSimulation.forward', 'tests/test_capability_pairs.py::test_pairwise',
     'Streamed ADE slabs with spectral plane observers.')
lane('streamed.dispersive', dict(execution=STREAMED, material=['dispersive_ade']),
     'torchfdtd/streamed_dispersive.py::StreamedDispersiveSimulation.forward', 'tests/test_capability_pairs.py::test_pairwise',
     'Streamed ADE slabs with point observations.')
lane('streamed.planes', dict(execution=STREAMED, monitor=['plane_dft', 'radiation_box']),
     'torchfdtd/adjoint_planes.py::DifferentiablePlaneSimulation.forward', 'tests/test_capability_pairs.py::test_pairwise',
     'Streamed slabs with spectral plane observers.')
lane('streamed.point', dict(execution=STREAMED), 'torchfdtd/streamed.py::StreamedSimulation.forward',
     'tests/test_capability_pairs.py::test_pairwise', 'Host or file-backed streamed slabs with point observations; '
     'the streamed execution mode runs it without gradient, the streamed adjoint with backward.')
lane('reversible.periodic', dict(execution=REV, source=['point'], monitor=['point'], boundary=['periodic']),
     'torchfdtd/reversible.py::ReversibleSimulation.forward', 'tests/test_capability_pairs.py::test_pairwise',
     'Lossless periodic reconstruction adjoint on a closed periodic cell.')
lane('reversible.cpml_planes', dict(execution=REV, monitor=['plane_dft', 'radiation_box']),
     'torchfdtd/reversible_cpml_planes.py::ReversibleCPMLPlaneSimulation.forward', 'tests/test_capability_pairs.py::test_pairwise',
     'Recorded-interface CPML adjoint with spectral planes.')
lane('reversible.cpml', dict(execution=REV), 'torchfdtd/reversible_cpml.py::ReversibleCPMLSimulation.forward',
     'tests/test_capability_pairs.py::test_pairwise', 'Recorded-interface CPML adjoint with point observations.')
lane('tensor_batch', dict(execution=TB), 'torchfdtd/tensor_batch.py::run_tensor_batch', 'tests/test_capability_pairs.py::test_pairwise',
     'One-case fused CUDA cohort; radiation boxes project the stored planes through native_radiation_box.',
     note='The cohort always runs the fused batch kernel; cuda_kernel="torch" is reported as fused_batch in the summary.')
lane('tiled', dict(execution=TI), 'torchfdtd/tiled.py::run_tiled', 'tests/test_capability_pairs.py::test_pairwise',
     'plan_tiles partitions the lateral extent; each tile runs Simulation.run and the output planes are stitched.')


# ----------------------------------------------------------------------------
# Scene recipe
# ----------------------------------------------------------------------------

def _source_name(config):
    return {'point': 'point', 'sheet': 'sheet', 'plane_oneway': 'oneway', 'tfsf': 'tfsf', 'mode': 'mode',
            'tiled_sheet': 'tiled sheet'}[config['source']]


def _materials(config):
    from .models import Material, LorentzPole
    base = [Material(name='Air', index=1, color='#b7c5d7')]
    kind = config['material']
    if kind == 'dielectric':
        base.append(Material(name='block', index=2.))
    elif kind == 'dispersive_ade':
        base.append(Material(name='block', model='multipole', epsilon_inf=2.,
                             poles=[LorentzPole(**DRUDE), LorentzPole(**POLE)]))
    elif kind in TENSOR_MATERIALS:
        base.append(Material(name='block', model='tensor', epsilon_tensor=TENSOR))
    elif kind == 'pec':
        base.append(Material(name='block', model='pec'))
    return base


def _region(config, graded_nodes=None):
    from .models import Region, BoundaryFace, Boundaries
    d = propagation_axis(config)
    size = [TRANSVERSE_UM]*3
    size[d] = LONGITUDINAL_UM
    faces = {}
    for name, kind in face_kinds(config).items():
        faces[name] = BoundaryFace(kind=kind, alpha=0.) if kind == 'pml' else BoundaryFace(kind=kind)
    phase = tuple(BLOCH_PHASE if (kind == 'bloch' and name.endswith('_min')) else 0.
                  for name, kind in face_kinds(config).items() if name.endswith('_min'))
    settings = dict(dimension=config['dimension'], size=tuple(size), mesh=MESH_UM, pml_cells=PML_CELLS, steps=STEPS,
                    material_sampling='yee', precision=config['precision'], snapshot_interval=STEPS,
                    backend='cpu' if config['backend'] == 'cpu' else 'cuda',
                    cuda_kernel='fused' if config['backend'] == 'cuda_fused' else 'torch',
                    boundaries=Boundaries(**faces), bloch_phase=phase)
    if config['mesh'] == 'graded':
        settings.update(mesh_type='graded', mesh_max=.2, mesh_ppw=6)
    elif config['mesh'] == 'explicit':
        settings.update(mesh_type='explicit', mesh_coordinates=graded_nodes)
    return Region(**settings)


def _sources(config, region):
    from .models import Source
    d = propagation_axis(config)
    kind = config['source']
    common = dict(name=_source_name(config), wavelength=WAVELENGTH_UM, pulse_cycles=2)
    if kind == 'point':
        center = [0.05, -0.05, 0.]
        if d == 2:
            center[2] = -0.15
        else:
            center[1] = -0.15
        return [Source(id='source', kind='point', component='Ez' if d == 2 else 'Ez', center=tuple(center), **common)]
    center = [0., 0., 0.]
    center[d] = -PORT_SOURCE_UM
    normal = 'xyz'[d]
    polarization = 'Ex'
    if kind == 'tfsf':
        size = [BOX_UM]*3
        size[d] = BOX_UM
        return [Source(id='source', kind='tfsf', normal=normal, component=polarization, center=(0., 0., 0.),
                       size=tuple(size), **common)]
    size = [0., 0., 0.]
    for a in transverse_axes(config):
        lo, hi = region.interior_bounds(a)
        size[a] = hi-lo
        if kind == 'sheet':
            size[a] = round(hi-lo-2*MESH_UM, 9)
        elif kind in ('plane_oneway', 'mode'):
            size[a] = region.actual_size[a] if transverse_kind(config) != 'pml' else round(hi-lo, 9)
        elif kind == 'tiled_sheet':
            size[a] = region.actual_size[a]
    if config['dimension'] == '2d':
        size[2] = 0.
    extra = {}
    if kind == 'plane_oneway':
        extra = dict(injection='oneway')
    elif kind == 'tiled_sheet':
        extra = dict(extend_through_pml=True)
    return [Source(id='source', kind='plane', normal=normal, component=polarization, center=tuple(center),
                   size=tuple(size), **common, **extra)]


def _spectrum():
    from .models import SpectrumSettings
    frequency = 299792458.0/(WAVELENGTH_UM*1e-6)
    return SpectrumSettings(sampling='custom', custom_frequencies_hz=[frequency], apodization='none')


def _monitors(config, region):
    from .models import Monitor, FieldMonitor
    d = propagation_axis(config)
    kind = config['monitor']
    if kind == 'point':
        center = [0.15, 0.05, 0.]
        if d == 2:
            center[2] = 0.25
        else:
            center[1] = 0.25
        return [Monitor(id='probe', name='probe', component='Ez', center=tuple(center))]
    if kind == 'plane_dft':
        center = [0., 0., 0.]
        center[d] = PORT_UM
        size = [0., 0., 0.]
        for a in transverse_axes(config):
            lo, hi = region.interior_bounds(a)
            size[a] = round(hi-lo-(2*MESH_UM if transverse_kind(config) in WALLS else 0.), 9)
        if config['dimension'] == '2d':
            size[2] = 1.
        return [FieldMonitor(id='plane', name='plane', normal='xyz'[d], center=tuple(center), size=tuple(size),
                             spectrum=_spectrum())]
    if kind == 'radiation_box':
        monitors = []
        for axis, name in enumerate('xyz'):
            for side, sign in (('min', -1), ('max', 1)):
                center = [0., 0., 0.]
                center[axis] = sign*BOX_UM/2
                size = [BOX_UM]*3
                size[axis] = 0.
                monitors.append(FieldMonitor(id=f'{name}_{side}', name=f'{name}_{side}', normal=name, center=tuple(center),
                                             size=tuple(size), spectrum=_spectrum()))
        return monitors
    return []   # mode_port: the network owns its detector planes


def uses_guide(config):
    """Mode scenes replace the block by a straight guide along the propagation axis."""
    return config['source'] == 'mode' or config['monitor'] == 'mode_port'


def _structures(config):
    from .models import Structure
    size = [BLOCK_UM]*3
    if uses_guide(config):
        size[propagation_axis(config)] = LONGITUDINAL_UM
    return [Structure(id='block', name='block', kind='rectangle', size=tuple(size), material='block')]


def example_project(config):
    """The Project that realizes one combination; raises the schema rejection of an inexpressible one."""
    validate_config(config)
    from .models import Project
    materials = _materials(config)
    graded_nodes = None
    if config['mesh'] == 'explicit':
        graded = example_project({**config, 'mesh': 'graded'})
        graded_nodes = tuple(tuple(float(v) for v in nodes) for nodes in graded.region.mesh_nodes)
    region = _region(config, graded_nodes)
    project = Project(name=config_id(config), region=region, materials=materials, structures=_structures(config),
                      sources=_sources(config, region), monitors=_monitors(config, region))
    return Project.model_validate(project.model_dump())


def mode_ports(config):
    """The two opposing fixed mode ports of the mode_port monitor along the propagation axis."""
    from .mode_network import FixedModePort
    return (FixedModePort('left', -PORT_UM, -PORT_SOURCE_UM, 1, (0,)),
            FixedModePort('right', PORT_UM, PORT_SOURCE_UM, -1, (0,)))


def radiation_box_ids():
    return {f'{a}_{s}': f'{a}_{s}' for a in 'xyz' for s in ('min', 'max')}


def radiation_box_bounds():
    return tuple((-BOX_UM/2, BOX_UM/2) for _ in range(3))
