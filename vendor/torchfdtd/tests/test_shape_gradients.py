"""Polygon and spline shape gradients: voxel VJP, adjoint chain, regressions and the convergence record."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from torchfdtd import (AdjointOptions, DensityParameterization, DifferentiablePlaneSimulation,
    DifferentiableSimulation, DifferentiableSolid, FieldMonitor, Monitor, Project, Region, Source,
    Structure, smooth_geometry_epsilon, spline_outline)
from torchfdtd.solver import C0, field_axes, voxelize

VERTICES = ((-.31, -.22), (.27, -.29), (.33, .08), (.04, .3), (-.28, .17))
# Off-grid controls: a vertex exactly on a Yee sample is a cone kink with a subgradient only.
CONTROLS = ((-.3137, -.2071), (.2913, -.2567), (.3481, .1123), (.0137, .3072), (-.2891, .1613))
RECORD = Path(__file__).resolve().parents[1]/'docs'/'validation'/'shape_gradient_polygon_3060.json'


def region(precision='float32', sampling='yee'):
    return Region(dimension='3d', size=(1.4, 1.3, 1.2), mesh=.1, pml_cells=3, steps=18,
                  precision=precision, material_sampling=sampling)


def skip_without(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')


def polygon_scene(p):
    """Vertices, z limits, rotation, center and epsilon all trainable."""
    return [DifferentiableSolid.polygon(p[:10].reshape(5, 2), p[10:12], epsilon=p[18],
                                        center=p[15:18], rotation=p[12:15])]


# SHA-256 of voxelize() at commit c297737, before differentiable polygons existed.
VOXELIZE_SHA = {
    ('yee', 'float32'): '70342051d047016453dc4f8e481ebe2f96757cc3a05eb34e95ddd8fa9fb38aa7',
    ('yee', 'float64'): 'e2925c5b298606893395519ccb1afff5385e290f87d451917a32a3da1304be26',
    ('cell', 'float32'): '31a8b614693799dcf25a22c81650c519dc35a269d29af0b0f0fd1d1e9b5c2017',
    ('cell', 'float64'): 'c1c803c3c9b7c375e4898b37f237be292cfbc45626fb01f80bfe5509f9409644',
}


@pytest.mark.parametrize('sampling', ['yee', 'cell'])
def test_polygon_fill_reduces_to_existing_sampling_and_voxelizer_is_bitwise_unchanged(sampling):
    structure = Structure(kind='polygon', material='SiN (constant n)', vertices=VERTICES,
                          size=(1., 1., .46), center=(.05, -.04, .03), rotation=23.)
    for precision in ('float32', 'float64'):
        r = region(precision, sampling)
        eps, _ = voxelize(Project(region=r, structures=[structure]))
        assert hashlib.sha256(np.ascontiguousarray(eps).tobytes()).hexdigest() == VOXELIZE_SHA[sampling, precision]
    solid = DifferentiableSolid.polygon(VERTICES, (-.23, .23), epsilon=4., center=(.05, -.04, .03),
                                        rotation=(0., 0., 23.))
    smooth = smooth_geometry_epsilon(r, [solid], width=1e-9, chunk_cells=101).numpy()
    if sampling == 'yee':
        axes = [field_axes(r, c) for c in ('Ex', 'Ey', 'Ez')]
        views = [(smooth[..., c], eps[..., c]) for c in range(3)]
    else:
        axes = [[(n[:-1]+n[1:])/2 for n in r.mesh_nodes]]
        views = [(smooth, eps)]
    faces = 0
    for axis, (fill, expected) in zip(axes, views):
        x, y, z = np.meshgrid(*axis, indexing='ij')
        # Samples exactly on an extrusion face are ambiguous in both models: the sampler's
        # inclusive test depends on rounding and the fill gives one half of the contrast.
        on_face = np.isclose(np.abs(z-.03), .23)
        faces += int(on_face.sum())
        assert np.array_equal(fill[~on_face], expected[~on_face])
        assert np.all((fill[on_face] == 1.) | np.isclose(fill[on_face], 2.5))
    assert (faces > 0) == (sampling == 'yee')


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def test_polygon_vjp_taylor_remainder_and_central_difference(device):
    skip_without(device)
    r = region('float64')
    p = torch.tensor([*sum(VERTICES, ()), -.21, .25, 7., -11., 23., .05, -.04, .03, 3.3],
                     dtype=torch.float64, device=device, requires_grad=True)
    seed = torch.randn((3, *r.shape), dtype=torch.float64, generator=torch.Generator().manual_seed(3)).movedim(0, -1).to(device)
    assert not seed.is_contiguous()

    def scalar(q):
        return (smooth_geometry_epsilon(r, polygon_scene(q), width=.08, chunk_cells=97)*seed).sum()
    value = scalar(p)
    gradient, = torch.autograd.grad(value, p)
    assert bool((gradient.abs() > 1e-3).all())
    direction = torch.randn(19, dtype=torch.float64, generator=torch.Generator().manual_seed(5)).to(device)
    direction[12:15] *= 10
    slope = float(gradient@direction)
    with torch.no_grad():
        h = 1e-5
        central = float((scalar(p+h*direction)-scalar(p-h*direction))/(2*h))
        remainders = [abs(float(scalar(p+h*direction)-value-h*slope)) for h in (1e-3, 2.5e-4)]
    assert math.isclose(central, slope, rel_tol=2e-4)
    # Superlinear remainder: the fill is piecewise C1, so a quarter step gains more than a factor of four.
    assert remainders[1] < remainders[0]/8


def test_spline_outline_geometry_periodicity_and_control_point_gradient():
    controls = torch.tensor(CONTROLS, dtype=torch.float64, requires_grad=True)
    catmull = spline_outline(controls, kind='catmull_rom', samples_per_segment=6)
    bspline = spline_outline(controls, kind='bspline', samples_per_segment=6)
    assert catmull.shape == bspline.shape == (30, 2)
    torch.testing.assert_close(catmull[::6], controls)
    torch.testing.assert_close(bspline[::6], (controls.roll(1, 0)+4*controls+controls.roll(-1, 0))/6)
    for outline, kind in ((catmull, 'catmull_rom'), (bspline, 'bspline')):
        torch.testing.assert_close(spline_outline(controls.roll(1, 0), kind=kind, samples_per_segment=6), outline.roll(6, 0))
    assert torch.autograd.gradcheck(lambda c: spline_outline(c, kind='bspline', samples_per_segment=4), (controls,))
    r = region('float64')
    seed = torch.randn((3, *r.shape), dtype=torch.float64, generator=torch.Generator().manual_seed(3)).movedim(0, -1)
    direction = torch.randn(5, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(9))
    for kind in ('catmull_rom', 'bspline'):
        def scalar(c):
            solid = DifferentiableSolid.spline(c, (-.2, .2), epsilon=4., kind=kind, samples_per_segment=6)
            return (smooth_geometry_epsilon(r, [solid], width=.08, chunk_cells=97)*seed).sum()
        gradient, = torch.autograd.grad(scalar(controls), controls)
        with torch.no_grad():
            h = 1e-5
            central = float((scalar(controls+h*direction)-scalar(controls-h*direction))/(2*h))
        assert math.isclose(central, float((gradient*direction).sum()), rel_tol=1e-4)


def test_polygon_and_spline_input_contracts():
    r = region()
    with pytest.raises(ValueError, match='upper z limit'):
        smooth_geometry_epsilon(r, [DifferentiableSolid.polygon(VERTICES, (.2, -.2), epsilon=4.)])
    with pytest.raises(ValueError, match='holes'):
        smooth_geometry_epsilon(r, [DifferentiableSolid.polygon(VERTICES, (-.2, .2), epsilon=4., holes=[((-.1, -.1), (.1, -.1), (0., .1))])])
    with pytest.raises(ValueError, match='cross'):
        smooth_geometry_epsilon(r, [DifferentiableSolid.polygon(((-.3, -.3), (.3, .3), (.3, -.3), (-.3, .25)), (-.2, .2), epsilon=4.)])
    with pytest.raises(ValueError, match='at least one'):
        smooth_geometry_epsilon(r, [DifferentiableSolid.polygon(VERTICES, (-.2, .2), epsilon=.5)])
    with pytest.raises(ValueError, match='pairs'):
        smooth_geometry_epsilon(r, [DifferentiableSolid.polygon((1., 2., 3.), (-.2, .2), epsilon=4.)])
    with pytest.raises(ValueError, match='kind'):
        spline_outline(torch.tensor(CONTROLS), kind='bezier')
    with pytest.raises(ValueError, match='samples_per_segment'):
        spline_outline(torch.tensor(CONTROLS), samples_per_segment=0)
    with pytest.raises(ValueError, match='three rows'):
        spline_outline(torch.tensor(CONTROLS[:2]))
    with pytest.raises(ValueError, match='precision'):
        smooth_geometry_epsilon(r, [DifferentiableSolid.spline(torch.tensor(CONTROLS, dtype=torch.float64), (-.2, .2), epsilon=4.)])


def adjoint_scene(vertices, controls):
    return [DifferentiableSolid.polygon(vertices, (-.2, .2), epsilon=3.4, center=(.12, -.06, 0.), rotation=(0., 0., 15.)),
            DifferentiableSolid.spline(controls, (-.5, .1), epsilon=2.1, center=(-.25, .15, .1),
                                       kind='bspline', samples_per_segment=4)]


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def test_vertex_and_control_gradients_through_checkpointed_adjoint(device):
    skip_without(device)
    precision = 'float64' if device == 'cpu' else 'float32'
    dtype = getattr(torch, precision)
    r = region(precision)
    project = Project(region=r, sources=[Source(center=(-.2, 0, 0), pulse='continuous')],
                      monitors=[Monitor(center=(.1, 0, 0)), Monitor(center=(0, .1, 0), component='Hy')])
    vertices = torch.tensor(np.array(VERTICES)*.6, dtype=dtype, device=device, requires_grad=True)
    controls = torch.tensor(np.array(CONTROLS)*.5, dtype=dtype, device=device, requires_grad=True)
    leaves = (vertices, controls)
    model = DifferentiableSimulation(project, AdjointOptions(checkpoints=2))
    frequency = torch.tensor([C0/1.55e-6], dtype=dtype, device=device)

    def epsilon(v, c):
        return smooth_geometry_epsilon(r, adjoint_scene(v, c), width=.1, chunk_cells=257)

    def signals(v, c):
        return model(epsilon(v, c)).signals.square().sum()

    def point_spectrum(v, c):
        return model.spectrum(epsilon(v, c), frequency).fields.abs().square().sum()
    if device == 'cuda':
        pytest.importorskip('cupy')
        expected = torch.autograd.grad(model.reference(epsilon(vertices, controls)).square().sum(), leaves)
        actual = torch.autograd.grad(signals(vertices, controls), leaves)
        for a, e in zip(actual, expected):
            assert bool(torch.isfinite(a).all()) and torch.linalg.vector_norm(a) > 0
            torch.testing.assert_close(a, e, rtol=3e-4, atol=1e-8)
        return
    plane = Project(region=r, sources=project.sources, monitors=[FieldMonitor(id='P', center=(.25, 0, 0), size=(0, .6, .5))])
    plane_model = DifferentiablePlaneSimulation(plane, AdjointOptions(checkpoints=2))

    def plane_spectrum(v, c):
        return plane_model(epsilon(v, c), frequency)['P'].flux().sum()
    generator = torch.Generator().manual_seed(11)
    directions = tuple(torch.randn(5, 2, dtype=dtype, generator=generator) for _ in leaves)
    for objective in (signals, point_spectrum, plane_spectrum):
        gradients = torch.autograd.grad(objective(vertices, controls), leaves)
        slope = sum(float((g*d).sum()) for g, d in zip(gradients, directions))
        with torch.no_grad():
            h = 1e-5
            plus = objective(*(x+h*d for x, d in zip(leaves, directions)))
            minus = objective(*(x-h*d for x, d in zip(leaves, directions)))
        assert all(bool(torch.isfinite(g).all()) and torch.linalg.vector_norm(g) > 0 for g in gradients)
        assert math.isclose(float((plus-minus)/(2*h)), slope, rel_tol=2e-4), objective.__name__


# Values at commit c297737 for the pre-existing solid VJP and DensityParameterization,
# computed in FP64 on the development machine. Other BLAS builds differ at round-off,
# so the comparison is at 1e-11 relative rather than bitwise; a changed formula
# moves these numbers by far more than that.
SOLID_FORWARD_NUMEL = 6552
SOLID_FORWARD_SUM = 8260.061907096566
SOLID_FORWARD_SUMSQ = 11011.767250520306
SOLID_FORWARD_SAMPLES = [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 3.3621829041649596, 1.2, 3.2505079582759167, 1.2, 1.2, 1.2, 1.2, 2.908838076293195, 1.2, 1.2, 1.2, 1.2, 1.2]
SOLID_GRADIENT = [8.872056388275427, 66.33811558833472, -78.66300315125868, -0.08282598286201565, -0.37789172133901927, -0.8303187137259035, -2.8946998305268195, 169.00180759381277, 99.30053372511483]
SOLID_BACKGROUND_GRADIENT = -14.47232522153175
DENSITY_FORWARD = [0.5501179440118242, 0.5453327144203235, 0.556384408776538, 0.5673815140890016, 0.5783126210580789, 0.5891665988314675, 0.5844829946857041, 0.5490114594633916, 0.5438867986921649, 0.5556644028687258, 0.5673815140890016, 0.5790237313554899, 0.5905770273552452, 0.5855681459521859, 0.5478510264204463, 0.5423989806313206, 0.5549236538314162, 0.5673815140890016, 0.5797547457564832, 0.5920260238343233, 0.5867048041828079, 0.5471079674231106, 0.5414460852085906, 0.5544492833564355, 0.5673815140890016, 0.5802225721333982, 0.5929528292812464, 0.5874318872179308, 0.5468519847905416, 0.5411177781955361, 0.5542858547163007, 0.5673815140890016, 0.5803836901863652, 0.5932719259332846, 0.5876822303015051, 0.5471079674231106, 0.5414460852085906, 0.5544492833564355, 0.5673815140890016, 0.5802225721333982, 0.5929528292812464, 0.5874318872179308, 0.5478510264204463, 0.5423989806313206, 0.5549236538314162, 0.5673815140890016, 0.5797547457564832, 0.5920260238343233, 0.5867048041828079, 0.5490114594633916, 0.5438867986921649, 0.5556644028687258, 0.5673815140890016, 0.5790237313554899, 0.5905770273552452, 0.5855681459521859, 0.5501179440118242, 0.5453327144203235, 0.556384408776538, 0.5673815140890016, 0.5783126210580789, 0.5891665988314675, 0.5844829946857041]
DENSITY_GRADIENT = [0.3100329681036418, 0.3128262897228875, 0.3236457443476525, 0.3338398417574649, 0.34375032773903313, 0.3541820746205542, 0.3567207122263846, 0.3406437867282541, 0.3427504712333693, 0.3534662132708876, 0.3633701395038443, 0.3728488727403771, 0.38283326528707656, 0.3844222004393447, 0.36614420836242406, 0.3672968340416528, 0.3774600193884651, 0.38661591624140673, 0.3951946766948338, 0.4042570440443827, 0.4046467005955308, 0.3843367254466323, 0.3842744841262181, 0.39347989547329415, 0.4015191632902449, 0.40885665976895735, 0.41664484125273415, 0.4156290630591044, 0.393518117798004, 0.3920627945573655, 0.39999055086595864, 0.40665668533472465, 0.4125462090833852, 0.4188447180723884, 0.41633162931152656, 0.39278491126375875, 0.3898876828331794, 0.3963440470481531, 0.40151916329024495, 0.4059020867037372, 0.41064642071483115, 0.40668953545753733, 0.3822106579979459, 0.37796889156796787, 0.38290578862471675, 0.3866159162414068, 0.3895741322251224, 0.39284273316932, 0.3876371387236767, 0.3628367393214092, 0.35748596218738404, 0.36098611511307965, 0.3633701395038441, 0.3650818512190609, 0.36705296420052175, 0.36090897053307164, 0.3364806028917665, 0.3303795166558507, 0.33260392276617234, 0.3338398417574649, 0.3344919379350265, 0.3353642060240624, 0.32868218924098125]


def test_existing_solid_and_density_gradients_are_unchanged():
    r = region('float64')
    p = torch.tensor([.02, -.03, .04, 11., -9., 17., 3.4, .21, .38], dtype=torch.float64, requires_grad=True)
    background = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    solids = [DifferentiableSolid.box((.43, .31, .29), center=p[:3], rotation=p[3:6], epsilon=p[6]),
              DifferentiableSolid.cylinder(p[7], p[8], center=(.16, -.08, .03), rotation=(13., -17., 8.), epsilon=3.1, radius_y=.17),
              DifferentiableSolid.ellipsoid((.18, .13, .11), center=(-.2, .12, .04), rotation=(19., 5., -11.), epsilon=2.2)]
    epsilon = smooth_geometry_epsilon(r, solids, background=background, width=.09, chunk_cells=113)
    seed = torch.randn((3, *r.shape), dtype=torch.float64, generator=torch.Generator().manual_seed(21)).movedim(0, -1)
    dp, db = torch.autograd.grad(epsilon, (p, background), seed)
    flat = epsilon.detach().flatten()
    assert flat.numel() == SOLID_FORWARD_NUMEL
    assert math.isclose(float(flat.sum()), SOLID_FORWARD_SUM, rel_tol=1e-11)
    assert math.isclose(float(flat.square().sum()), SOLID_FORWARD_SUMSQ, rel_tol=1e-11)
    torch.testing.assert_close(flat[::flat.numel()//24], torch.tensor(SOLID_FORWARD_SAMPLES, dtype=torch.float64), rtol=1e-11, atol=1e-13)
    torch.testing.assert_close(dp, torch.tensor(SOLID_GRADIENT, dtype=torch.float64), rtol=1e-11, atol=1e-13)
    assert math.isclose(float(db), SOLID_BACKGROUND_GRADIENT, rel_tol=1e-11)
    model = DensityParameterization((9, 7), spacing_um=.05, initial=torch.linspace(-1, 1, 63, dtype=torch.float64).reshape(9, 7),
                                    filter_radius_um=.08, symmetry='mirror_x', beta=3., eta=.45, dtype=torch.float64)
    density = model()
    (density*torch.linspace(.3, 1.7, 63, dtype=torch.float64).reshape(9, 7)).sum().backward()
    torch.testing.assert_close(density.detach().flatten(), torch.tensor(DENSITY_FORWARD, dtype=torch.float64), rtol=1e-11, atol=1e-13)
    torch.testing.assert_close(model.design.grad.flatten(), torch.tensor(DENSITY_GRADIENT, dtype=torch.float64), rtol=1e-11, atol=1e-13)


def relative_l2(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def test_shape_gradient_convergence_record_and_refinement_contract():
    from benchmarks.shape_gradient_polygon import TOLERANCES, VERTICES as BENCH_VERTICES, make_project
    report = json.loads(RECORD.read_text(encoding='utf8'))
    assert report['tolerances'] == TOLERANCES and report['vertices'] == [list(v) for v in BENCH_VERTICES]
    cases = report['cases']
    references = {(row['width_um'], row['step_um']): row for row in report['references']}
    for width in (.08, .04):
        fine, half = references[width, .002], references[width, .001]
        assert relative_l2(fine['gradient'], half['gradient']) < TOLERANCES['reference_step_halving']
    for row in cases+list(references.values()):
        assert row['conservation_error'] < TOLERANCES['conservation']
    series = [row for row in cases if row['group'] == 'fixed_width']
    assert [row['mesh_um'] for row in series] == [.04, .02, .01]
    errors = [relative_l2(row['gradient'], references[.08, .001]['gradient']) for row in series]
    assert errors[0] > errors[1] > errors[2] and errors[2] < TOLERANCES['fine_relative_l2']
    narrow = [row for row in cases if row['group'] == 'narrow_width']
    assert relative_l2(narrow[-1]['gradient'], references[.04, .001]['gradient']) < TOLERANCES['fine_relative_l2']
    baseline = series[-1]
    for group in ('precision_control', 'duration_control', 'pml_control'):
        row = next(row for row in cases if row['group'] == group)
        assert abs(row['transmission']-baseline['transmission']) < TOLERANCES['control_transmission'], group
        assert relative_l2(row['gradient'], baseline['gradient']) < TOLERANCES['control_gradient_relative'], group
    projects = [make_project(h) for h in (.04, .02, .01, .005)]
    for project in projects:
        r = project.region
        assert r.size == projects[0].region.size and r.precision == 'float32'
        assert math.isclose(r.steps*r.time_step, 60e-15, rel_tol=1e-14)
        assert math.isclose(r.pml_cells*r.mesh, .24, rel_tol=1e-14)
        source, first = project.sources[0], projects[0].sources[0]
        assert (source.center, source.size, source.wavelength, source.pulse_cycles) == (
            first.center, first.size, first.wavelength, first.pulse_cycles)
        assert [m.center for m in project.monitors] == [m.center for m in projects[0].monitors]
