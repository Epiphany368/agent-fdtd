"""Constraints, physical filters, derivatives and an actual streamed objective."""
import pytest
import torch

from torchfdtd.design_parameterization import DensityParameterization


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def direct_filter(values, spacing, radius, periodic):
    result = torch.empty_like(values)
    nx, ny = values.shape
    for x in range(nx):
        for y in range(ny):
            value, weight = 0., 0.
            for a in range(nx):
                for b in range(ny):
                    dx, dy = abs(x-a), abs(y-b)
                    if periodic:
                        dx, dy = min(dx, nx-dx), min(dy, ny-dy)
                    distance = ((dx*spacing[0])**2+(dy*spacing[1])**2)**.5
                    w = max(0., 1-distance/radius)
                    value = value+w*values[a, b]
                    weight += w
            result[x, y] = value/weight
    return result


@pytest.mark.parametrize('boundary', ['periodic', 'truncate'])
@pytest.mark.parametrize('radius', [.27, 1.4])
def test_physical_radius_filter_matches_independent_minimum_image_or_truncated_average(boundary, radius):
    initial = torch.linspace(.05, .95, 24, dtype=torch.float64).reshape(4, 6)
    module = DensityParameterization(initial.shape, spacing_um=(.2, .1), initial=initial,
        mode='density', filter_radius_um=radius, boundary=boundary, beta=0, dtype=torch.float64)
    expected = direct_filter(initial, (.2, .1), radius, boundary == 'periodic')
    torch.testing.assert_close(module(), expected, rtol=1e-14, atol=1e-14)
    constant = DensityParameterization((4, 6), spacing_um=(.2, .1), initial=.37,
        mode='density', filter_radius_um=radius, boundary=boundary, beta=0, dtype=torch.float64)
    torch.testing.assert_close(constant(), torch.full((4, 6), .37, dtype=torch.float64), rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize('symmetry', ['mirror_x', 'mirror_y', 'mirror_xy', 'rotate180', 'rotate90', 'dihedral4'])
def test_symmetry_is_exact_and_parameter_gradient_matches_finite_difference(symmetry):
    from torchfdtd.design_parameterization import _images
    initial = torch.linspace(-1.1, .9, 25, dtype=torch.float64).reshape(5, 5)
    module = DensityParameterization((5, 5), spacing_um=.12, initial=initial,
        symmetry=symmetry, filter_radius_um=.23, beta=3., eta=.42, dtype=torch.float64)
    result = module()
    for image in _images(result, symmetry):
        assert torch.equal(result, image)
    weights = torch.linspace(.1, 1.3, 25, dtype=torch.float64).reshape(5, 5)
    direction = torch.cos(torch.arange(25, dtype=torch.float64)).reshape(5, 5)
    def objective():
        return (module().square()*weights).sum()
    gradient, = torch.autograd.grad(objective(), module.design)
    with torch.no_grad():
        h = 1e-5
        module.design.copy_(initial+h*direction)
        high = objective()
        module.design.copy_(initial-h*direction)
        low = objective()
        module.design.copy_(initial)
    torch.testing.assert_close((gradient*direction).sum(), (high-low)/(2*h), rtol=2e-8, atol=2e-9)


def test_fixed_masks_are_exact_and_masked_parameters_have_no_indirect_gradient():
    mask = torch.zeros((5, 5), dtype=torch.bool)
    mask[0, 0] = mask[-1, 0] = mask[0, -1] = mask[-1, -1] = True
    module = DensityParameterization((5, 5), spacing_um=.1, symmetry='dihedral4',
        filter_radius_um=.21, beta=4, fixed_mask=mask, fixed_values=1.)
    result = module()
    assert torch.equal(result[mask], torch.ones(4))
    gradient, = torch.autograd.grad(result.sum(), module.design)
    assert torch.equal(gradient[mask], torch.zeros(4))
    assert gradient[~mask].abs().sum() > 0
    bad = mask.clone()
    bad[0, 0] = False
    with pytest.raises(ValueError, match='respect'):
        DensityParameterization((5, 5), spacing_um=.1, symmetry='dihedral4', fixed_mask=bad)
    with pytest.raises(ValueError, match='square grid'):
        DensityParameterization((4, 5), spacing_um=.1, symmetry='rotate90')
    with pytest.raises(ValueError, match='equal physical spacing'):
        DensityParameterization((5, 5), spacing_um=(.1, .2), symmetry='rotate90')


def test_projection_continuation_serialization_and_optimizer_resume_are_reproducible(tmp_path):
    settings = dict(shape=(4, 4), spacing_um=.1, filter_radius_um=.19, symmetry='mirror_xy')
    module = DensityParameterization(**settings)
    optimizer = torch.optim.SGD(module.parameters(), lr=.03, momentum=.8)
    weights = torch.arange(16).reshape(4, 4)/16
    (module()*weights).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    module.advance_beta(2.)
    module.set_beta(5.)
    checkpoint = tmp_path/'design.pt'
    torch.save(dict(model=module.state_dict(), optimizer=optimizer.state_dict()), checkpoint)
    restored = DensityParameterization(**settings)
    resumed = torch.optim.SGD(restored.parameters(), lr=.03, momentum=.8)
    state = torch.load(checkpoint, weights_only=True)
    restored.load_state_dict(state['model'])
    resumed.load_state_dict(state['optimizer'])
    assert float(restored.beta) == 5. and int(restored.continuation_updates) == 2
    assert torch.equal(module(), restored())
    for design, opt in ((module, optimizer), (restored, resumed)):
        (design()*weights).sum().backward()
        opt.step()
    assert torch.equal(module.design, restored.design)
    incompatible = DensityParameterization((4, 4), spacing_um=.2, filter_radius_um=.19, symmetry='mirror_xy')
    before = incompatible.design.detach().clone()
    with pytest.raises(ValueError, match='configuration differs'):
        incompatible.load_state_dict(state['model'])
    assert torch.equal(before, incompatible.design)
    wrapper = torch.nn.Sequential(incompatible)
    saved_wrapper = torch.nn.Sequential(module).state_dict()
    with pytest.raises(ValueError, match='configuration differs'):
        wrapper.load_state_dict(saved_wrapper)
    assert torch.equal(before, incompatible.design)


def test_explicit_beta_continuation_sharpens_without_advancing_on_forward():
    initial = torch.tensor([[0., .2, .5, .8, 1.]], dtype=torch.float64)
    module = DensityParameterization(initial.shape, spacing_um=.1, initial=initial,
        mode='density', beta=0, dtype=torch.float64)
    torch.testing.assert_close(module(), initial, rtol=0, atol=0)
    module.set_beta(1.)
    soft = module()
    module.advance_beta(4., maximum=3.)
    sharper = module()
    assert float(module.beta) == 3. and int(module.continuation_updates) == 2
    assert torch.all(sharper*(1-sharper) <= soft*(1-soft))
    assert torch.equal(sharper[:, (0, -1)], initial[:, (0, -1)])
    assert torch.equal(module(), sharper)
    assert int(module.continuation_updates) == 2


def test_hard_outputs_require_explicit_surrogate_and_preserve_its_label():
    module = DensityParameterization((3, 4), spacing_um=.1,
        initial=torch.linspace(-.8, .8, 12).reshape(3, 4), beta=3.)
    weights = torch.linspace(.1, 1., 12).reshape(3, 4)
    soft, = torch.autograd.grad((module()*weights).sum(), module.design)
    hard = module(hard=True)
    assert not hard.requires_grad and set(hard.flatten().tolist()) == {0., 1.}
    surrogate = module(hard=True, straight_through=True)
    assert torch.equal(surrogate, hard)
    gradient, = torch.autograd.grad((surrogate*weights).sum(), module.design)
    torch.testing.assert_close(gradient, soft, rtol=0, atol=0)
    assert 'surrogate' in module.gradient_semantics(hard=True, straight_through=True)
    with pytest.raises(ValueError, match='hard=True'):
        module(straight_through=True)
    fractional = DensityParameterization((2, 2), spacing_um=.1,
        fixed_mask=torch.ones((2, 2), dtype=torch.bool), fixed_values=.3)
    with pytest.raises(ValueError, match='binary fixed'):
        fractional(hard=True)


def test_density_mode_bounds_and_projection_are_explicit():
    module = DensityParameterization((2, 2), spacing_um=.1, mode='density', beta=0.)
    with torch.no_grad():
        module.design.copy_(torch.tensor([[-.2, .2], [.8, 1.2]]))
    before = module.design.detach().clone()
    assert torch.equal(module(), torch.tensor([[0., .2], [.8, 1.]]))
    assert torch.equal(module.design, before)
    module.project_parameters_()
    assert torch.equal(module.design, torch.tensor([[0., .2], [.8, 1.]]))
    with pytest.raises(ValueError, match='only for density'):
        DensityParameterization((2, 2), spacing_um=.1).project_parameters_()


def test_parameterized_optimizer_connects_to_real_streamed_fdtd(tmp_path):
    from test_streamed_density import make_model, objective
    solver, _ = make_model(tmp_path, steps=60)
    parameterization = DensityParameterization((3, 2), spacing_um=(.4, .6),
        initial=torch.tensor([[-.8, -.2], [.3, .1], [.6, -.4]]),
        filter_radius_um=.65, symmetry='mirror_x', beta=2.)
    optimizer = torch.optim.SGD(parameterization.parameters(), lr=.1)
    before = parameterization.design.detach().clone()
    response = solver(parameterization())
    loss = objective(response)
    loss.backward()
    assert parameterization.design.grad is not None
    assert torch.isfinite(parameterization.design.grad).all()
    assert parameterization.design.grad.norm() > 1e-7
    optimizer.step()
    assert not torch.equal(before, parameterization.design)
    assert solver.last_report['plan']['material_input'] == 'streamed_density'
    assert solver.last_report['batch']['parameter_bytes'] == 3*2*4
    assert solver.last_report['batch']['replayed_cases'] == 2
