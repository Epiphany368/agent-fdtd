"""CPU-only unequal fixed-guide networks and independent Fresnel checks."""
import pytest
import torch
import weakref
import numpy as np

from test_mode_network import setup
from torchfdtd import AdjointOptions
from torchfdtd.mode_network import ModeNetwork
from torchfdtd.adjoint_planes import DifferentiablePlaneResult


def network(steps=600):
    project, ports = setup(steps=steps)
    return ModeNetwork(project, ports, options=AdjointOptions(checkpoints=4, backward_kernel='torch'),
                       port_permittivities={'left': 1., 'right': 1.44})


def test_explicit_sections_contracts_and_fixed_profiles():
    project, ports = setup(steps=10)
    with pytest.raises(ValueError, match='either'):
        ModeNetwork(project, ports, 1., port_permittivities={'left': 1., 'right': 2.})
    with pytest.raises(ValueError, match='exactly'):
        ModeNetwork(project, ports, port_permittivities={'left': 1.})
    for value in (torch.tensor(1., requires_grad=True), lambda u, v: torch.tensor(1., requires_grad=True)):
        with pytest.raises(ValueError, match='Trainable profile'):
            ModeNetwork(project, ports, port_permittivities={'left': value, 'right': 1.44})
    net = network(10)
    with pytest.raises(ValueError, match='Select port'):
        net.reference_epsilon()
    with pytest.raises(ValueError, match='Unknown'):
        net.reference_epsilon(port='missing')
    epsilon = net.reference_epsilon(port='left')
    with pytest.raises(ValueError, match='exterior material'):
        net(epsilon)
    assert torch.all(net.reference_epsilon(port='right') == 1.44)


def test_unequal_native_fresnel_interior_fd_and_zero_exterior_vjp():
    net = network()
    epsilon = net.reference_epsilon(port='left')
    epsilon[20:] = 1.44
    epsilon.requires_grad_()
    result = net(epsilon)
    s = result.s
    # Independent continuum normal-incidence interface powers. At h=lambda/7.75,
    # the deliberately predeclared tolerance allows discrete interface dispersion.
    reflectance = ((1.-1.2)/(1.+1.2))**2
    transmittance = 4.*1.*1.2/(1.+1.2)**2
    assert torch.allclose(s.diagonal().abs().square(), torch.full((2,), reflectance), atol=.012, rtol=0)
    assert torch.allclose(torch.stack((s[1, 0], s[0, 1])).abs().square(), torch.full((2,), transmittance), atol=.025, rtol=0)
    assert (s-s.T).abs().max() < .012
    loss = s[1, 0].real + .3*s[0, 1].imag
    gradient, = torch.autograd.grad(loss, epsilon)
    assert torch.count_nonzero(gradient[:net.indices[0]+2]) == 0
    assert torch.count_nonzero(gradient[net.indices[1]-1:]) == 0
    mask = torch.zeros_like(epsilon); mask[20:22] = 1
    step = .004
    with torch.no_grad():
        hi = net(epsilon+step*mask).s
        lo = net(epsilon-step*mask).s
        finite = ((hi[1, 0].real+.3*hi[0, 1].imag)-(lo[1, 0].real+.3*lo[0, 1].imag))/(2*step)
    torch.testing.assert_close((gradient*mask).sum(), finite, rtol=.015, atol=4e-4)
    assert result.report['calibration_volume_limit'] == 1
    assert result.report['calibration_volume_bytes'] == epsilon.numel()*epsilon.element_size()
    print({'S': s.detach().tolist(), 'vjp': float((gradient*mask).sum()), 'fd': float(finite)})


def test_calibration_guides_are_sequential_and_not_retained(monkeypatch):
    import torchfdtd.mode_network as module
    net = network(10)
    epsilon = net.reference_epsilon(port='left'); epsilon[20:] = 1.44
    epsilon.requires_grad_()
    original = net.reference_epsilon
    references, names = [], []
    def reference(**kwargs):
        assert all(ref() is None for ref in references)
        value = original(**kwargs)
        references.append(weakref.ref(value)); names.append(kwargs['port'])
        return value
    class Fake:
        def __init__(self, project, launch, options):
            pass
        def __call__(self, value, frequency):
            a = value.mean().to(torch.complex64).reshape(1, 1, 1)
            return {name: DifferentiablePlaneResult(torch.cat((a, 2*a), -1),
                torch.tensor(frequency), torch.zeros(1, 3), torch.ones(1),
                (1, 1), 'x', 'stub', {}) for name in ('left', 'right')}
    monkeypatch.setattr(net, 'reference_epsilon', reference)
    monkeypatch.setattr(module, 'ModeInjectedPlaneSimulation', Fake)
    monkeypatch.setattr(module, '_decompose', lambda plane, launches, tolerance:
                        (plane.fields[..., 0], plane.fields[..., 1]))
    result = net(epsilon)
    assert names == ['left', 'right']
    assert all(ref() is None for ref in references)
    result.s.real.sum().backward()
    assert all(ref() is None for ref in references)
    assert epsilon.grad[18:22].abs().max() > 0


def test_discrete_interface_oracle_algebra_and_reflectionless_negative_control():
    from benchmarks.mode_network_unequal_oracle import yee_interface, MAX_COMPLEX_ERROR, C_UM_S
    dt=.99/np.sqrt(3)*.2/C_UM_S
    kwargs=dict(epsilon_left=1.,epsilon_right=1.44,h_um=.2,dt_seconds=dt)
    s,details=yee_interface(**kwargs)
    np.testing.assert_allclose(s.conj().T@s,np.eye(2),atol=2e-15,rtol=0)
    np.testing.assert_allclose(s,s.T,atol=5e-16,rtol=0)
    ql,qr=details['q_rad_per_cell']
    r,t,rr,tr=[complex(*a) for a in details['interface_electric_amplitudes']]
    # Independent check of the actual j=-1 and j=0 Yee equations.
    k2=details['kappa']**2
    e={j:np.exp(1j*ql*j)+r*np.exp(-1j*ql*j) if j<0 else t*np.exp(1j*qr*j)
       for j in (-2,-1,0,1)}
    assert abs(e[0]-(2-k2)*e[-1]+e[-2])<1e-15
    assert abs(e[1]-(2-1.44*k2)*e[0]+e[-1])<1e-15
    assert abs(tr-1-rr)<1e-15
    reflectionless=np.array([[0,s[0,1]/abs(s[0,1])],[s[1,0]/abs(s[1,0]),0]])
    assert abs(reflectionless-s).max()>20*MAX_COMPLEX_ERROR
    uniform,uniform_details=yee_interface(**dict(kwargs,epsilon_right=1.))
    expected=np.exp(1j*uniform_details['q_rad_per_cell'][0]*2/.2)
    np.testing.assert_allclose(uniform,[[0,expected],[expected,0]],rtol=0,atol=2e-15)


def test_unequal_native_complex_s_against_independent_discrete_interface():
    # The original 8-um/5-cell-CPML fixture failed this stricter gate and its
    # immutable measurement is retained. Use the diagnosed 12-um/15-cell
    # lower-reflection followup, without changing mesh or detector planes.
    from benchmarks.mode_network_unequal_cpml import run
    record=run()
    assert record['accepted'], record['complex_entry_absolute_error']
