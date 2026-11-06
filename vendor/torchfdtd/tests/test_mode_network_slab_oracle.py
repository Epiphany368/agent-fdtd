"""Tiny scalar checks for the independent slab oracle, no FDTD solves."""
import numpy as np
from benchmarks.mode_network_slab_oracle import continuum_slab, discrete_slab, C_UM_S


def test_continuum_four_equations_match_airy_and_conserve_power():
    n0, ns, k = 1.5, np.sqrt(2.6), 2*np.pi/1.55
    interface = (n0-ns)/(n0+ns)
    phase = np.exp(1j*ns*k*.4)
    r = interface*(1-phase**2)/(1-interface**2*phase**2)*np.exp(2j*n0*k*.8)
    t = (1-interface**2)*phase/(1-interface**2*phase**2)*np.exp(1j*n0*k*1.6)
    s = continuum_slab()
    np.testing.assert_allclose(s, [[r, t], [t, r]], rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(np.sum(abs(s)**2, axis=0), 1, rtol=2e-14)


def test_no_contrast_phase_and_discrete_continuum_limit():
    continuum = continuum_slab(epsilon=2.25)
    np.testing.assert_allclose(continuum[1, 0], np.exp(1j*2*np.pi/1.55*1.5*2), rtol=1e-14)
    assert abs(continuum[0, 0]) < 1e-14
    errors = []
    for h in (.1, .05, .025):
        dt = .99*h/(np.sqrt(3)*C_UM_S)
        vacuum = discrete_slab(h, dt, epsilon=2.25)
        q = 2*np.arcsin(1.5*np.sin(np.pi*C_UM_S/1.55*dt)/(.99/np.sqrt(3)))
        np.testing.assert_allclose(vacuum[1, 0], np.exp(1j*q/h*2), atol=2e-13, rtol=2e-13)
        assert abs(vacuum[0, 0]) < 1e-13
        slab = discrete_slab(h, dt)
        np.testing.assert_allclose(np.sum(abs(slab)**2, axis=0), 1, rtol=2e-13)
        errors.append(abs(slab[1, 0]-continuum_slab()[1, 0]))
    assert errors[1] < .26*errors[0] and errors[2] < .26*errors[1]
