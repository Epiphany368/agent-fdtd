"""Cheap independent checks for the physical convergence benchmark contract."""
import math
import numpy as np
from benchmarks.gradient_mesh import airy, make_project


def test_airy_derivatives_against_complex_amplitude_transfer_formula():
    def transmission(d,e):
        n=math.sqrt(e)
        r=(1-n)/(1+n)
        z=np.exp(2j*np.pi*n*d/1.55)
        return abs((1-r*r)*z/(1-r*r*z*z))**2
    for d,e in ((.23,2.25),(.19,3.1),(.38,1.4)):
        h=1e-5
        expected=[transmission(d,e),(transmission(d+h,e)-transmission(d-h,e))/(2*h),
                  (transmission(d,e+h)-transmission(d,e-h))/(2*h)]
        np.testing.assert_allclose(airy(d,e),expected,rtol=2e-7,atol=2e-9)


def test_refinement_preserves_physical_domain_time_source_and_absorber():
    projects=[make_project(h) for h in (.04,.02,.01)]
    first=projects[0]
    for project in projects:
        r=project.region
        assert r.size==first.region.size
        assert r.precision=='float32'
        assert math.isclose(r.steps*r.time_step,90e-15,rel_tol=1e-14)
        assert math.isclose(r.pml_cells*r.mesh,.4,rel_tol=1e-14)
        source=project.sources[0]
        assert (source.center,source.size,source.wavelength,source.pulse_cycles)==(
            first.sources[0].center,first.sources[0].size,first.sources[0].wavelength,first.sources[0].pulse_cycles)
