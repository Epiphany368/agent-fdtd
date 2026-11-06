"""Independent physical checks, beyond comparing the same code on two devices."""
import numpy as np
import pytest

from torchfdtd import Project, Region, Structure, Source, Monitor, Simulation
from torchfdtd.solver import C0


def test_vacuum_pulse_propagation_speed():
    p=Project(region=Region(size=(8,6,2),mesh=.05,steps=400,pml_cells=12,backend='cpu',precision='float64'),
              sources=[Source(center=(-1.5,0,0),wavelength=1,pulse_cycles=1)],
              monitors=[Monitor(center=(-.5,0,0)),Monitor(center=(.5,0,0))])
    r=Simulation(p).run()
    use=r.times<25e-15
    near,far=r.signals[use,0],r.signals[use,1]
    correlation=np.correlate(far,near,mode='full')
    delay_steps=np.argmax(correlation)-(len(near)-1)
    measured=delay_steps*(r.times[1]-r.times[0])
    assert measured==pytest.approx(1e-6/C0,rel=.08)
    assert np.max(abs(r.electric[...,:2]))==0  # TMz decoupling.


def test_dielectric_slab_transmission_against_fresnel():
    p=Project(region=Region(size=(12,10,1),mesh=.05,steps=1500,pml_cells=12,backend='cpu',precision='float64',snapshot_interval=1500),
              structures=[Structure(size=(.25,10,1),material='SiO2 (constant n)')],
              sources=[Source(kind='plane',center=(-3,0,0),size=(0,8,0),wavelength=1.55,pulse_cycles=2)],
              monitors=[Monitor(center=(2.5,0,0))])
    # Exactly 5 cells thick. Material permittivity has no dispersive fit.
    p.structures[0].center=(.025,0,0)
    p.materials[1].index=1.5
    slab=Simulation(p).run()
    p.structures=[]
    air=Simulation(p).run()
    phase=np.exp(2j*np.pi*C0/(1.55e-6)*slab.times)
    amplitude=np.dot(phase,slab.signals[:,0])/np.dot(phase,air.signals[:,0])
    measured=abs(amplitude)**2
    n=1.5;delta=2*np.pi*n*.25/1.55
    expected=1/(1+((n*n-1)/(2*n))**2*np.sin(delta)**2)
    assert measured==pytest.approx(expected,abs=.035),f'T={measured}, analytic={expected}'
