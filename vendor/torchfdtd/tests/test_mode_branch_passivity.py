"""Passivity of open branch networks with the radiation defect measured independently.

The guided S matrix of an open junction is not unitary: power that leaves
through radiation or unselected modes is missing from every column. This
test measures that missing power without the modal projection, from the net
Poynting flux through a closed box of full-extent planes around the
junction, driven by the same fixed modal launch. The column power must not
exceed one, every guided power must not exceed the total flux through its
plane, and the flux through the box must close on the incident power within
the program's complete-channel-balance limit, so that the defect
1 - sum |S|^2 is accounted for by measured power rather than being an
unexplained loss. S^H S = I is not asserted anywhere.

The box is the 0.1 um Y-branch and crossing geometry of test_mode_branches
on a 0.05 um mesh: at 0.1 um the flux quadrature alone misses 2 percent of
the incident power on the crossing (an O(h^2) error, 0.34 percent at
0.05 um), which is outside the 1 percent balance limit.
"""
import numpy as np
import pytest
import torch

from torchfdtd import AdjointOptions, Project, Region, Source
from torchfdtd.models import FieldMonitor
from torchfdtd.mode_branches import ModeBranchNetwork, prepare_aperture_modal_launch
from torchfdtd.mode_injection import ModeInjectedPlaneSimulation
from test_mode_branches import (CORE, CLAD, crossing_ports, crossing_sections, device_name, yee_epsilon,
                                ybranch_epsilon, ybranch_ports, ybranch_sections)

C0 = 299792458.0
MESH, PML_UM, STEPS = .05, .8, 2400   # 229 fs window; the closure is unchanged at 3200 steps
CLOSURE_BUDGET = .01       # completion_gates.json lossless_complete_channel_balance_absolute_error
GUIDED_EXCESS = .01        # guided power may not exceed the full-plane flux by more than this
PASSIVITY_MARGIN = .002    # the margin the existing branch tests record for column power
STRAIGHT_SIDE_LEAK = .01   # side flux of the calibration guide relative to its incident power


def fine_branch_project():
    """test_mode_branches.branch_project at half the mesh step and the same physical PML."""
    layers = round(PML_UM/MESH)
    faces = {a+'_'+s: dict(kind='pml', layers=layers) for a in 'xy' for s in ('min', 'max')}
    faces.update({'z_'+s: dict(kind='periodic') for s in ('min', 'max')})
    r = Region(dimension='3d', size=(4.8, 4.8, .5), mesh=MESH, pml_cells=layers, steps=STEPS,
               material_sampling='yee', boundaries=faces, precision='float32')
    return Project(region=r, sources=[Source(kind='plane', normal='x', center=(-1.4, 0., 0.),
                   size=(0., 1.2, .5), pulse_cycles=2)], monitors=[])


def box_planes(extent):
    """Closed box through the x = +-1 um port planes, spanning +-extent in y."""
    planes = []
    for name, normal, centre in (('west', 'x', (-1., 0., 0.)), ('east', 'x', (1., 0., 0.)),
                                 ('south', 'y', (0., -extent, 0.)), ('north', 'y', (0., extent, 0.))):
        size = tuple(0. if a == normal else (.5 if a == 'z' else (2*extent if normal == 'x' else 2.))
                     for a in 'xyz')
        planes.append(FieldMonitor(id=name, normal=normal, center=centre, size=size))
    return planes


def outward_flux(planes):
    """Signed outward Poynting flux through each face of the box."""
    return dict(west=-float(planes['west'].flux()), east=float(planes['east'].flux()),
                south=-float(planes['south'].flux()), north=float(planes['north'].flux()))


CASES = {
    'ybranch': dict(ports=ybranch_ports, sections=ybranch_sections, epsilon=ybranch_epsilon, extent=1.4,
                    transmitted=('east',), guided_out=(1, 2)),
    'crossing': dict(ports=crossing_ports, sections=crossing_sections,
                     epsilon=lambda region, device: yee_epsilon(
                         region, lambda x, y, z: np.where((np.abs(y) < .2-1e-9) | (np.abs(x) < .2-1e-9), CORE, CLAD), device),
                     extent=1., transmitted=('east', 'south', 'north'), guided_out=(1, 2, 3)),
}


@pytest.mark.parametrize('name', list(CASES))
def test_branch_network_passivity_with_flux_closure_and_radiation_defect(name):
    case = CASES[name]
    device = device_name()
    p = fine_branch_project()
    ports = case['ports']()
    network = ModeBranchNetwork(p, ports, options=AdjointOptions(checkpoints=4), port_permittivities=case['sections']())
    epsilon = case['epsilon'](p.region, device)
    with torch.no_grad():
        result = network(epsilon)
    s = result.s
    column_power = torch.tensor(result.report['column_power'])
    assert column_power.max() < 1+PASSIVITY_MARGIN

    # The same launch as the network's first column, observed on full planes.
    boxed = p.model_copy(deep=True)
    boxed.monitors = box_planes(case['extent'])
    boxed = Project.model_validate(boxed.model_dump())
    section = case['sections']()[ports[0].name]
    launch = prepare_aperture_modal_launch(boxed, section, mode_index=0, num_modes=1)
    simulation = ModeInjectedPlaneSimulation(boxed, launch, AdjointOptions(checkpoints=4))
    frequency = [C0/(launch.mode.wavelength_um*1e-6)]
    straight = yee_epsilon(p.region, lambda x, y, z: section(y, z), device)
    with torch.no_grad():
        calibration = outward_flux(simulation(straight, frequency))
        junction = outward_flux(simulation(epsilon, frequency))
    incident = -calibration['west']
    assert incident > 0
    # The calibration guide carries the launch through the box without leaks.
    assert abs(calibration['east']/incident-1) < CLOSURE_BUDGET
    assert abs(calibration['south'])+abs(calibration['north']) < STRAIGHT_SIDE_LEAK*incident

    fractions = {face: value/incident for face, value in junction.items()}
    closure = sum(fractions.values())
    reflected_total = 1+fractions['west']
    transmitted_total = sum(fractions[face] for face in case['transmitted'])
    guided_reflection = float(s[0, 0].abs().square())
    guided_transmission = float(sum(s[i, 0].abs().square() for i in case['guided_out']))
    defect = 1-float(column_power[0])
    side_radiation = sum(fractions[face] for face in ('south', 'north', 'east')
                         if face not in case['transmitted'])
    assert abs(closure) < CLOSURE_BUDGET
    assert guided_reflection < reflected_total+GUIDED_EXCESS
    assert guided_transmission < transmitted_total+GUIDED_EXCESS
    # With the box closed, the defect equals the measured non-guided power up
    # to the closure error: defect - nonguided == closure identically.
    nonguided = (reflected_total-guided_reflection)+(transmitted_total-guided_transmission)+side_radiation
    print({'case': name, 'device': device, 'incident_flux': incident, 'outward_fractions': fractions,
           'closure': closure, 'column_power': column_power.tolist(), 'defect': defect,
           'guided_reflection': guided_reflection, 'reflected_total': reflected_total,
           'guided_transmission': guided_transmission, 'transmitted_total': transmitted_total,
           'side_radiation': side_radiation, 'nonguided_measured': nonguided,
           'calibration_east_over_incident': calibration['east']/incident,
           'calibration_side_fraction': (abs(calibration['south'])+abs(calibration['north']))/incident,
           'reciprocity': float((s-s.T).abs().max()), 'time_step_s': p.region.time_step})
