import fdtd
import numpy as np
import pytest
import torch

from torchfdtd import Material, Project, Structure, Simulation
from torchfdtd.boundaries import YeeGrid
from torchfdtd.materials import MaterialADE, permittivity
from torchfdtd.solver import voxelize
from test_solver import small


@pytest.mark.parametrize('model', ['drude', 'lorentz'])
def test_discrete_constitutive_response(model):
    """Drive D at one cell and independently measure D/E after settling."""
    fdtd.set_backend('numpy')
    p = small(); g = YeeGrid(p.region)
    m = Material(name='test', model=model, epsilon_inf=2, plasma_rad_s=2e15,
                 resonance_rad_s=2e15, linewidth_rad_s=3e14, collision_rad_s=6e14)
    state = MaterialADE(g, m, np.array([0]))
    f = 1/(64*g.time_step)
    electric = []
    displacement = []
    previous = 0.
    for n in range(8192):
        drive = np.sin(2*np.pi*f*(n+1)*g.time_step)
        old, response = state.prepare(g.E)
        g.E[0,0,0,0] += (drive-previous)/m.epsilon_inf
        state.correct(g.E, old, response)
        previous = drive
        if n >= 4096:
            electric.append(g.E[0,0,0,0]); displacement.append(drive)
    phase = np.exp(2j*np.pi*np.arange(4096)/64)
    measured = np.dot(displacement, phase)/np.dot(electric, phase)
    assert measured == pytest.approx(permittivity(m, f, g.time_step), rel=2e-12)
    assert measured.imag > 0


def test_material_ownership_respects_overlap_and_disabled_objects():
    p = small()
    p.materials.append(Material(name='metal', model='drude'))
    p.structures = [Structure(material='metal', size=(2,2,1), mesh_order=3),
                    Structure(material=p.materials[0].name, size=(2,2,1), mesh_order=1),
                    Structure(material='metal', size=(3,3,1), mesh_order=1, enabled=False)]
    eps, _, owner = voxelize(p, with_ownership=True)
    assert not np.any(owner == len(p.materials)-1)
    assert np.array_equal(eps, voxelize(p)[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('model', ['drude', 'lorentz'])
@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_dispersive_cuda_graph_and_cpu_agree(model, precision):
    p = small(precision=precision); p.region.steps = 400
    p.materials.append(Material(name='dispersive', model=model))
    p.structures[0].material = 'dispersive'
    p.region.boundaries.y_min.kind = p.region.boundaries.y_max.kind = 'bloch'
    p.region.bloch_phase = (0, .2, 0)
    cpu = Simulation(p).run()
    p.region.backend = 'cuda'
    gpu = Simulation(p).run()
    plain = Simulation(p).run(cuda_graph=False)
    tol = 3e-5 if precision == 'float32' else 5e-12
    for result in (gpu, plain):
        assert np.linalg.norm(cpu.electric-result.electric)/np.linalg.norm(cpu.electric) < tol
        assert np.linalg.norm(cpu.signals-result.signals)/np.linalg.norm(cpu.signals) < tol
    assert gpu.summary['cuda_graph']


def test_material_defaults_and_passivity():
    assert Material(name='old', index=2).instantaneous_epsilon == 4
    for key in ('delta_epsilon', 'epsilon_inf', 'collision_rad_s', 'linewidth_rad_s'):
        with pytest.raises(ValueError):
            Material(name='gain', **{key: -1})
    with pytest.raises(ValueError):
        permittivity(Material(name='a'), 0)


def test_material_facade_renames_references_and_rejects_unsupported_parameters():
    from torchfdtd import FDTD
    fd = FDTD(); name = fd.addmaterial('Lorentz')
    fd.addrect(material=name); fd.setmaterial(name, 'name', 'resonator')
    fd.setmaterial('resonator', 'Lorentz Linewidth', 2e14)
    assert fd.project.structures[-1].material == 'resonator'
    assert fd.getfdtdindex('resonator', [200e12])[0].imag > 0
    with pytest.raises(ValueError, match='Unsupported'):
        fd.setmaterial('resonator', 'conductivity', 1)
    with pytest.raises(ValueError, match='already exists'):
        fd.setmaterial('resonator', 'name', 'Air')
    assert fd.project.structures[-1].material == 'resonator'


def test_material_preview_rejects_singular_and_invalid_ranges(tmp_path):
    from fastapi.testclient import TestClient
    from torchfdtd.server import create_app
    with TestClient(create_app(tmp_path)) as client:
        material = Material(name='metal', model='drude').model_dump()
        response = client.post('/api/materials/preview?dt_fs=.1', json=material)
        assert response.status_code == 200
        data = response.json()
        assert len(data['n']) == 301 and min(data['k']) > 0
        assert data['numerical_n'] != data['n']
        assert client.post('/api/materials/preview?wavelength_start=2&wavelength_stop=1', json=material).status_code == 422
        assert client.post('/api/materials/preview?dt_fs=100', json=material).status_code == 422
        resonant = Material(name='undamped', model='lorentz', linewidth_rad_s=0,
                            resonance_rad_s=2*np.pi*(299792458/(1.55*1e-6))).model_dump()
        assert client.post('/api/materials/preview?wavelength_start=1.55&wavelength_stop=1.55', json=resonant).status_code == 422


@pytest.mark.parametrize('kind', [0,2,4])
def test_fsp_database_material_mapping_and_priority(kind, monkeypatch):
    from torchfdtd import fsp
    from torchfdtd.fsp_binary import FspDocument, Reader
    from torchfdtd.fsp_native import convert_fsp
    from test_fsp_native import fixture, items
    from test_fsp_binary import mapping
    monkeypatch.setattr(fsp, 'load_api', lambda: pytest.fail('Independent import loaded vendor API'))
    doc = FspDocument(fixture())
    material = dict(materialuuid='test-id',name='Controlled material',type=kind,anisotropy=0,
        permittivity=np.eye(3)*2,priority=7,omegaplasma=np.eye(3)*2e15,nuplasma=np.eye(3)*1e14,
        omegalorentz=np.eye(3)*1.7e15,deltalorentz=np.eye(3)*1e14,epsilonlorentz=np.eye(3)*1.2)
    record = Reader(mapping(items(material))).mapping()
    doc.materials = [record]
    sphere = next(n for n in doc.nodes() if n.legacy.get('kind') == 8)
    sphere.properties['materialuuid'].value = 'test-id'
    sphere.legacy['override_mesh_order'] = False
    converted = convert_fsp(doc)
    assert converted.project is not None, converted.issues
    p = converted.project
    assert p.materials[0].model == {0:'dielectric',2:'drude',4:'lorentz'}[kind]
    assert p.structures[0].mesh_order == 7
    if kind == 2:
        assert any(issue['code'] == 'drude_curve_accuracy' for issue in converted.issues)
    record['anisotropy'].value = 1
    assert convert_fsp(doc).project is None
    record['anisotropy'].value = 0
    record['permittivity'].value[1,1] = 3
    assert convert_fsp(doc).project is None
    record['permittivity'].value[1,1] = 2
    record['type'].value = 7
    assert convert_fsp(doc).project is None


@pytest.mark.parametrize('sampling', ['cell', 'yee'])
def test_dispersive_material_is_frozen_inside_pml(sampling):
    """A Lorentz slab reaching into the PML keeps its ADE only in the interior."""
    fdtd.set_backend('numpy')
    from torchfdtd.materials import configure_materials, pml_cell_mask, frozen_pml_frequency_hz
    p = small(dimension='3d'); p.region.material_sampling = sampling; p.region.pml_dispersion = 'frozen'
    p.materials.append(Material(name='sin', model='lorentz', epsilon_inf=1, resonance_rad_s=1.37e16,
                                linewidth_rad_s=1e13, delta_epsilon=3.0))
    p.structures = [Structure(material='sin', size=(4, 4, 1))]          # spans the whole x/y extent, PML included
    p = Project.model_validate(p.model_dump())
    eps, _, owner = voxelize(p, with_ownership=True)
    g = YeeGrid(p.region); g.inverse_permittivity[:] = 1/(eps if eps.ndim == 4 else eps[..., None])
    configure_materials(g, p, owner)
    pml = pml_cell_mask(p.region, owner.shape).reshape(-1)
    if owner.ndim == 4:
        pml = np.repeat(pml, owner.shape[3])
    sin = owner.reshape(-1) == len(p.materials)-1
    assert np.any(sin & pml) and np.any(sin & ~pml)
    (state,) = g.material_states
    assert not np.any(pml[state.indices])                           # no ADE state inside the PML
    assert np.count_nonzero(sin & ~pml) == len(state.indices)      # every interior cell keeps it
    expected = 1/permittivity(p.materials[-1], frozen_pml_frequency_hz(p)).real
    flat = g.inverse_permittivity.reshape(-1) if owner.ndim == 4 else g.inverse_permittivity.reshape(-1, 3)[:, 0]
    assert np.allclose(flat[sin & pml], expected)
    assert np.allclose(flat[sin & ~pml], 1/p.materials[-1].epsilon_inf)
    p.region.pml_dispersion = 'ade'
    g2 = YeeGrid(p.region); g2.inverse_permittivity[:] = 1/(eps if eps.ndim == 4 else eps[..., None])
    configure_materials(g2, p, owner)
    assert len(g2.material_states[0].indices) == np.count_nonzero(sin)


def test_frozen_pml_rejects_negative_permittivity_before_the_run_diverges():
    """A Drude slab reaching into the PML has Re(eps) < 0 at the source centre; freezing it must refuse, not clamp."""
    from torchfdtd import Region, Source, Monitor
    from torchfdtd.materials import configure_materials, frozen_pml_frequency_hz
    metal = Material(name='metal', model='drude', epsilon_inf=1.0, plasma_rad_s=1.2e16, collision_rad_s=1e14)
    p = Project(name='t', region=Region(dimension='2d', size=(2., 2., .1), mesh=.1, pml_cells=3, steps=120,
                                        material_sampling='yee', backend='cpu', pml_dispersion='frozen'),
                materials=[Material(name='Air', index=1), metal],
                structures=[Structure(id='b', name='b', kind='rectangle', center=(0., .6, 0.), size=(4., .4, .1), material='metal')],
                sources=[Source(id='s', name='s', kind='point', component='Ez', center=(-.3, 0., 0.), wavelength=1.0, pulse_cycles=2)],
                monitors=[Monitor(id='m', name='m', component='Ez', center=(.3, .2, 0.))])
    f0 = frozen_pml_frequency_hz(p)
    eps = permittivity(metal, f0).real
    assert eps < 0
    fdtd.set_backend('numpy')
    epsilon, _, owner = voxelize(p, with_ownership=True)
    g = YeeGrid(p.region); g.inverse_permittivity[:] = 1/(epsilon if epsilon.ndim == 4 else epsilon[..., None])
    with pytest.raises(ValueError, match=r'metal: pml_dispersion="frozen" has no meaning for a real permittivity of -39\.47') as info:
        configure_materials(g, p, owner)
    assert f'{f0:.6g} Hz' in str(info.value)
    assert np.all(g.inverse_permittivity > 0) and np.all(g.inverse_permittivity <= 1)   # nothing clamped to 1/1e-3
    with pytest.raises(ValueError, match='metal: pml_dispersion="frozen" has no meaning'):
        Simulation(p).run()
    # The same slab off the PML is allowed: the frozen path never touches it.
    p.structures[0].center = (0., .3, 0.); p.structures[0].size = (1.0, .4, .1)
    p = Project.model_validate(p.model_dump())
    assert Simulation(p).run().summary['steps'] == 120
