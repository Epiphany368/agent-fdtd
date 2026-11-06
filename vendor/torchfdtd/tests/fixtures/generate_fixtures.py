"""Write the frozen project and result fixtures read by tests/test_project_compatibility.py.

The files are frozen at the schema version they were written with. When the
schema changes, keep these files as they are and add new ones next to them:
the old files then exercise the migration path. Run from the repository root:

    python tests/fixtures/generate_fixtures.py
"""
import threading
from pathlib import Path

from torchfdtd import (BoundaryFace, FieldMonitor, LorentzPole, Material, MeshRefinement, Monitor, Project, Region,
                       Simulation, Source, Structure)
from torchfdtd.models import demo_project

ROOT = Path(__file__).resolve().parent


def browser_scene():
    """A scene with the settings the workbench exposes: graded mesh, dispersion, plane monitor, float64 on the CPU."""
    return Project(
        name='Graded slab | ADE material | plane monitor',
        region=Region(dimension='2d', size=(6, 4, 2), mesh=0.1, mesh_type='graded', mesh_max=0.15, mesh_grading=1.25, mesh_ppw=8,
                      mesh_refinements=[MeshRefinement(name='fine', center=(0.8, 0, 0), size=(1.2, 1.2, 1))],
                      material_sampling='yee', steps=400, pml_cells=6, backend='cpu', precision='float64',
                      snapshot_interval=50, boundaries={'y_min': BoundaryFace(kind='pec'), 'y_max': BoundaryFace(kind='pec')}),
        materials=[Material(name='Air', index=1.0),
                   Material(name='Dispersive', model='multipole', epsilon_inf=2.0, poles=[LorentzPole()])],
        structures=[Structure(id='slab', name='slab', material='Dispersive', center=(0.8, 0, 0), size=(0.6, 4, 2))],
        sources=[Source(id='src', name='pulse', kind='plane', center=(-1.6, 0, 0), size=(0, 2, 0), pulse='broadband',
                        time_definition='wavelength', wavelength_start=1.2, wavelength_stop=1.4)],
        monitors=[Monitor(id='probe', name='probe', component='Ez', center=(1.8, 0, 0)),
                  FieldMonitor(id='plane', name='plane', normal='x', center=(1.6, 0, 0), size=(0, 2, 1))])


def tiny_result_project():
    return Project(name='tiny 2D result', region=Region(dimension='2d', size=(2, 2, .1), mesh=.1, pml_cells=3, steps=40,
                                                        snapshot_interval=10, backend='cpu'),
                   sources=[Source(id='src', wavelength=1, pulse_cycles=1)], monitors=[Monitor(id='probe', center=(.4, 0, 0))])


def save(project, path):
    """Project.save with LF line endings on every platform, so the frozen files hash the same everywhere."""
    path.write_text(project.model_dump_json(indent=2) + '\n', encoding='utf-8', newline='\n')


def main():
    projects = ROOT / 'projects'
    projects.mkdir(exist_ok=True)
    for name in ('waveguide', 'scatterer', '3d', 'pmc'):
        save(demo_project(name), projects / f'demo_{name}.json')
    save(browser_scene(), projects / 'graded_ade_plane.json')
    (projects / 'minimal_defaults.json').write_text(
        '{\n  "name": "Hand-written minimal scene",\n'
        '  "structures": [{"name": "core", "size": [8, 0.65, 0.4]}],\n'
        '  "sources": [{"center": [-2.5, 0, 0], "wavelength": 1.55}],\n'
        '  "monitors": [{"name": "output", "center": [2, 0, 0]}]\n}\n', encoding='utf-8', newline='\n')
    results = ROOT / 'npz'
    results.mkdir(exist_ok=True)
    project = tiny_result_project()
    save(project, projects / 'tiny_result.json')
    Simulation(project).run().save(results / 'tiny_2d_complete.npz')
    event = threading.Event()

    def stop_after_first_frame(state):
        if state['step'] >= 20:
            event.set()
    Simulation(project).run(progress=stop_after_first_frame, cancel=event).save(results / 'tiny_2d_cancelled.npz')


if __name__ == '__main__':
    main()
