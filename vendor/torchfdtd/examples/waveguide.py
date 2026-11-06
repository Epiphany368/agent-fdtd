from torchfdtd import Project, Region, Structure, Source, Monitor, Simulation

project = Project(
    name='SiN waveguide',
    region=Region(size=(8, 6, 2), mesh=0.05, steps=700, backend='auto'),
    structures=[Structure(name='waveguide', size=(8, 0.65, 0.4))],
    sources=[Source(center=(-2.5, 0, 0), wavelength=1.55)],
    monitors=[Monitor(name='output', center=(2, 0, 0))],
)
project.save('waveguide.json')  # Open this file in the web workbench.
result = Simulation(project).run()
result.save('results/waveguide.npz')
print(result.summary)

