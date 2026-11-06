"""Run mixed grids and durations using python examples/grouped_batch.py."""
from torchfdtd import (BatchCase, Material, Monitor, Project, Region, Source,
                        Structure, plan_grouped_batch, run_grouped_batch)


def main():
    cases = []
    for radius in (.25, .35, .45, .55):
        for mesh, steps in ((.1, 400), (.075, 600)):
            p = Project(name=f'sphere-{radius}-{mesh}',
                region=Region(size=(4.8, 4.8, 4.8), mesh=mesh, steps=steps,
                    pml_cells=6, backend='cuda', cuda_kernel='fused', precision='float32',
                    material_sampling='yee', snapshot_interval=steps),
                materials=[Material(name='dielectric', index=1.5)],
                structures=[Structure(kind='sphere', radius=radius, material='dielectric')],
                sources=[Source(center=(-1.2, 0, 0), wavelength=1.55, pulse_cycles=2)],
                monitors=[Monitor(center=(1.2, 0, 0))])
            cases.append(BatchCase(p.name, p, {'radius': radius, 'mesh': mesh}))
    plan = plan_grouped_batch(cases, cohort_size=4)
    print('Groups:', plan['group_count'], 'Cohorts:', [c['indices'] for c in plan['cohorts']])
    report = run_grouped_batch(cases, cohort_size=4,
        objective=lambda r: float(abs(r.signals).max()))
    report.raise_for_errors()
    for item in report.items:
        print(item.id, item.parameters, item.metrics)
    print('Complete call seconds:', report.seconds)


if __name__ == '__main__':
    main()
