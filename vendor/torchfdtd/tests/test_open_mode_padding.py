"""Two fixed-mesh padding/PML checks, independent of the mesh-refinement gate.

Historical baseline is reused, not recomputed: root reported the h=.1 result
from test_open_mode_ports.py::test_fully_confined_fiber_two_mesh_continuum_comparison
on the open-mode-ports worktree. Its square transverse domain is 4um, PML .8um,
core index2.2, cladding1, radius.4um, wavelength1.55um. This is a finite test
of box/PML sensitivity, not an observed convergence order or general guarantee.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from torchfdtd import Region
from torchfdtd.open_mode_ports import solve_open_waveguide_modes


BASELINE_BETA = complex(7.311263320357341,-3.1240177914815926e-8)
BETA_SHIFT_LIMIT = 1e-4
TAIL_LIMIT = 1e-4
MODE_BUDGET_BYTES = int(1.4*1024**3)
# Declared before either solve. Only these two new eigensolves are performed.
CASES = ((4.8,.8),(4.8,1.0))


def test_fixed_mesh_physical_padding_and_pml_sensitivity(record_property):
    root=Path(__file__).resolve().parents[1]
    paths=('torchfdtd/open_mode_ports.py','torchfdtd/open_mode_operators.py',
           'torchfdtd/boundaries.py','tests/test_open_mode_padding.py')
    def hashes():
        return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in paths}
    before=hashes()
    rows=[]
    for size,pml in CASES:
        region=Region(dimension='3d',size=(size,size,1.6),mesh=.1,steps=10,
            boundaries={axis+'_'+side:dict(kind='pml',layers=round(pml/.1) if axis!='z' else 3)
                        for axis in 'xyz' for side in ('min','max')},
            material_sampling='yee',precision='float32',backend='cpu')
        start=time.perf_counter()
        mode=solve_open_waveguide_modes(
            lambda x,y:np.where(x*x+y*y<.4**2-1e-10,2.2**2,1.),
            region=region,normal='z',wavelength_um=1.55,cladding_epsilon=1.,
            num_modes=1,mode_budget_bytes=MODE_BUDGET_BYTES,confinement_tolerance=TAIL_LIMIT)[0]
        shift=abs(mode.beta_per_um-BASELINE_BETA)/abs(BASELINE_BETA)
        row=dict(outer_size_um=size,pml_thickness_um=pml,
            physical_transverse_size_um=size-2*pml,mesh_um=.1,shape=list(region.shape),
            beta_per_um=[mode.beta_per_um.real,mode.beta_per_um.imag],
            relative_shift_from_historical_baseline=shift,
            collar_field_norm_fraction=mode.diagnostics['collar_energy_fraction'],
            eigenpair_residual=mode.eigenpair_residual,maxwell_residual=mode.maxwell_residual,
            engineering_reservation_bytes=mode.diagnostics['engineering_reservation_bytes'],
            wall_seconds=time.perf_counter()-start)
        rows.append(row)
        print(json.dumps(row,sort_keys=True))
    pml_shift=abs(complex(*rows[1]['beta_per_um'])-complex(*rows[0]['beta_per_um']))/abs(complex(*rows[0]['beta_per_um']))
    report=dict(criteria=dict(relative_beta_shift_max=BETA_SHIFT_LIMIT,tail_max=TAIL_LIMIT),
        baseline=dict(beta_per_um=[BASELINE_BETA.real,BASELINE_BETA.imag],
            outer_size_um=4.,pml_thickness_um=.8,mesh_um=.1,recomputed=False,
            provenance='Root-reported previous test_open_mode_ports.py::test_fully_confined_fiber_two_mesh_continuum_comparison h=.1 result, original runtime hash not supplied'),
        cases=rows,relative_shift_between_new_pml_cases=pml_shift,
        mode_budget_bytes=MODE_BUDGET_BYTES,source_sha256_before=before,
        source_sha256_after=hashes(),scope='Fixed mesh, padding/PML sensitivity only. No FDTD or GPU.')
    record_property('open_mode_padding_measurements',json.dumps(report,sort_keys=True))
    print(json.dumps(report,sort_keys=True))
    # Evaluate after recording both planned cases, preserving every measured row.
    assert all(row['relative_shift_from_historical_baseline']<=BETA_SHIFT_LIMIT for row in rows)
    assert pml_shift<=BETA_SHIFT_LIMIT
    assert all(row['collar_field_norm_fraction']<=TAIL_LIMIT for row in rows)
    assert all(row['engineering_reservation_bytes']<=MODE_BUDGET_BYTES for row in rows)
