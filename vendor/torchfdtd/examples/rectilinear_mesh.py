"""Independent axis spacing and frozen node arrays, without a GUI."""
import numpy as np
from torchfdtd import (Project,Region,Source,Monitor,Boundaries,BoundaryFace,
                        Simulation,run_tensor_batch)


def project():
    cyclic=Boundaries(**{a+'_'+side:BoundaryFace(kind='periodic') for a in 'yz' for side in ('min','max')})
    return Project(name='Rectilinear pulse',region=Region(dimension='3d',size=(6.4,3.2,3.2),
        mesh_steps=(.05,.2,.2),material_sampling='yee',pml_cells=16,steps=800,
        backend='auto',cuda_kernel='fused',boundaries=cyclic),
        sources=[Source(kind='plane',injection='oneway',normal='x',center=(-1.3,0,0),size=(0,3.2,3.2),pulse_cycles=1)],
        monitors=[Monitor(center=(1.1,0,0))])


if __name__=='__main__':
    p=project()
    # Exact node arrays are part of the JSON/Python/NPZ model, in micrometres.
    data=p.model_dump();r=data['region']
    r.update(mesh_type='explicit',mesh_steps=None,mesh_coordinates=[a.tolist() for a in p.region.mesh_nodes])
    frozen=Project.model_validate(data)
    assert frozen.region.shape==(128,16,16)
    result=Simulation(frozen).run()
    assert np.max(np.abs(result.signals))>0
    print(result.summary)
    if result.summary['backend'].startswith('cuda'):
        other=frozen.model_copy(deep=True);other.sources[0].phase=45
        batch=run_tensor_batch([frozen,other],cohort_size=2);batch.raise_for_errors()
        print('Independent cases:',len(batch.items))
