import fdtd
import numpy as np
import pytest
import torch

from torchfdtd.boundaries import YeeGrid
from torchfdtd.materials import MaterialADE
from torchfdtd.run_control import StateDiagnostics
from test_solver import small
from test_multipole import multi_material


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('complex_fields', [False, True])
def test_fused_diagnostics_weighted_weak_states_and_nonfinite(precision, complex_fields):
    pytest.importorskip('cupy')
    dtype = torch.float32 if precision=='float32' else torch.float64
    old_dtype = torch.get_default_dtype()
    try:
        fdtd.set_backend(f'torch.cuda.{precision}')
        fdtd.backend.float = dtype
        p=small(backend='cuda',precision=precision)
        p.region.mesh_type='graded';p.region.mesh_max=.2
        if complex_fields:
            p.region.boundaries.y_min.kind=p.region.boundaries.y_max.kind='bloch'
            p.region.bloch_phase=(0,.2,0)
        g=YeeGrid(p.region)
        rng=np.random.default_rng(42)
        def values(array, scale):
            data=rng.normal(size=tuple(array.shape))*scale
            if array.is_complex():data=data+1j*rng.normal(size=tuple(array.shape))*scale
            array.copy_(torch.as_tensor(data,device=array.device,dtype=array.dtype))
        g.inverse_permittivity.fill_(.5)
        state=MaterialADE(g,multi_material(),np.array([0,3,8]),components=True)
        g.material_states=[state]
        reference=StateDiagnostics(g,fused=False)
        fused=StateDiagnostics(g)
        assert fused.backend=='fused_cuda'
        for scale in (1e-25,1e3):
            for array in (g.E,g.H,state.P,state.Q):values(array,scale)
            np.testing.assert_allclose(fused.measure(),reference.measure(),rtol=2e-13,atol=0)
            assert fused.measure()[0]>0
        # A non-finite Drude P must not escape detection just because omega=0.
        saved=state.P[0,0].clone();state.P[0,0]=float('nan')
        with pytest.raises(FloatingPointError,match='Non-finite'):fused.measure()
        state.P[0,0]=saved
        segment=next(segments[0] for segments in g.cpml.values() if segments)
        segment['psi'].reshape(-1)[-1]=float('inf')
        with pytest.raises(FloatingPointError,match='Non-finite'):fused.measure()
    finally:
        fdtd.set_backend('numpy');fdtd.backend.float=np.float64
        torch.set_default_dtype(old_dtype)
