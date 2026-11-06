"""One cell per full Yee step is sufficient and sometimes necessary in x."""
import pytest
import torch

from torchfdtd import BoundaryFace, Project, Region
from torchfdtd.differentiable import _System
from torchfdtd.spacetime import SlabBlockOperator


class UnderHalo(SlabBlockOperator):
    def tiles(self,depth):
        return super().tiles(depth-1)


@pytest.mark.parametrize('depth',[1,3,6])
def test_impulse_reaches_exact_causal_edge_and_exposes_insufficient_halo(depth):
    region=Region(dimension='3d',size=(3.2,1.2,1.2),mesh=.1,steps=12,
                  precision='float64',pml_cells=3)
    region.boundaries.x_min=region.boundaries.x_max=BoundaryFace(kind='periodic')
    p=Project(region=region,sources=[],monitors=[])
    epsilon=torch.full(region.shape,1.7,dtype=torch.float64)
    host=_System(p,epsilon,prepare_updates=False)
    # A plane H_y impulse excites the x second difference through E_z.
    # It reaches cell 16 from distance K after exactly K complete E/H steps.
    for distance in (depth,depth+1):
        initial=tuple(torch.zeros_like(s) for s in host.state())
        initial[1][16+distance,:,:,1]=1.
        expected=initial
        for step in range(depth):expected=host.reference_step(expected,step,epsilon)
        actual,_=SlabBlockOperator(host,1,'cpu').forward(epsilon,initial,0,depth)
        for got,want in zip(actual,expected):torch.testing.assert_close(got,want,rtol=0,atol=0)
        edge=expected[1][16,:,:,1]
        if distance==depth:
            assert float(edge.abs().max())>0
            insufficient,_=UnderHalo(host,1,'cpu').forward(epsilon,initial,0,depth)
            assert float(insufficient[1][16,:,:,1].abs().max())==0
        else:
            assert float(edge.abs().max())==0
