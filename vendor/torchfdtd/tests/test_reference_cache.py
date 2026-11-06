from dataclasses import replace
import pytest
import torch
from torchfdtd import PlaneReferenceCache,periodic_layer_response
from test_polarization import bases


def test_lru_budget_aliasing_and_fixed_fields():
    planes,_,_=bases();p=planes[0]
    size=sum(getattr(p,k).numel()*getattr(p,k).element_size() for k in ('fields','frequency_hz','points_um','weights'))
    cache=PlaneReferenceCache(2*size)
    original=p.fields.clone()
    cache._get('a','cpu',lambda:{'plane':p})
    p.fields.zero_()
    hit=cache._get('a','cpu',lambda:pytest.fail('Unexpected compute'))
    torch.testing.assert_close(hit['plane'].fields,original)
    hit['plane'].fields.zero_()
    cache._get('b','cpu',lambda:{'plane':p})
    torch.testing.assert_close(cache._get('a','cpu',lambda:None)['plane'].fields,original)
    cache._get('c','cpu',lambda:{'plane':p})
    assert set(cache._entries)=={'a','c'} and cache.evictions==1
    assert cache.tensor_bytes==2*size
    assert all(v.fields.device.type=='cpu' for entry,_ in cache._entries.values() for v in entry.values())
    cache.clear();assert cache.tensor_bytes==0 and not cache._entries
    with pytest.raises(ValueError,match='no-grad'):
        cache._get('trainable','cpu',lambda:{'plane':replace(p,fields=original.requires_grad_())})
    small=PlaneReferenceCache(size-1)
    small._get('large','cpu',lambda:{'plane':p})
    assert small.tensor_bytes==0


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_real_layer_cache_preserves_design_gradient_and_invalidates(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    spec=dict(wavelength_um=.5,background_index=1.4,design_index=1.8,period_um=(.8,.8),
        height_um=.2,detector_offset_um=.5,theta_inside_rad=.1,phi_rad=.3)
    kwargs=dict(mesh=.1,steps=320,pml_cells=6,quadrature_counts=(4,4))
    cache=PlaneReferenceCache(1024*1024)
    d=torch.full((2,2),.3,dtype=torch.float64,device=device,requires_grad=True)
    first=periodic_layer_response(d,spec,reference_cache=cache,**kwargs)
    fg,=torch.autograd.grad(first.square().sum(),d)
    second=periodic_layer_response(d,spec,reference_cache=cache,**kwargs)
    sg,=torch.autograd.grad(second.square().sum(),d)
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    torch.testing.assert_close(fg,sg,rtol=0,atol=0)
    assert cache.misses==2 and cache.hits==2 and cache.tensor_bytes>0
    # A design change reuses the homogeneous reference, not the sample solution.
    with torch.no_grad():
        changed=periodic_layer_response(d+.1,spec,reference_cache=cache,**kwargs)
        assert not torch.allclose(changed,second)
        assert cache.hits==4
        periodic_layer_response(d,dict(spec,wavelength_um=.52),reference_cache=cache,**kwargs)
        assert cache.misses==4
        periodic_layer_response(d,spec,reference_cache=cache,**dict(kwargs,steps=321))
        assert cache.misses==6
