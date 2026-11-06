"""CPU construction contracts for the opt-in recovered SC-PML algebra."""
import fdtd
import numpy as np
import pytest
from torchfdtd import BoundaryFace, Boundaries, Region, Project, Simulation
from torchfdtd.boundaries import YeeGrid
from torchfdtd.vendor_scpml import PRESETS, coefficients, face_factors


def test_default_serialization_has_no_vendor_fields():
    f=BoundaryFace()
    assert f.profile=='native'
    assert f.model_dump()==dict(kind='pml',layers=None,sigma_scale=1.,kappa=1.,alpha=1e-8,
                               polynomial=3.,alpha_polynomial=0.)
    assert BoundaryFace.model_validate(f.model_dump())==f


@pytest.mark.parametrize('preset',tuple(PRESETS))
def test_initial_layer_counts_and_explicit_overrides(preset):
    face=BoundaryFace(profile='vendor-scpml',vendor_preset=preset)
    r=Region(dimension='3d',size=(20,20,20),mesh=.1,
             boundaries=Boundaries(x_min=face))
    assert r.pml_layers(0,0)==PRESETS[preset].initial_layers
    r.boundaries.x_min=face.model_copy(update={'layers':8})
    assert r.pml_layers(0,0)==8
    assert BoundaryFace.model_validate(face.model_dump())==face


def test_native_layer_limit_preserved_and_vendor_256_allowed():
    with pytest.raises(ValueError,match='100 layers'):BoundaryFace(layers=101)
    assert BoundaryFace(profile='vendor-scpml',vendor_preset='stabilized',layers=256).layers==256


@pytest.mark.parametrize('settings',[
    dict(vendor_preset='custom'),dict(vendor_auto_scale=False),
    dict(profile='vendor-scpml',sigma_scale=2),
    dict(profile='vendor-scpml',kind='pec'),
    dict(profile='vendor-scpml',vendor_parameters={'K':4}),
    dict(profile='vendor-scpml',vendor_preset='custom',vendor_parameters={'unknown':1}),
])
def test_invalid_or_ambiguous_settings_rejected(settings):
    with pytest.raises(ValueError):BoundaryFace(**settings)


def test_missing_custom_keys_and_undefined_n1():
    for k in ('ka','b','c'):
        assert np.array_equal(coefficients(8,.5,preset='custom',parameters={},dt=1)[k],
                              coefficients(8,.5,dt=1)[k])
    with pytest.raises(ValueError,match='undefined'):coefficients(1,1,dt=1)


def test_per_face_grid_widths_and_off_switch():
    g=([0,1,3,6],[0,2,5,9],[0,4,9,15])
    dt=.99/519255768.9819587/2
    # Evaluate the recovered expression in its declared operation order.
    # Rational simplification to .5/d introduces a different floating rounding.
    expected=[min(1.,dt/(.99*d/519255768.9819587)) for d in (1,3,2,4,4,6)]
    np.testing.assert_array_equal(face_factors(g,dt,'3d'),expected)
    np.testing.assert_array_equal(face_factors(g,dt,'3d',False),np.ones(6))
    np.testing.assert_array_equal(face_factors(([0],[],[]),dt,'2d'),np.ones(6))


def test_plan_and_existing_grid_share_vendor_arrays_with_auto_scale_off():
    fdtd.set_backend('numpy');fdtd.backend.float=np.float64
    b=Boundaries(x_min=BoundaryFace(profile='vendor-scpml',vendor_preset='stabilized',
                                    layers=8,vendor_auto_scale=False))
    p=Project(region=Region(size=(4,4,1),mesh=.1,pml_cells=8,steps=10,backend='cpu',boundaries=b))
    g=YeeGrid(p.region)
    # Preparing a plan allocates coefficients but does not execute a time step.
    plan=Simulation(p).plan
    plan.verify_grid(g)
    for (forward,axis,_),segments in g.cpml.items():
        if axis!=0:continue
        for s in segments:
            if s['side']!=0:continue
            expected=coefficients(8,1 if forward else .5,preset='stabilized',dt=p.region.time_step)
            j=slice(0,8) if forward else slice(1,8)
            for key,ref in (('inv_k','ka'),('b','b'),('c','c')):
                np.testing.assert_array_equal(s[key].reshape(-1),expected[ref][j])
