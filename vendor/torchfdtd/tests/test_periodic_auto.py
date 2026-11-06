from unittest.mock import patch

import pytest
import torch

from torchfdtd import PeriodicLayerResponse, PlaneReferenceCache
from test_periodic_adjoint import SPEC, SETTINGS

MIB = 1024**2
GIB = 1024**3


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def forbidden(*args, **kwargs):
    pytest.fail('Automatic admission must not run a calibration solve or build dense geometry')


def test_auto_resident_no_calibration_and_torch_chain():
    with patch.object(PeriodicLayerResponse, 'forward', forbidden), patch(
            'torchfdtd.periodic_adjoint.periodic_density_layer', forbidden):
        model = PeriodicLayerResponse.auto(SPEC, **SETTINGS, density_shape=(2, 2),
            device='cpu', gpu_budget_bytes=GIB, host_budget_bytes=GIB,
            reference_cache=PlaneReferenceCache(MIB))
    report = model.selection_report
    assert report['mode'] == 'resident' and report['calibration_solves'] == 0
    assert len(report['attempts']) == 1
    report['mode'] = 'changed inspection copy'
    assert model.selection_report['mode'] == 'resident'
    density = torch.tensor([[.2, .4], [.5, .3]], requires_grad=True)
    response = model(density)
    response.square().sum().backward()
    assert density.grad.dtype == response.dtype == torch.float32
    assert torch.isfinite(density.grad).all() and density.grad.norm() > 0
    assert model.last_report['selection']['mode'] == 'resident'


@pytest.mark.parametrize('host_budget, expected', [(GIB, 'dram'), (40*MIB, 'file')])
def test_auto_shrinks_tiles_and_selects_admitted_storage(tmp_path, host_budget, expected):
    spec = dict(SPEC, period_um=(4., 4.))
    directory = tmp_path/'unallocated-state'
    with patch('torch.cuda.mem_get_info', return_value=(100*GIB, 100*GIB)), patch(
            'torchfdtd.state_store.disk_free', return_value=200*GIB), patch.object(
            PeriodicLayerResponse, 'forward', forbidden), patch(
            'torchfdtd.periodic_adjoint.periodic_density_layer', forbidden):
        model = PeriodicLayerResponse.auto(spec, **SETTINGS, density_shape=(2, 2),
            gpu_budget_bytes=16*MIB, host_budget_bytes=host_budget,
            reference_cache=PlaneReferenceCache(MIB), state_directory=directory,
            disk_budget_bytes=GIB)
        plan = model.plan()
    report = model.selection_report
    assert report['mode'] == expected and report['calibration_solves'] == 0
    assert report['attempts'][0]['mode'] == 'resident' and not report['attempts'][0]['admitted']
    options = report['policy']['streamed']
    assert options['tile_transfers'] == 'async' and options['slab_width'] < 40
    assert options['temporal_depth'] <= 8 and options['checkpoints'] == 2
    assert options['disk_free_reserve_bytes'] == 100*GIB
    assert plan['gpu_reservation_bytes'] <= 16*MIB
    assert plan['host_reservation_bytes'] <= host_budget
    assert (plan['disk_reservation_bytes'] > 0) == (expected == 'file')
    assert model.project.region.steps == SETTINGS['steps']
    assert model.project.region.precision == 'float32'
    assert not directory.exists()


def test_auto_never_uses_unconfigured_or_insufficient_disk(tmp_path):
    spec = dict(SPEC, period_um=(4., 4.))
    settings = dict(SETTINGS, density_shape=(2, 2), gpu_budget_bytes=16*MIB,
        host_budget_bytes=40*MIB, reference_cache=PlaneReferenceCache(MIB))
    with patch('torch.cuda.mem_get_info', return_value=(100*GIB, 100*GIB)), patch.object(
            PeriodicLayerResponse, 'forward', forbidden):
        with pytest.raises(ValueError, match='No periodic execution policy fits'):
            PeriodicLayerResponse.auto(spec, **settings)
        with patch('torchfdtd.state_store.disk_free', return_value=100*GIB):
            with pytest.raises(ValueError, match='disk budget or available disk space'):
                PeriodicLayerResponse.auto(spec, **settings,
                    state_directory=tmp_path/'unused', disk_budget_bytes=GIB)
    assert not (tmp_path/'unused').exists()
