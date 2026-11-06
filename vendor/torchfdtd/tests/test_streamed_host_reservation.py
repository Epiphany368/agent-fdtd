"""Dense-parameter host reservation follows the measured ledger scope.

The host ledger (benchmarks/streamed_host_ledger.py) measured two full-size
host tensors in backward and none in forward for a contiguous real CPU scalar
epsilon with real fields, synchronous reusable CUDA tiles, file-backed banks and
point observations. That scope reserves four parameter copies. Every other path
keeps the conservative eight. The metadata-only estimate must agree.
"""
from dataclasses import replace

import pytest
import torch

from torchfdtd import BoundaryFace, StreamedAdjointOptions
from torchfdtd.streamed import _reservation, estimate_streamed_memory
from test_differentiable import project


def options(tmp_path, **overrides):
    values = dict(device='cuda', slab_width=4, temporal_depth=2, checkpoints=1, state_storage='disk',
                  state_directory=tmp_path, disk_budget_bytes=64 * 1024 ** 2)
    values.update(overrides)
    return StreamedAdjointOptions(**values)


@pytest.fixture
def scene(monkeypatch):
    monkeypatch.setattr('torchfdtd.streamed.host_memory', lambda: dict(available_bytes=1024 ** 4))
    monkeypatch.setattr('torchfdtd.state_store.disk_free', lambda _: 1024 ** 4)
    monkeypatch.setattr('torchfdtd.streamed.cuda_budget_limit', lambda device, gpu, budget: budget)
    return project('3d', precision='float32', steps=10)


def test_measured_scope_reserves_four_parameter_copies(scene, tmp_path):
    eps = torch.full(scene.region.shape, 1.7, dtype=torch.float32)
    report = _reservation(scene, eps, options(tmp_path))
    assert report['dense_parameter_multiplier'] == 4
    assert report['dense_parameter_reservation_bytes'] == 4 * eps.numel() * eps.element_size()
    wide = torch.full((*scene.region.shape[:2], 2 * scene.region.shape[2]), 1.7, dtype=torch.float32)[..., ::2]
    assert wide.shape == eps.shape and not wide.is_contiguous()
    fallback = _reservation(scene, wide, options(tmp_path))
    assert fallback['dense_parameter_multiplier'] == 8
    assert fallback['host_reservation_bytes'] - report['host_reservation_bytes'] == 4 * eps.numel() * eps.element_size()


@pytest.mark.parametrize('name', ['cpu_tiles', 'host_banks', 'async_tiles', 'no_reuse', 'complex_fields', 'diagonal'])
def test_paths_outside_the_ledger_keep_eight_copies(scene, tmp_path, name):
    eps = torch.full(scene.region.shape, 1.7, dtype=torch.float32)
    settings = options(tmp_path)
    if name == 'cpu_tiles':
        settings = replace(settings, device='cpu')
    elif name == 'host_banks':
        settings = options(tmp_path, state_storage='host', state_directory=None, disk_budget_bytes=None)
    elif name == 'async_tiles':
        settings = replace(settings, tile_transfers='async')
    elif name == 'no_reuse':
        settings = replace(settings, reuse_tile_buffers=False)
    elif name == 'complex_fields':
        scene.region.boundaries.x_min = BoundaryFace(kind='bloch')
        scene.region.boundaries.x_max = BoundaryFace(kind='bloch')
        scene.region.bloch_phase = (.4, 0, 0)
    elif name == 'diagonal':
        eps = torch.full((*scene.region.shape, 3), 1.7, dtype=torch.float32)
    report = _reservation(scene, eps, settings)
    assert report['dense_parameter_multiplier'] == 8
    assert report['dense_parameter_reservation_bytes'] == 8 * eps.numel() * eps.element_size()


def test_metadata_estimate_matches_execution_reservation(scene, tmp_path):
    eps = torch.full(scene.region.shape, 1.7, dtype=torch.float32)
    executed = _reservation(scene, eps, options(tmp_path))
    estimated = estimate_streamed_memory(scene, options(tmp_path))
    assert estimated['dense_parameter_multiplier'] == executed['dense_parameter_multiplier'] == 4
    assert estimated['host_reservation_bytes'] == executed['host_reservation_bytes']
    diagonal = estimate_streamed_memory(scene, options(tmp_path), diagonal=True)
    assert diagonal['dense_parameter_multiplier'] == 8
