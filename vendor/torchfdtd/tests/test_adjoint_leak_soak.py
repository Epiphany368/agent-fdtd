"""Adjoint leak soak: the recorded 150-iteration soak of every differentiable path, its rendered
document, and 30-iteration CPU regressions that assert no growth of live tensors, objects or
solver state holders after a short warm-up."""
import gc
import json
import os
from pathlib import Path

import psutil
import pytest
import torch

from benchmarks import adjoint_leak_soak as soak
from torchfdtd import (Project, Region, Structure, Source, Monitor, FieldMonitor, Material, Boundaries, BoundaryFace,
                       AdjointOptions, DifferentiableSimulation, DifferentiablePlaneSimulation, DispersiveSimulation,
                       StreamedSimulation, StreamedAdjointOptions, tensor_from_project)
from torchfdtd.reversible import ReversibleSimulation
from torchfdtd.source_adjoint import SourceWaveformSimulation

ROOT = Path(__file__).resolve().parents[1]
CASE = json.loads((ROOT/'docs'/'validation'/'cases'/'ADJOINT_LEAK_SOAK.json').read_text(encoding='utf-8'))
RECORD_PATH = ROOT/'docs'/'validation'/'adjoint_leak_soak_3060.json'


@pytest.fixture(scope='module')
def record():
    return json.loads(RECORD_PATH.read_text(encoding='utf-8'))


# ----------------------------------------------------------------------------------------------- the record
def test_record_covers_every_path_with_the_declared_loop(record):
    assert {p['path'] for p in record['paths']} == set(soak.PATHS)
    assert record['acceptance'] == CASE['acceptance']
    assert record['iterations'] == 150 and record['warm_up'] == 20 and record['sample_every'] == 10
    for out in record['paths']:
        assert out['error'] is None, (out['path'], out['error'])
        assert [s['iteration'] for s in out['samples']] == list(range(10, 151, 10)), out['path']
        assert out['first_loss'] is not None and out['last_loss'] is not None


def test_every_verdict_follows_from_its_samples_and_the_case_limits(record):
    for out in record['paths']:
        again = soak.judge(out['samples'], CASE['acceptance'])
        assert again['passed'] == out['judgement']['passed'] == out['passed'], out['path']
        for key, check in again['checks'].items():
            assert check == out['judgement']['checks'][key], (out['path'], key)
    failing = [p['path'] for p in record['paths'] if not p['passed']]
    assert failing == record['failing']
    for name in failing:
        assert any(name in finding for finding in record['findings']), f'{name} fails without a recorded finding'


def test_growth_measures_are_within_limits_for_every_passing_path(record):
    limits = CASE['acceptance']
    for out in record['paths']:
        if not out['passed']:
            continue
        c = out['judgement']['checks']
        assert c['torch_allocated']['growth'] <= limits['torch_allocated_growth_max_bytes']
        assert c['torch_reserved']['growth'] <= limits['torch_reserved_growth_max_bytes']
        assert c['live_tensors']['growth'] <= limits['live_tensors_growth_max']
        assert c['cuda_graphs']['growth'] <= limits['cuda_graphs_growth_max']
        assert c['gc_objects']['growth'] <= limits['gc_objects_growth_max']
        assert c['rss']['growth'] <= limits['rss_growth_max_bytes']
        for key, check in c.items():
            if key.startswith(('cache:', 'instances:')):
                assert check['growth'] <= limits['cache_growth_max'], (out['path'], key, check)


def test_document_is_rendered_from_the_record(record):
    document = (ROOT/'docs'/'ADJOINT_LEAK_SOAK.md').read_text(encoding='utf-8')
    assert document == soak.render(record)
    for out in record['paths']:
        assert f"| {out['path']} |" in document


# ----------------------------------------------------------------------------------------------- CPU regressions
PULSE = dict(pulse='gaussian', wavelength=1.3, pulse_cycles=2, pulse_offset=6e-15)


def tiny_2d(monitors=None, sources=None, steps=24):
    r = Region(dimension='2d', size=(2., 1.6, 1.), mesh=.1, pml_cells=3, steps=steps, precision='float64', backend='cpu',
               material_sampling='yee', snapshot_interval=10000)
    return Project(region=r, sources=sources or [Source(component='Ez', center=(-.5, .1, 0), **PULSE)],
                   monitors=monitors or [Monitor(component='Ez', center=(.5, 0, 0))])


def tiny_3d(faces=None, materials=(), structures=(), size=(1.2, 1.2, 1.2), steps=16):
    kwargs = dict(boundaries=Boundaries(**faces)) if faces else {}
    r = Region(dimension='3d', size=size, mesh=.1, pml_cells=3, steps=steps, precision='float32', backend='cpu',
               material_sampling='yee', snapshot_interval=10000, **kwargs)
    return Project(region=r, materials=[Material(name='void', index=1), *materials], structures=list(structures),
                   sources=[Source(component='Ez', center=(-.2, .1, 0), **PULSE)], monitors=[Monitor(component='Ez', center=(.2, 0, 0))])


def eps64(p, value=1.8):
    return torch.nn.Parameter(torch.full(p.region.shape, value, dtype=torch.float64))


def build(name):
    if name == 'differentiable':
        p = tiny_2d()
        model = DifferentiableSimulation(p, AdjointOptions(checkpoints=2))
        eps = eps64(p)
        return [eps], lambda: model(eps).signals.square().mean()
    if name == 'plane':
        p = tiny_2d(monitors=[FieldMonitor(center=(.4, 0, 0), size=(0, .6, 1.), normal='x', downsample=2)])
        model = DifferentiablePlaneSimulation(p, AdjointOptions(checkpoints=2))
        eps = eps64(p)
        frequency = torch.tensor([2.1e14, 2.6e14], dtype=torch.float64)
        return [eps], lambda: sum(v.fields.abs().square().sum() for v in model(eps, frequency, block_size=4).values())
    if name == 'dispersive':
        p = tiny_2d()
        model = DispersiveSimulation(p, AdjointOptions(checkpoints=2))
        eps = eps64(p)
        strength = torch.nn.Parameter(torch.tensor([.8, .4], dtype=torch.float64))
        omega = torch.nn.Parameter(torch.tensor([0., 1.7], dtype=torch.float64))
        gamma = torch.nn.Parameter(torch.tensor([.2, .3], dtype=torch.float64))
        return [eps, strength, omega, gamma], lambda: model(eps, strength*1e30, omega*1e15, gamma*1e15).signals.square().mean()
    if name == 'source_waveform':
        p = tiny_2d(sources=[Source(component='Ez', center=(-.5, .1, 0), **PULSE), Source(component='Ey', center=(-.4, -.2, 0), **PULSE)])
        model = SourceWaveformSimulation(p, AdjointOptions(checkpoints=2))
        eps = eps64(p)
        waves = torch.nn.Parameter(model.default_waveforms())
        return [eps, waves], lambda: model(eps, waves).signals.square().mean()
    if name == 'streamed':
        p = tiny_2d()
        model = StreamedSimulation(p, StreamedAdjointOptions(device='cpu', slab_width=6, temporal_depth=4, checkpoints=2))
        eps = eps64(p)
        return [eps], lambda: model(eps).signals.square().mean()
    if name == 'reversible':
        faces = {f'{a}_{s}': BoundaryFace(kind='periodic') for a in 'xyz' for s in ('min', 'max')}
        p = tiny_3d(faces=faces, size=(.8, .8, .8))
        model = ReversibleSimulation(p)
        eps = torch.nn.Parameter(torch.full(p.region.shape, 1.8, dtype=torch.float32).contiguous())
        return [eps], lambda: model(eps).signals.square().mean()
    if name == 'tensor_project':
        tensor = Material(name='tensor', model='tensor', epsilon_tensor=(2.4, 2.1, 2.7, .15, 0., .1))
        p = tiny_3d(materials=[tensor], structures=[Structure(name='block', center=(0, 0, 0), size=(.2, .2, .2), material='tensor')])
        adapter = tensor_from_project(p, checkpoints=2)
        table = torch.nn.Parameter(torch.tensor([[1., 1., 1., 0., 0., 0.], [2.4, 2.1, 2.7, .15, 0., .1]], dtype=torch.float32))
        return [table], lambda: adapter(table).signals.square().mean()
    raise KeyError(name)


def counts():
    gc.collect()
    gc.collect()
    objects = gc.get_objects()
    tensors = instances = 0
    for o in objects:
        t = type(o)
        if torch.Tensor in t.__mro__:
            tensors += 1
        elif t.__name__ in soak.COUNTED and t.__module__.startswith('torchfdtd'):
            instances += 1
    return dict(objects=len(objects), tensors=tensors, instances=instances)


@pytest.mark.parametrize('name', ['differentiable', 'plane', 'dispersive', 'source_waveform', 'streamed', 'reversible', 'tensor_project'])
def test_thirty_iterations_on_cpu_retain_nothing_after_warm_up(name):
    torch.set_num_threads(1)
    parameters, loss_fn = build(name)
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    history = []
    for i in range(1, 31):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for v in parameters:
                v.clamp_(min=1. if v.ndim >= 3 and name != 'tensor_project' else -10.)
        del loss
        if i % 10 == 0:
            history.append(counts())
    warm, later = history[0], history[1:]
    assert all(h['tensors'] == warm['tensors'] for h in later), (name, [h['tensors'] for h in history])
    assert all(h['instances'] == warm['instances'] for h in later), (name, [h['instances'] for h in history])
    assert max(h['objects'] for h in later)-warm['objects'] <= 100, (name, [h['objects'] for h in history])
    del parameters, loss_fn, optimizer
    assert counts()['instances'] <= warm['instances']


def test_snapshot_reports_the_bounded_module_caches():
    process = psutil.Process(os.getpid())
    sample = soak.snapshot(process, cuda=False)
    caches = sample['caches']
    from torchfdtd import cuda_kernels, injection, streamed_cost, streamed_work
    assert caches['cuda_kernels_compile'] <= cuda_kernels._compile.cache_info().maxsize == 64
    assert caches['injection_incident_line'] <= injection._incident_line.cache_info().maxsize == 8
    assert caches['streamed_cost_replay_blocks'] <= streamed_cost.replay_blocks.cache_info().maxsize == 1024
    assert caches['streamed_work_replay_summary'] <= streamed_work._replay_summary.cache_info().maxsize == 4096
    assert caches['capabilities_rules'] > 0 and caches['capabilities_lanes'] > 0
    assert sample['live_tensors'] >= 0 and sample['gc_objects'] > 0 and 'torch_allocated' not in sample
