"""Checkpoint completeness (G5-08).

The solver journal must carry every state the streamed run needs: E, H, every
CPML psi segment, the stored PMC face banks and the ADE P and Q banks. The
optimization checkpoint must carry the design parameters, the Adam moments
and step count, the projection and continuation state, the generator states,
the effective source waveform and the configuration fingerprint. Solver
restart and optimizer restart are tested separately; a checkpoint lacking any
listed state is refused by name.
"""
from dataclasses import replace
import json
import os
import shutil

import pytest
import torch

from torchfdtd import StreamedSimulation, StreamedDispersiveSimulation, Monitor, Source
from torchfdtd.design_checkpoint import DesignCheckpoint, REQUIRED_STATES
from torchfdtd.design_parameterization import DensityParameterization
from torchfdtd.identity import restart_key
from torchfdtd.solver import voxelize
from torchfdtd.streamed import _StreamedExecution
from torchfdtd.streamed_restart import RestartJournal
from test_pmc_general import scene as pmc_scene, at, PROFILE
from test_streamed_restart import options, scene, Interrupted, interrupt


# ---- solver restart alone ------------------------------------------------------------
def dispersive_inputs(p):
    rng = torch.Generator().manual_seed(492)
    shape = p.region.shape
    eps = (1.5+.2*torch.rand(shape, generator=rng, dtype=torch.float64)).requires_grad_()
    strength = (torch.tensor([.3, .2], dtype=torch.float64)*1e30).requires_grad_()
    omega0 = (torch.tensor([.9, 1.4], dtype=torch.float64)*1e15).requires_grad_()
    gamma = (torch.tensor([.15, .2], dtype=torch.float64)*1e14).requires_grad_()
    return eps, strength, omega0, gamma


def fixture(kind, tmp_path, **overrides):
    """(project, streamed model factory, inputs, expected auxiliary counts, execution)."""
    if kind == 'cpml':
        p = scene()
        inputs = (torch.full(p.region.shape, 1.7, dtype=torch.float64, requires_grad=True),)
        model = lambda settings: StreamedSimulation(p, settings)
        execution = _StreamedExecution()
        return p, model, inputs, dict(faces=0, poles=0), execution
    if kind == 'faces':
        # CPML on two walls and stored PMC faces on three: psi segments and face banks coexist.
        kinds = (('pml', 'pmc'), ('pmc', 'pml'), ('pec', 'pmc'))
        p = pmc_scene(kinds, size=(1.0, 1.0, .6), steps=13, precision='float64',
                      profiles={'x_min': PROFILE, 'y_max': dict(layers=3, kappa=2., alpha=.01)})
        p.sources = [Source(center=at(p.region, 'Ez', (6, 4, 3)), component='Ez', pulse='continuous', wavelength=.5)]
        p.monitors = [Monitor(center=at(p.region, 'Ez', (7, 3, 2)), component='Ez'),
                      Monitor(center=at(p.region, 'Hx', (p.region.shape[0], 2, 3)), component='Hx')]
        p = type(p).model_validate(p.model_dump())
        epsilon, _ = voxelize(p)
        inputs = (torch.as_tensor(epsilon).contiguous().requires_grad_(),)
        model = lambda settings: StreamedSimulation(p, settings)
        return p, model, inputs, dict(faces=None, poles=0), _StreamedExecution()
    p = scene()
    inputs = dispersive_inputs(p)
    model = lambda settings: StreamedDispersiveSimulation(p, settings)
    return p, model, inputs, dict(faces=0, poles=2), _StreamedExecution()


@pytest.mark.parametrize('kind', ['cpml', 'faces', 'ade'])
def test_solver_journal_carries_every_auxiliary_state_and_resumes(tmp_path, monkeypatch, kind):
    p, model, inputs, counts, execution = fixture(kind, tmp_path)
    settings = options(tmp_path, 'host', restart_directory=None, slab_width=3 if kind == 'faces' else 4)
    reference = model(settings)(*inputs)
    signals = reference.signals.detach().clone()
    gradients = torch.autograd.grad(reference.signals.abs().square().sum(), inputs)
    journaled = replace(settings, restart_directory=str(tmp_path / 'journal'))
    # The host state tuple of this fixture: E, H, one psi per CPML segment, the
    # stored E and H face banks and, for ADE, P, Q and their face banks.
    sim = model(journaled)
    interrupt(monkeypatch, 'forward', 2)
    with pytest.raises(Interrupted):
        sim(*inputs)
    monkeypatch.undo()
    meta = json.loads((tmp_path / 'journal' / 'forward-2' / 'meta.json').read_text())
    arrays = meta['arrays']
    plain = _StreamedExecution()
    host = plain.host(p, inputs[0].detach(), None)
    state = host.state()
    segments = len(host.segments)
    faces_e, faces_h = len(host.grid.faces['E']), len(host.grid.faces['H'])
    assert segments > 0 and (faces_e > 0 and faces_h > 0) == (kind == 'faces')
    assert len(state) == 2+segments+faces_e+faces_h
    names = ['E', 'H']+[f'psi[{i}]' for i in range(segments)]+[f'face_E[{i}]' for i in range(faces_e)]+[f'face_H[{i}]' for i in range(faces_h)]
    if kind == 'ade':
        # P and Q banks of the two poles follow the dielectric prefix, in the x-first slab layout.
        names += ['P', 'Q']
        for description, name in zip(arrays[len(state):len(state)+2], ('P', 'Q')):
            assert tuple(description['shape']) == (p.region.shape[0], 2, *p.region.shape[1:], 3), name
        state = None
    assert arrays[-1]['file'] == 'signals.bin' and tuple(arrays[-1]['shape']) == tuple(signals.shape)
    assert len(arrays) == len(names)+1, names
    for name, description in zip(names, arrays[:-1]):
        assert description['dtype'] == 'float64' and description['bytes'] > 0 and len(description['sha256']) == 64, name
    if state is not None:
        for name, description, value in zip(names, arrays[:-1], state):
            assert tuple(description['shape']) == tuple(value.shape) and description['bytes'] == value.numel()*value.element_size(), name
        # The record restores exactly the state the operator reaches after two blocks.
        operator = plain.operator(host, journaled, None)
        current = host.state()
        for index in range(2):
            current, _ = operator.forward(inputs[0].detach(), current, 3*index, 3)
        journal = RestartJournal(tmp_path / 'journal', json.loads((tmp_path / 'journal' / 'contract.json').read_text()))
        block, loaded, loaded_signals = journal.load_forward(operator.new_state, signals)
        journal.close()
        assert block == 2
        for name, a, b in zip(names, loaded, current):
            assert torch.equal(a, b), name
        assert torch.equal(loaded_signals[:6], signals[:6])
    # Forward resume, then backward resume, reproduce the uninterrupted run bitwise on CPU.
    resumed = model(journaled)(*inputs)
    assert resumed.report['forward_resumed_from_block'] == 2
    torch.testing.assert_close(resumed.signals.detach(), signals, rtol=0, atol=0)
    interrupt(monkeypatch, 'transpose', 2)
    with pytest.raises(Interrupted):
        torch.autograd.grad(resumed.signals.abs().square().sum(), inputs)
    monkeypatch.undo()
    del resumed
    again = model(journaled)(*inputs)
    assert again.report['forward_resumed_from_block'] == 'complete'
    later = torch.autograd.grad(again.signals.abs().square().sum(), inputs)
    assert again.report['backward_resumed_from_block'] == 3
    for a, b in zip(later, gradients):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert (tmp_path / 'journal' / 'complete.json').exists()


# ---- optimizer restart alone ---------------------------------------------------------
def parameterization():
    return DensityParameterization((16, 15), spacing_um=.1, mode='density', initial=.5, filter_radius_um=.15,
                                   boundary='truncate', beta=1., eta=.5, dtype=torch.float64)


def optimize(p, settings, root, iterations, *, seed=7):
    """Projected Adam over a filtered density with beta continuation, a random objective weight and per-iteration journals."""
    param = parameterization()
    optimizer = torch.optim.Adam(param.parameters(), lr=.05, foreach=False)
    model = StreamedSimulation(p, settings)
    checkpoint = DesignCheckpoint(root / 'checkpoint', model.plan, options=settings)
    restored = checkpoint.load(param, optimizer)
    if restored is None:
        start, history = 0, []
        torch.manual_seed(seed)
    else:
        start, history = restored['iteration'], restored['history']
    for k in range(start, iterations):
        if k == 2:
            param.advance_beta(2.)
        weight = torch.rand(3, dtype=torch.float64)
        density = param()
        eps = (1+density)[:, :, None].contiguous()
        journal = root / 'journal' / f'iteration-{k}'
        result = StreamedSimulation(p, replace(settings, restart_directory=str(journal)))(eps)
        objective = (result.signals.square().sum(0)*weight).sum()
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        history.append(dict(iteration=k, objective=float(objective.detach()), gradient_l2=float(param.design.grad.norm()),
                            beta=float(param.beta), weight=weight.tolist(), resumed=result.report.get('backward_resumed_from_block')))
        optimizer.step()
        param.project_parameters_()
        checkpoint.save(iteration=k+1, parameterization=param, optimizer=optimizer, history=history)
        shutil.rmtree(journal)
    return history, param.design.detach().clone(), optimizer.state_dict()


def test_optimizer_restart_reproduces_the_uninterrupted_history_bitwise(tmp_path, monkeypatch):
    p = scene()
    settings = options(tmp_path, 'host', restart_directory=None)
    expected, design, optimizer_state = optimize(p, settings, tmp_path / 'reference', 4)
    assert len(expected) == 4 and expected[2]['beta'] == 2. and all(h['resumed'] is None for h in expected)
    # Interrupt the backward of the third iteration after two transposes: the
    # checkpoint holds two updates, the iteration's journal holds the partial backward.
    interrupt(monkeypatch, 'transpose', 2*5+2)
    with pytest.raises(Interrupted):
        optimize(p, settings, tmp_path / 'interrupted', 4)
    monkeypatch.undo()
    assert json.loads((tmp_path / 'interrupted' / 'checkpoint' / 'checkpoint.json').read_text())['iteration'] == 2
    assert (tmp_path / 'interrupted' / 'journal' / 'iteration-2' / 'latest-backward.json').exists()
    actual, resumed_design, resumed_optimizer = optimize(p, settings, tmp_path / 'interrupted', 4)
    assert [h['resumed'] for h in actual] == [None, None, 3, None]
    for a, b in zip(actual, expected):
        assert a['objective'] == b['objective'] and a['gradient_l2'] == b['gradient_l2']
        assert a['beta'] == b['beta'] and a['weight'] == b['weight'], a
    assert torch.equal(resumed_design, design)
    for key in ('exp_avg', 'exp_avg_sq', 'step'):
        assert torch.equal(resumed_optimizer['state'][0][key], optimizer_state['state'][0][key]), key


def test_checkpoint_round_trip_restores_each_named_state(tmp_path):
    p = scene()
    settings = options(tmp_path, 'host', restart_directory=None)
    plan = StreamedSimulation(p, settings).plan
    param = parameterization()
    optimizer = torch.optim.Adam(param.parameters(), lr=.05, foreach=False)
    torch.manual_seed(11)
    (param().sum()*torch.rand((), dtype=torch.float64)).backward()
    optimizer.step()
    param.advance_beta(4.)
    checkpoint = DesignCheckpoint(tmp_path / 'checkpoint', plan, options=settings)
    saved = checkpoint.save(iteration=1, parameterization=param, optimizer=optimizer, history=[dict(objective=1.)])
    assert saved['iteration'] == 1 and len(saved['sha256']) == 64
    draws = (torch.rand(2), torch.randn(2))
    parameters = {k: v.clone() for k, v in param.state_dict().items() if isinstance(v, torch.Tensor)}
    moments = {k: v.clone() for k, v in optimizer.state_dict()['state'][0].items()}
    # Disturb every state, then restore.
    (param().sum()).backward()
    optimizer.step()
    param.advance_beta(2.)
    torch.rand(5)
    other, other_optimizer = parameterization(), None
    other_optimizer = torch.optim.Adam(other.parameters(), lr=.05, foreach=False)
    restored = DesignCheckpoint(tmp_path / 'checkpoint', plan, options=settings).load(other, other_optimizer)
    assert restored['iteration'] == 1 and restored['history'] == [dict(objective=1.)]
    for key, value in parameters.items():
        assert torch.equal(other.state_dict()[key], value), key
    assert float(other.beta) == 4. and int(other.continuation_updates) == 1
    assert other.get_extra_state()['filter_radius_um'] == .15 and other.get_extra_state()['boundary'] == 'truncate'
    for key, value in moments.items():
        assert torch.equal(other_optimizer.state_dict()['state'][0][key], value), key
    assert int(other_optimizer.state_dict()['state'][0]['step']) == 1
    assert torch.equal(torch.rand(2), draws[0]) and torch.equal(torch.randn(2), draws[1])
    assert checkpoint.fingerprint == restart_key(plan, options=settings)
    assert checkpoint.waveform and torch.equal(checkpoint.waveform[0]['samples'], torch.from_numpy(plan.sources[0].terms[0].samples.copy()))


def rewrite(root, mutate):
    from torchfdtd.streamed_restart import sha256_file
    path, meta_path = root / 'checkpoint.pt', root / 'checkpoint.json'
    payload = torch.load(path, weights_only=False)
    mutate(payload)
    torch.save(payload, path)
    meta = json.loads(meta_path.read_text())
    meta.update(sha256=sha256_file(path), bytes=path.stat().st_size)
    meta_path.write_text(json.dumps(meta))


@pytest.mark.parametrize('name', REQUIRED_STATES)
def test_checkpoint_missing_a_state_is_refused_by_name(tmp_path, name):
    p = scene()
    settings = options(tmp_path, 'host', restart_directory=None)
    plan = StreamedSimulation(p, settings).plan
    param = parameterization()
    optimizer = torch.optim.Adam(param.parameters(), lr=.05, foreach=False)
    checkpoint = DesignCheckpoint(tmp_path / 'checkpoint', plan, options=settings)
    checkpoint.save(iteration=3, parameterization=param, optimizer=optimizer, history=[])
    rewrite(tmp_path / 'checkpoint', lambda payload: payload.pop(name))
    with pytest.raises(ValueError, match=f'missing the state {name}\\b'):
        checkpoint.load(parameterization(), torch.optim.Adam(parameterization().parameters(), foreach=False))


def test_checkpoint_refuses_another_configuration_waveform_or_damage(tmp_path):
    p = scene()
    settings = options(tmp_path, 'host', restart_directory=None)
    plan = StreamedSimulation(p, settings).plan
    param = parameterization()
    optimizer = torch.optim.Adam(param.parameters(), lr=.05, foreach=False)
    checkpoint = DesignCheckpoint(tmp_path / 'checkpoint', plan, options=settings)
    checkpoint.save(iteration=1, parameterization=param, optimizer=optimizer, history=[])
    other = scene()
    other.region.steps = 14
    with pytest.raises(ValueError, match='fingerprint'):
        DesignCheckpoint(tmp_path / 'checkpoint', StreamedSimulation(other, settings).plan, options=settings).load(parameterization(), optimizer)
    with pytest.raises(ValueError, match='fingerprint'):
        DesignCheckpoint(tmp_path / 'checkpoint', plan, options=replace(settings, slab_width=8)).load(parameterization(), optimizer)
    rewrite(tmp_path / 'checkpoint', lambda payload: payload['waveform'][0]['samples'].mul_(2))
    with pytest.raises(ValueError, match='effective source waveform'):
        checkpoint.load(parameterization(), optimizer)
    checkpoint.save(iteration=1, parameterization=param, optimizer=optimizer, history=[])
    data = bytearray((tmp_path / 'checkpoint' / 'checkpoint.pt').read_bytes())
    data[len(data)//2] ^= 0x5A
    (tmp_path / 'checkpoint' / 'checkpoint.pt').write_bytes(bytes(data))
    with pytest.raises(ValueError, match='checksum mismatch'):
        checkpoint.load(parameterization(), optimizer)
    (tmp_path / 'checkpoint' / 'checkpoint.pt').write_bytes(bytes(data[:100]))
    with pytest.raises(ValueError, match='holds 100 of'):
        checkpoint.load(parameterization(), optimizer)
    # A parameterization built with another filter refuses the saved parameters.
    checkpoint.save(iteration=1, parameterization=param, optimizer=optimizer, history=[])
    wrong = DensityParameterization((16, 15), spacing_um=.1, mode='density', initial=.5, filter_radius_um=.25,
                                    boundary='truncate', beta=1., eta=.5, dtype=torch.float64)
    with pytest.raises(ValueError, match='configuration differs'):
        checkpoint.load(wrong, optimizer)


def test_checkpoint_with_a_pickled_object_is_refused_without_executing_it(tmp_path):
    p = scene()
    settings = options(tmp_path, 'host', restart_directory=None)
    plan = StreamedSimulation(p, settings).plan
    param = parameterization()
    optimizer = torch.optim.Adam(param.parameters(), lr=.05, foreach=False)
    checkpoint = DesignCheckpoint(tmp_path / 'checkpoint', plan, options=settings)
    checkpoint.save(iteration=1, parameterization=param, optimizer=optimizer, history=[])
    executed = tmp_path / 'executed'

    class Hostile:
        def __reduce__(self):
            return os.makedirs, (str(executed),)

    # A complete, correctly checksummed checkpoint whose extra state is a pickled object.
    rewrite(tmp_path / 'checkpoint', lambda payload: payload.update(extra=Hostile()))
    with pytest.raises(ValueError, match='pickled object'):
        checkpoint.load(parameterization(), optimizer)
    assert not executed.exists()
    torch.load(tmp_path / 'checkpoint' / 'checkpoint.pt', weights_only=False)   # control: an unrestricted load runs the payload
    assert executed.is_dir()
