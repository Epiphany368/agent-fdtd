"""Public policy selection and UI/Python configuration admission."""
import ast
from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient

from torchfdtd import (AdjointExecutionPolicy, AdjointOptions, PeriodicDesignConfig,
                       ReversibleCPMLOptions, periodic_design_plan)
from torchfdtd.execution_tuning import tune_adjoint_execution
from torchfdtd.server import create_app
from test_reversible_cpml_planes import fixture


def test_explicit_recorded_policy_preserves_existing_arguments_and_scope():
    legacy = AdjointExecutionPolicy(AdjointOptions(), None, 'cpu', 2**30)
    assert legacy.recorded is None and legacy.host_budget_bytes == 2**30
    for settings in ({}, {'recorded': object()},
                     {'resident': AdjointOptions(), 'recorded': ReversibleCPMLOptions()}):
        with pytest.raises(ValueError):
            AdjointExecutionPolicy(**settings)
    policy = AdjointExecutionPolicy(recorded=ReversibleCPMLOptions(), device='cpu')
    project, epsilon, _ = fixture()
    assert policy.checkpoints == 0 and policy.temporal_depth == 1
    with pytest.raises(ValueError, match='background epsilon'):
        policy.simulation(project)
    with pytest.raises(ValueError, match='nondispersive'):
        policy.simulation(project, dispersive=True, fixed_background_epsilon=1.3)
    with pytest.raises(ValueError, match='only to recorded'):
        legacy.simulation(project, fixed_background_epsilon=1.3)
    with patch('torchfdtd.execution_tuning._ExecutionWorkload', side_effect=AssertionError('trial allocation')):
        with pytest.raises(ValueError, match='[Rr]ecorded'):
            tune_adjoint_execution(project, epsilon, candidates=[policy])


def test_recorded_design_plan_and_export_do_not_run_fields(tmp_path):
    config = PeriodicDesignConfig(execution='recorded', device='cpu',
        initial_density=[[.2, .4], [.5, .3]], quadrature_counts=(4, 4),
        recorded_trace_chunk_steps=7)
    with patch('torchfdtd.periodic_adjoint.PeriodicLayerResponse._compute_response',
               side_effect=AssertionError('planning executed fields')):
        plan = periodic_design_plan(config)
    assert plan['execution'] == 'recorded' and plan['calibration_solves'] == 0
    assert plan['gpu_reservation_bytes'] == 0
    assert plan['total_host_reservation_bytes'] > plan['host_reservation_bytes'] > 0
    app = create_app(tmp_path)
    with TestClient(app) as client:
        normalized = client.post('/api/design/config', json=config.model_dump(mode='json'))
        assert normalized.status_code == 200
        source = client.post('/api/design/python', json=normalized.json()).text
        assignment = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign))
        assert ast.literal_eval(assignment.value.args[0]) == normalized.json()
        invalid = normalized.json() | {'recorded_trace_chunk_steps': 0}
        assert client.post('/api/design/plan', json=invalid).status_code == 422
    app.state.pool.shutdown()


@pytest.mark.parametrize('storage,transfers', [('cpu', 'async'), ('device', 'sync')])
def test_cuda_design_policy_chooses_supported_trace_transport(storage, transfers):
    from torchfdtd.periodic_design import _prepare
    config = PeriodicDesignConfig(execution='recorded', device='cuda',
                                  recorded_trace_storage=storage)
    with patch.object(torch.cuda, 'is_available', return_value=True), \
            patch('torchfdtd.periodic_design.PeriodicLayerResponse') as factory:
        factory.return_value.plan.return_value = {'host_reservation_bytes': 100}
        factory.return_value.selection_report = None
        _prepare(config)
    policy = factory.call_args.kwargs['policy']
    assert policy.recorded.trace_transfers == transfers
    assert policy.recorded.resident_budget_bytes == int(config.gpu_budget_gib*1024**3)
