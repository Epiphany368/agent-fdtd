import pytest
import torch
from benchmarks.cr_spectral_objective import check_density_direction


def test_direction_sweep_detects_incorrect_gradient_and_taylor_order():
    x=torch.tensor([[.2,.4],[.6,.8]],dtype=torch.float64,requires_grad=True)
    def objective(d):
        return (d.exp()+d**3).sum()
    g,=torch.autograd.grad(objective(x),x)
    result=check_density_direction(objective,x,g,[.002,.001,.0005])
    assert result['passed'] and all(row['passed'] for row in result['rows'])
    residuals=[row['first_order_residual'] for row in result['rows']]
    assert all(3.9<a/b<4.1 for a,b in zip(residuals,residuals[1:]))
    wrong=check_density_direction(objective,x,-g,[.002,.001,.0005])
    assert not wrong['passed']
    assert check_density_direction(objective,x,g,[.002,.001,.0005])==result


@pytest.mark.parametrize('steps',[[0],[-.1],[float('nan')],[float('inf')],[.3],[]])
def test_direction_check_rejects_invalid_perturbations_before_evaluation(steps):
    def forbidden(d):
        pytest.fail('Invalid perturbations must not run the solver')
    x=torch.tensor([.1,.9],dtype=torch.float64)
    with pytest.raises(ValueError):check_density_direction(forbidden,x,torch.ones_like(x),steps)


def test_direction_check_rejects_nonfinite_objective():
    x=torch.tensor([.5],dtype=torch.float64)
    with pytest.raises(ValueError,match='Non-finite'):
        check_density_direction(lambda d: d.sum()*float('nan'),x,torch.ones_like(x),[.001])
