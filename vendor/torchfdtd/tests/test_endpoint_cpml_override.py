"""Early CPML override rejection before endpoint topology allocation."""
import pytest
from test_endpoint_project import project
from torchfdtd.endpoint_project import endpoint_from_project


def test_periodic_project_rejects_explicit_pml_override_before_allocation(monkeypatch):
    import torchfdtd.pmc_cpml as module
    monkeypatch.setattr(module, 'EndpointCPMLSimulation',
                        lambda *args, **kwargs: pytest.fail('Unsupported override reached construction'))
    with pytest.raises(ValueError, match='not periodic or Bloch faces'):
        endpoint_from_project(project(), boundary_faces=[('pml', 'pmc'), ('pmc', 'pmc'), ('pmc', 'pmc')])
