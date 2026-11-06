"""Selected-frequency differentiable periodic-layer detector responses."""
import hashlib,json,math
import torch
from .models import Project,Region,Source,FieldMonitor,BoundaryFace,Boundaries
from .differentiable import AdjointOptions
from .adjoint_planes import DifferentiablePlaneSimulation
from .density_layer import periodic_density_layer
from .polarization import calibrate_plane_polarization,mix_plane_fields
from .detector_allocation import quadrant_intensity_allocation
from .solver import C0


def _periodic_project(spec,*,dtype,mesh,steps,pml_cells,forward_kernel,memory_mode='resident'):
    for key in ('wavelength_um','background_index','design_index','height_um','detector_offset_um','theta_inside_rad','phi_rad'):
        value=spec[key]
        if isinstance(value,torch.Tensor) or not math.isfinite(value):
            raise ValueError('Specification values must be fixed finite scalars.')
    if mesh<=0 or not math.isfinite(mesh) or any(spec[k]<=0 for k in ('wavelength_um','height_um','detector_offset_um')):
        raise ValueError('Mesh, wavelength, height and detector offset must be positive.')
    if not 0<=spec['theta_inside_rad']<math.pi/2:
        raise ValueError('Incidence must be forward propagating.')
    wavelength=spec['wavelength_um'];n=spec['background_index'];height=spec['height_um'];period=spec['period_um']
    theta=spec['theta_inside_rad'];phi=spec['phi_rad'];detector=height/2+spec['detector_offset_um']
    source_z=-height/2-2*wavelength/n;probe_z=-height/2-wavelength/n
    half=math.ceil((max(detector,abs(source_z))+max(1.,20*mesh))/mesh)*mesh
    kt=[2*math.pi*n/wavelength*math.sin(theta)*v for v in (math.cos(phi),math.sin(phi))]
    project=Project(region=Region(dimension='3d',size=(*period,2*half),mesh=mesh,steps=steps,pml_cells=pml_cells,
        memory_mode=memory_mode,precision='float64' if dtype==torch.float64 else 'float32',background_index=n,material_sampling='yee',cuda_kernel=forward_kernel,
        bloch_phase=(kt[0]*period[0],kt[1]*period[1],0),
        boundaries=Boundaries(x_min=BoundaryFace(kind='bloch'),x_max=BoundaryFace(kind='bloch'),
                             y_min=BoundaryFace(kind='bloch'),y_max=BoundaryFace(kind='bloch'))),
        sources=[Source(id='periodic-plane-source',kind='plane',normal='z',size=(*period,0),center=(0,0,source_z),component='Ex',wavelength=wavelength,pulse_cycles=1)],
        monitors=[FieldMonitor(id=name,normal='z',size=(*period,0),center=(0,0,z)) for name,z in [('incident',probe_z),('detector',detector)]])
    # Under the workbench's memory admission the design's planning is counted before it is built.
    from .solver import admit_planning
    admit_planning(project)
    return project,kt


def periodic_layer_response(density,spec,*,mesh,steps,pml_cells=12,
                            quadrature_counts=(24,24),pixel_origin='cell_edges',options=None,reference_cache=None,forward_kernel='torch'):
    """Compute Cartesian x/y responses with shape (2,4), well order R,G2,G1,B.

    spec supplies wavelength_um, background_index, design_index, period_um,
    height_um, detector_offset_um, theta_inside_rad and phi_rad. All are fixed.
    This selected-frequency dielectric model differentiates only density.
    Two homogeneous references are evaluated without gradients. An optional
    PlaneReferenceCache retains only their compact spectral planes on CPU.
    Use spectral_pupil_response to bound residency across wavelengths/rays.
    Arithmetic material averaging and time/mesh convergence remain user checks.
    """
    if not isinstance(density,torch.Tensor) or density.dtype not in (torch.float32,torch.float64):
        raise ValueError('Density must be real FP32 or FP64.')
    project,kt=_periodic_project(spec,dtype=density.dtype,mesh=mesh,steps=steps,
        pml_cells=pml_cells,forward_kernel=forward_kernel)
    n,height,wavelength,theta,phi=(spec[k] for k in ('background_index','height_um','wavelength_um','theta_inside_rad','phi_rad'))
    def epsilon(d):return periodic_density_layer(d,project.region,bottom_um=-height/2,top_um=height/2,
        background_epsilon=n*n,design_epsilon=spec['design_index']**2,pixel_origin=pixel_origin)
    frequency=[C0/(wavelength*1e-6)];models=[];refs=[]
    with torch.no_grad():
        background=torch.full_like(epsilon(density),n*n)
        for component in ('Ex','Ey'):
            p=project.model_copy(deep=True);p.sources[0].component=component
            model=DifferentiablePlaneSimulation(p,options or AdjointOptions(checkpoints=4),quadrature_counts={'incident':quadrature_counts,'detector':quadrature_counts})
            models.append(model)
            if reference_cache is None:reference=model(background,frequency)
            else:
                from .reference_cache import PlaneReferenceCache
                if not isinstance(reference_cache,PlaneReferenceCache):raise ValueError('Expected PlaneReferenceCache.')
                payload=dict(project=p.model_dump(mode='json'),frequency=frequency,quadrature_counts=quadrature_counts,
                             dtype=str(density.dtype),device=str(density.device))
                key=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
                reference=reference_cache._get(key,density.device,lambda:model(background,frequency))
            refs.append(reference)
        pvec=density.new_tensor([math.cos(theta)*math.cos(phi),math.cos(theta)*math.sin(phi)])
        svec=density.new_tensor([-math.sin(phi),math.cos(phi)])
        targets=[math.cos(phi)*pvec-math.sin(phi)*svec,math.sin(phi)*pvec+math.cos(phi)*svec]
        coefficients=[calibrate_plane_polarization([r['incident'] for r in refs],[kt],v[None]) for v in targets]
    planes=[model(epsilon(density),frequency)['detector'] for model in models]
    responses=[]
    for c in coefficients:
        sample=mix_plane_fields(planes,c)
        reference=mix_plane_fields([r['detector'] for r in refs],c)
        responses.append(quadrant_intensity_allocation(sample,sample.normalized_flux(reference))[0])
    return torch.stack(responses)
