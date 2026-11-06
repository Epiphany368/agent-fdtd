# ---------------------------------------------------------------------------
# COPIED, not invented. Source: the reference project 20261010_torchfdtd-lumerical-parity
#   src/run_b9_native_shape_n8.py
#   at commit 9b0f61a41b026342420422b7d208a046d7b6e45b
#   sha256(source) = 6351b1fddfa97fc938d24b2e900dffc04a4e27c0ebd3cec297213e3a13e9b503
#   copied 2026-10-18 by leg 1 (F0+F8). This project owns this copy now.
# ---------------------------------------------------------------------------
"""B9 v1.23 native shape derivative on the N8 guided fixture.

Same fixture, source, eigenmode launch, monitors, objective and checks as
run_b6_gradient.py (B6/B7). Only the permittivity construction changes: the
stack (substrate, BOX/TOX cladding background, core guides, teeth, air) is a
list of ordered DifferentiableSolid boxes rasterized by the native
smooth_geometry_epsilon with a fixed transition width, instead of the
hand-built complete-interface coefficient completion. The common x-width of
the eight narrow teeth is the differentiable parameter, with the left tooth
edge fixed, exactly as in B7.

The launch eigenmode is solved on the same smoothed cross-section the FDTD
grid holds at the source plane (a numpy replica of the native smoothstep,
verified by the solver's own frozen-source-material check).

mode 'geometry' runs no simulation: it records d(sum epsilon)/d(width) per
Yee component against the tooth shift and transition width, normalized by the
shift-invariant ideal value."""
from parity import *
import hashlib
import torch
from torchfdtd import AdjointOptions, DifferentiableSolid, smooth_geometry_epsilon
from torchfdtd.solver import field_axes
from torchfdtd.mode_injection import ModeInjectedPlaneSimulation, modal_plane_amplitudes
from torchfdtd.mode_branches import prepare_aperture_modal_launch

FAR = 1e3  # um; half-infinite boxes end far outside every domain


def step(distance, width):
    fraction = np.clip(distance/width+.5, 0, 1)
    return fraction**2*(3-2*fraction)


def scene(cfg, c):
    """Region/project exactly as run_b6_gradient.py, plus the native solid list."""
    dx=c['dx_um'];dy=c['dy_um'];N=c.get('periods',8)
    nz=c.get('z_cells',2);zspan=nz*.04
    shift=c.get('shift_nm',0)*.001;end=N*.9+.26325+shift
    xmax=-3+math.ceil((end+6)/.04-1e-9)*.04;span=xmax+3;xc=(xmax-3)/2
    boundaries=Boundaries(x_min=BoundaryFace(kind='pml',layers=round(.64/dx)),
        x_max=BoundaryFace(kind='pml',layers=round(.64/dx)),
        y_min=BoundaryFace(kind='pml',layers=round(.4/dy)),
        y_max=BoundaryFace(kind='pml',layers=round(.4/dy)),
        z_min=BoundaryFace(kind='periodic'),z_max=BoundaryFace(kind='periodic'))
    r=Region(dimension='3d',size=(span,6,zspan),mesh=dx,mesh_steps=(dx,dy,.04),
        pml_cells=min(round(.64/dx),50),precision='float64',backend='cuda',
        cuda_kernel='fused',cuda_monitor_kernel='fused',material_sampling='yee',
        interface_method='staircase',snapshot_interval=10000,boundaries=boundaries)
    r.steps=math.ceil(c.get('duration_fs',500)*1e-15/r.time_step)
    spec=SpectrumSettings(sampling='frequency',apodization='none',custom_frequencies_hz=[C0/(cfg['wavelength_um']*1e-6)])
    snap=lambda x: -span/2+round((x-xc+span/2)/dx)*dx
    locs=dict(outside=-1.6,near=-.8+shift,far=end+.8,
        **{'box-left':-.8+shift,'box-right':end+.8})
    monitors=[FieldMonitor(id=k,center=(snap(x),.12 if k in ('near','far','outside') else 0,0),
        size=(0,3.6 if k in ('near','far','outside') else 3.,zspan),normal='x',
        spectrum=spec.model_copy(deep=True)) for k,x in locs.items()]
    for k,y in [('top',1.5),('bottom',-1.5)]:
        monitors.append(FieldMonitor(id=k,center=((locs['near']+locs['far'])/2-xc,y,0),
            size=(locs['far']-locs['near'],0,zspan),normal='y',spectrum=spec.model_copy(deep=True)))
    p=Project(name=c['id'],region=r,sources=[Source(id='source',kind='plane',normal='x',
        direction='+',component='Ey',center=(snap(-1.44),.12,0),size=(0,3.6,zspan),
        wavelength=cfg['wavelength_um'],pulse_cycles=4)],monitors=monitors)
    ec=cfg['core_index']**2;eb=cfg['cladding_index']**2
    transition=c['transition_nm']*.001
    # Ordered stack over the BOX/TOX cladding background; later solids overwrite.
    layers=[(-FAR,-2.,ec),(2.2,FAR,1.)]   # (y_lo, y_hi, epsilon): substrate, air
    def solids(width):
        def box(x0,x1,y0,y1,e):
            # World x -> origin-centred region x. Edges may be tensors.
            return DifferentiableSolid.box(((x1-x0),y1-y0,2*FAR),epsilon=e,
                                           center=((x0+x1)/2-xc,(y0+y1)/2,0.))
        out=[box(-FAR,FAR,y0,y1,e) for y0,y1,e in layers]
        out+=[box(-FAR,shift,0.,.22,ec),box(end,FAR,0.,.22,ec)]   # input and output guides
        for k in range(N):
            a=shift+k*.9+.26325;b=shift+k*.9+.6525
            out.append(box(width.new_tensor(a),a+width,0.,.22,ec))   # narrow tooth, left edge fixed
            out.append(box(b,b+.2475,0.,.22,ec))                       # wide tooth, fixed
        return out
    def epsilon(width):
        return smooth_geometry_epsilon(r,solids(width),background=eb,width=transition,chunk_cells=1<<18)
    def section(y,z):
        # Source-plane cross-section of the same ordered stack (input guide occupies x).
        v=np.full(np.broadcast(y,z).shape,eb)
        for y0,y1,e in layers+[(0.,.22,ec)]:
            occ=step(np.minimum(y-y0,y1-y),transition)
            v=v*(1-occ)+e*occ
        return v
    return dict(r=r,p=p,xc=xc,shift=shift,end=end,N=N,ec=ec,eb=eb,transition=transition,
                epsilon=epsilon,section=section)


def geometry_only(cfg, report, save):
    """d(sum of epsilon samples)/d(width) per component, no simulation."""
    rows=[]
    for g in cfg['geometry_cases']:
        for t in g['transition_nm']:
            for s in g['shift_nm']:
                c=dict(g,id=f"geom-x{round(g['dx_um']*1000)}-w{t:g}-s{s:g}",transition_nm=t,shift_nm=s)
                sc=scene(cfg,c);r=sc['r']
                w=torch.tensor(cfg['width_um'],dtype=torch.float64,device='cuda',requires_grad=True)
                eps=sc['epsilon'](w);sums=eps.sum(dim=(0,1,2))
                fill=[float(torch.autograd.grad(sums[i],w,retain_graph=True)[0]) for i in range(3)]
                # Shift-invariant ideal: each of N moving edges sweeps 1/dx samples per um along x,
                # each carrying (ec-eb) times the smoothed core-layer y occupancy.
                ideal=[]
                for comp in ('Ex','Ey','Ez'):
                    ax=field_axes(r,comp);y=ax[1]
                    occ=step(np.minimum(y-0.,.22-y),sc['transition'])
                    ideal.append(sc['N']*(sc['ec']-sc['eb'])*occ.sum()*len(ax[2])/c['dx_um'])
                row=dict(id=c['id'],dx_um=c['dx_um'],dy_um=c['dy_um'],transition_nm=t,shift_nm=s,
                    shape=list(r.shape),fill_derivative_per_um=dict(zip(('Ex','Ey','Ez'),fill)),
                    ideal_fill_derivative_per_um=dict(zip(('Ex','Ey','Ez'),ideal)),
                    normalized_edge_weight=dict(zip(('Ex','Ey','Ez'),[f/i for f,i in zip(fill,ideal)])))
                rows.append(row);report['geometry_only']=rows;save();print(json.dumps(row),flush=True)
                del eps,sums,w;torch.cuda.empty_cache()
    report.update(status='completed',simulation_executed=False)
    save()


def main():
    cfg,run=load_run();effective=json.loads(Path(os.environ['EFFECTIVE_JSON']).read_text())
    report=dict(policy=cfg['policy'],status='in_progress',cases=[],hardware=torch.cuda.get_device_name(),
        precision='float64',physical_mapping_certified=False,acceptance_granted=False,
        objective='far forward guided power / near forward guided power',
        scope='Native differentiable-shape width derivative on the N8 fixture (plan v1.23); diagnostic only, no device, fiber or optimizer claim')
    def save():
        write_json(run/'readout.json',report);write_json(os.environ['EFFECTIVE_JSON'],effective)
    if cfg.get('mode')=='geometry':
        return geometry_only(cfg,report,save)
    frequency=[C0/(cfg['wavelength_um']*1e-6)]
    for c in cfg['cases']:
        started=time.perf_counter();sc=scene(cfg,c);r=sc['r'];p=sc['p']
        modes=[prepare_aperture_modal_launch(p,sc['section'],mode_index=i,num_modes=2) for i in range(2)]
        scores=[float(np.sum(abs(m.mode.fields[...,1])**2)/np.sum(abs(m.mode.fields[...,[1,2]])**2)) for m in modes]
        launch=modes[int(np.argmax(scores))]
        assert max(scores)>1-1e-8 and launch.mode.maxwell_residual<1e-5
        write_json(run/(c['id']+'-project.json'),p.model_dump(mode='json'))
        effective['resolved_cases'].append(dict(id=c['id'],shape=r.shape,dt_s=r.time_step,steps=r.steps,
            launch_identity=launch.identity,parameter='all narrow-tooth x-widths in um (left edge fixed)',
            permittivity='native smooth_geometry_epsilon, ordered DifferentiableSolid boxes',
            transition_um=sc['transition'],shift_um=sc['shift']))
        case=dict(id=c['id'],dx_um=c['dx_um'],dy_um=c['dy_um'],shift_nm=c.get('shift_nm',0),
            transition_nm=c['transition_nm'],shape=list(r.shape),steps=r.steps,dt_s=r.time_step,
            launch_effective_index=launch.mode.beta_per_um*cfg['wavelength_um']/(2*math.pi),
            evaluations=[],finite_differences=[])
        report['cases'].append(case);save()
        model=ModeInjectedPlaneSimulation(p,launch,AdjointOptions(checkpoints=32,gpu_budget_bytes=32*1024**3))
        report.update(checkpoint_slots=32,gpu_budget_bytes=32*1024**3)
        def evaluate(width,adjoint=False,label=''):
            t=time.perf_counter();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            parameter=torch.tensor(width,dtype=torch.float64,device='cuda',requires_grad=adjoint)
            if adjoint:base=sc['epsilon'](parameter)
            else:
                with torch.no_grad():base=sc['epsilon'](parameter)
            t_geometry=time.perf_counter()-t
            if adjoint:planes=model(base,frequency)
            else:
                with torch.no_grad():planes=model(base,frequency)
            amps={k:modal_plane_amplitudes(planes[k],launch) for k in ('near','far','outside')}
            objective=(amps['far']['forward']/amps['near']['forward']).abs().square().sum()
            derivative=None;source_gradient=None
            if adjoint:
                torch.cuda.synchronize();t_forward=time.perf_counter()-t
                gp,ge=torch.autograd.grad(objective,(parameter,base))
                derivative=float(gp)
                (yb,ye),(zb,ze)=launch.aperture
                source_gradient=float(ge[launch.electric_index-1:launch.electric_index+2,yb:ye,zb:ze].abs().max())
                del ge
            torch.cuda.synchronize()
            arrays={}
            for name,plane in planes.items():
                for key in ('fields','weights','points_um','frequency_hz'):arrays[name+'_'+key]=getattr(plane,key).detach().cpu().numpy()
            for name,value in amps.items():
                for key,a in value.items():arrays[name+'_'+key]=a.detach().cpu().numpy()
            np.savez_compressed(run/(c['id']+'-'+label+'-raw.npz'),**arrays)
            row=dict(id=label,width_um=width,adjoint=adjoint,T=float(objective.detach()),
                R=float((amps['near']['backward']/amps['near']['forward']).abs().square().sum()),
                derivative_T_per_um=derivative,frozen_source_neighborhood_gradient=source_gradient,
                finite=all(bool(np.isfinite(a).all()) for a in arrays.values()),
                geometry_seconds=t_geometry,forward_seconds=(t_forward if adjoint else None),
                elapsed_seconds=time.perf_counter()-t,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                solver_report=dict(planes['near'].report),
                epsilon_sha256=hashlib.sha256(base.detach().cpu().numpy().tobytes()).hexdigest())
            del planes,base;torch.cuda.empty_cache()
            case['evaluations'].append(row);save()
            print(json.dumps({k:v for k,v in row.items() if k!='solver_report'}),flush=True)
            return row
        w0=cfg['width_um']
        base=evaluate(w0,True,'baseline-adjoint');ad=base['derivative_T_per_um']
        for h in c.get('finite_difference_steps_um',cfg['finite_difference_steps_um']):
            plus=evaluate(w0+h,False,f'h{h:g}-plus');minus=evaluate(w0-h,False,f'h{h:g}-minus')
            fd=(plus['T']-minus['T'])/(2*h)
            row=dict(step_um=h,central_T_derivative_per_um=fd,adjoint_T_derivative_per_um=ad,
                absolute_error=abs(fd-ad),relative_error=abs(fd-ad)/max(abs(fd),abs(ad),1e-12))
            case['finite_differences'].append(row);save();print(json.dumps(row),flush=True)
        finest=case['finite_differences'][-1]
        case['checks']=dict(finite=all(e['finite'] for e in case['evaluations']) and math.isfinite(ad),
            source_frozen=base['frozen_source_neighborhood_gradient']==0,
            finest_derivative=finest['relative_error']<.01 or finest['absolute_error']<1e-5)
        case.update(T=base['T'],R=base['R'],adjoint_T_derivative_per_um=ad,
            all_internal_checks_passed=all(case['checks'].values()),
            peak_allocated_bytes=max(e['gpu_peak_allocated_bytes'] for e in case['evaluations']),
            elapsed_seconds=time.perf_counter()-started)
        save()
    report.update(status='completed',simulation_executed=True,full_device_gradient_certified=False)
    save()


if __name__=='__main__':main()
