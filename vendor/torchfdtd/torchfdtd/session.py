"""A small, explicitly scoped Lumerical-style Python editing facade.

This API uses SI metres, as Lumerical scripting does. The native Project API
uses micrometres. Unsupported commands and properties always raise errors.
"""
from __future__ import annotations

import numpy as np

from .models import Project, Region, Structure, Source, Monitor, FieldMonitor, Material, SpectrumSettings, SourceTimeSettings, TimeSignal
from .solver import Simulation
from .waveforms import pulse_parameters


def _shape_property(data,key,value):
    """Convert a native solid editing property from SI, without validation."""
    if key in ('x','y','z'):
        c=list(data['center']);c['xyz'.index(key)]=float(value)*1e6;data['center']=c
    elif key in ('x span','y span','z span'):
        s=list(data['size']);s['xyz'.index(key[0])]=float(value)*1e6;data['size']=s
    elif key in ('radius','outer radius','inner radius','radius 2','radius 3','outer radius 2','inner radius 2'):
        name={'outer radius':'radius','outer radius 2':'radius_2'}.get(key,key.replace(' ','_'))
        data[name]=float(value)*1e6
    elif key=='vertices':data['vertices']=(np.asarray(value,dtype=float)*1e6).tolist()
    elif key=='make ellipsoid':data['make_ellipsoid']=bool(value)
    elif key in ('theta start','theta stop'):data[key.replace(' ','_')]=float(value)
    elif key in ('first axis','second axis','third axis'):
        axes=list(data['rotation_axes']);axes[('first axis','second axis','third axis').index(key)]=str(value).lower();data['rotation_axes']=axes
    elif key in ('rotation 1','rotation 2','rotation 3'):
        angles=list(data['rotation_angles']);angles[int(key[-1])-1]=float(value);data['rotation_angles']=angles
    else:return False
    return True


class FDTD:
    def __init__(self, project=None):
        self.project=project or Project()
        self.selection='FDTD'
        self.result=None

    def _layout(self):
        if self.result is not None:
            raise RuntimeError('Call switchtolayout() before editing the simulation.')

    def addmaterial(self, model):
        self._layout()
        kinds = {'Dielectric': 'dielectric', 'Plasma': 'drude', 'Lorentz': 'lorentz'}
        if model not in kinds: raise ValueError('Supported models: Dielectric, Plasma, Lorentz.')
        i = 1
        while any(m.name == f'{model} {i}' for m in self.project.materials): i += 1
        material = Material(name=f'{model} {i}', model=kinds[model])
        self.project.materials.append(material)
        return material.name

    def setmaterial(self, name, property, value):
        self._layout()
        matches = [m for m in self.project.materials if m.name == name]
        if len(matches) != 1: raise ValueError('Material name must be unique and exist.')
        material = matches[0]
        fields = {'name': 'name', 'color': 'color'}
        if material.model == 'tensor':fields['Permittivity tensor']='epsilon_tensor'
        else:fields.update({'Refractive Index':'index'} if material.model == 'dielectric' else {'Permittivity':'epsilon_inf'})
        if material.model == 'drude': fields.update({'Plasma resonance':'plasma_rad_s', 'Plasma collision':'collision_rad_s'})
        if material.model == 'lorentz': fields.update({'Lorentz Permittivity':'delta_epsilon', 'Lorentz Resonance':'resonance_rad_s', 'Lorentz Linewidth':'linewidth_rad_s'})
        if property not in fields: raise ValueError(f'Unsupported {material.model} material property: {property}')
        data = material.model_dump(); data[fields[property]] = value
        updated = Material(**data)
        if updated.name != name and any(m.name == updated.name for m in self.project.materials):
            raise ValueError('Material name already exists.')
        self.project.materials[self.project.materials.index(material)] = updated
        for obj in self.project.structures:
            if obj.material == name: obj.material = updated.name

    def fitmaterial(self, name, data, *, options=None):
        """Fit and install a named material only when its tolerance is met.

        ``data`` uses OpticalData's explicit wavelength units, not facade SI
        lengths. Failed fits leave the project unchanged. Use fit_material
        directly to inspect an unconverged best-fit candidate.
        """
        from .material_fit import fit_material
        self._layout()
        matches=[(i,m) for i,m in enumerate(self.project.materials) if m.name==name]
        if len(matches)>1:raise ValueError('Material name must be unique.')
        color=matches[0][1].color if matches else '#6ca8dd'
        result=fit_material(data,name=name,color=color,options=options)
        material=result.require_tolerance()
        if matches:self.project.materials[matches[0][0]]=material
        else:self.project.materials.append(material)
        return result

    def getfdtdindex(self, name, frequencies, fmin=None, fmax=None):
        from .materials import permittivity
        matches = [m for m in self.project.materials if m.name == name]
        if len(matches) != 1: raise ValueError('Material name must be unique and exist.')
        return np.sqrt(permittivity(matches[0], frequencies))

    def _objects(self):
        return self.project.structures+self.project.sources+self.project.monitors

    def _selected(self):
        if self.selection=='FDTD':return self.project.region
        matches=[o for o in self._objects() if o.id==self.selection]
        if not matches:raise ValueError('No selected object')
        return matches[0]

    def select(self,name):
        if name=='FDTD':self.selection=name;return
        matches=[o for o in self._objects() if o.name==name]
        if len(matches)!=1:raise ValueError('Selection requires one uniquely named object')
        self.selection=matches[0].id

    def _add(self,category,obj,props):
        self._layout();getattr(self.project,category).append(obj);self.selection=obj.id
        for key,value in props.items():self.set(key.replace('_',' '),value)
        return self._selected()

    def addfdtd(self,**props):
        self._layout();self.selection='FDTD'
        for key,value in props.items():self.set(key.replace('_',' '),value)

    def _add_structure(self,kind,props):
        self._layout();data=Structure(kind=kind).model_dump();remaining={}
        for key,value in props.items():
            if not _shape_property(data,key.replace('_',' ').lower(),value):remaining[key]=value
        shape=Structure.model_validate(data)
        previous=self.project.model_copy(deep=True);selection=self.selection
        try:return self._add('structures',shape,remaining)
        except Exception:
            self.project=previous;self.selection=selection
            raise

    def addrect(self,**props):return self._add_structure('rectangle',props)
    def addcircle(self,**props):return self._add_structure('circle',props)
    def addring(self,**props):return self._add_structure('ring',props)
    def addsphere(self,**props):return self._add_structure('sphere',props)
    def addpoly(self,**props):return self._add_structure('polygon',props)

    def setgeometry(self,**props):
        """Atomically edit coupled solid properties. Lengths/vertices use metres.

        Rotation angles use right-handed fixed world axes, in degrees, applied
        in first/second/third order. The native legacy z rotation applies first.
        """
        self._layout();obj=self._selected()
        if not isinstance(obj,Structure):raise ValueError('Select a structure before editing its geometry.')
        data=obj.model_dump()
        for key,value in props.items():
            if not _shape_property(data,key.replace('_',' ').lower(),value):raise ValueError(f'Unsupported geometry property: {key}')
        replacement=Structure.model_validate(data)
        self.project.structures=[replacement if o.id==obj.id else o for o in self.project.structures]
    def adddipole(self,**props):return self._add('sources',Source(kind='point'),props)
    def addplane(self,**props):
        """Normal-incidence one-way plane with periodic transverse boundaries.

        Set injection='soft' for a bidirectional sheet. Geometry uses SI metres.
        """
        axis=str(props.get('normal',props.get('injection_axis','x'))).lower().replace('-axis','')
        if axis not in ('x','y','z'):raise ValueError('Injection axis must be x, y or z.')
        normal='xyz'.index(axis)
        r=self.project.region
        spans=r.actual_size if props.get('injection','oneway')=='oneway' else tuple(r.interior_bounds(i)[1]-r.interior_bounds(i)[0] for i in range(3))
        size=tuple(0 if i==normal or (i==2 and r.dimension=='2d') else s for i,s in enumerate(spans))
        component=props.get('component',props.get('polarization','Ex' if normal==2 else 'Ez'))
        source=Source(kind='plane',injection=props.get('injection','oneway'),normal=axis,size=size,
                      component=component,theta=props.get('theta'),phi=props.get('phi',0))
        # Validate coupled orientation fields together, independent of keyword
        # order. Sequential theta/phi updates can pass through a longitudinal
        # intermediate direction even when the final vector is transverse.
        remaining={k:v for k,v in props.items() if k not in ('injection','normal','injection_axis','component','polarization','theta','phi')}
        return self._add('sources',source,remaining)
    def addtime(self,**props):return self._add('monitors',Monitor(),props)
    def addtfsf(self,**props):
        """Closed normal-incidence TFSF box. Geometry arguments use SI metres."""
        axis=str(props.get('normal',props.get('injection_axis','x'))).lower().replace('-axis','')
        if axis not in ('x','y','z'):raise ValueError('Injection axis must be x, y or z.')
        component=props.get('component',props.get('polarization','Ex' if axis=='z' else 'Ez'))
        source=Source(kind='tfsf',normal=axis,component=component,
                      theta=props.get('theta'),phi=props.get('phi',0))
        remaining={k:v for k,v in props.items() if k not in ('normal','injection_axis','component','polarization','theta','phi')}
        return self._add('sources',source,remaining)
    def addpower(self,**props):
        """Native planar six-component DFT and signed flux (no automatic normalization)."""
        return self._add('monitors',FieldMonitor(),props)
    def addprofile(self,**props):return self.addpower(**props)
    def adddft(self,**props):
        """Native point DFT monitor, not a general Lumerical surface power monitor."""
        return self._add('monitors',Monitor(spectrum=SpectrumSettings(sampling='frequency',apodization='none')),props)

    def set(self,key,value):
        self._layout();obj=self._selected();data=obj.model_dump();key=key.lower()
        if isinstance(obj,Structure) and _shape_property(data,key,value):pass
        elif key in ('x','y','z') and isinstance(obj,(Structure,Source,Monitor)):
            coords=list(data['center']);coords['xyz'.index(key)]=float(value)*1e6;data['center']=coords
        elif key in ('x span','y span','z span') and hasattr(obj,'size'):
            sizes=list(data['size']);sizes['xyz'.index(key[0])]=float(value)*1e6;data['size']=sizes
        elif key in ('radius','outer radius','inner radius') and isinstance(obj,Structure):
            data['inner_radius' if key=='inner radius' else 'radius']=float(value)*1e6
        elif key in ('dx','dy','dz') and isinstance(obj,Region):
            if obj.mesh_type=='explicit':raise ValueError('Edit explicit nodes with setmesh(), or select a generated mesh first.')
            steps=list(obj.axis_steps);steps['xyz'.index(key[1])]=float(value)*1e6
            data.update(mesh_steps=steps,material_sampling='yee')
        elif key=='mesh step' and isinstance(obj,Region):
            data.update(mesh=float(value)*1e6,mesh_steps=None,mesh_coordinates=None,mesh_type='uniform')
        elif key=='time step' and isinstance(obj,Region):
            data['time_step_override']=None if value is None else float(value)
        elif key=='dimension' and isinstance(obj,Region):
            data['dimension']=str(value).lower()
        elif isinstance(obj,Region) and key=='mesh type':
            data['mesh_type']=str(value).lower()
            if data['mesh_type']!='uniform':data['material_sampling']='yee'
        elif isinstance(obj,Region) and key=='maximum mesh step':
            data['mesh_max']=float(value)*1e6
        elif isinstance(obj,Region) and key=='interface method':
            data['interface_method']=value
            if value=='subpixel':data['material_sampling']='yee'
        elif isinstance(obj,Region) and key in ('material sampling','subpixel quadrature','mesh grading','mesh ppw','mesh auto refine'):
            data[key.replace(' ','_')]=value
        elif isinstance(obj,Region) and key in [a+' '+s+' bc' for a in 'xyz' for s in ('min','max')]:
            axis, side, _ = key.split()
            kind = str(value).strip().lower().replace('-', '').replace(' ', '')
            if kind == 'symmetry':kind = 'symmetric'
            if kind not in ('pml', 'periodic', 'bloch', 'pec', 'antisymmetric', 'pmc', 'symmetric'):
                raise ValueError('Supported native boundaries: PML, Periodic, Bloch, PEC, Anti-Symmetric, PMC, Symmetric')
            data['boundaries'][axis+'_'+side]['kind'] = kind
            # A cyclic boundary always controls both ends of the axis.
            other = axis+'_'+('max' if side=='min' else 'min')
            if kind in ('periodic', 'bloch') or data['boundaries'][other]['kind'] in ('periodic', 'bloch'):
                data['boundaries'][other]['kind'] = kind
            if kind != 'bloch':
                phase=list(data['bloch_phase']);phase['xyz'.index(axis)]=0;data['bloch_phase']=phase
        elif isinstance(obj,Region) and key in ('bloch phase x','bloch phase y','bloch phase z'):
            phase=list(data['bloch_phase']);phase['xyz'.index(key[-1])]=float(value);data['bloch_phase']=phase
        elif key in ('wavelength','wavelength center') and isinstance(obj,Source):
            data['wavelength']=float(value)*1e6
        elif isinstance(obj,Source) and key in ('set wavelength','set frequency'):
            if not value:raise ValueError('Select set time domain to leave automatic range mode')
            data.update(pulse='broadband',time_definition='wavelength' if key=='set wavelength' else 'frequency')
        elif isinstance(obj,Source) and key in ('wavelength start','wavelength stop','frequency start','frequency stop'):
            if key.startswith('wavelength'):
                data['wavelength_start' if key.endswith('start') else 'wavelength_stop']=float(value)*1e6
            else:
                if float(value)<=0:raise ValueError('Frequency must be positive')
                data['wavelength_stop' if key.endswith('start') else 'wavelength_start']=299792458.0/float(value)*1e6
        elif isinstance(obj,Source) and key in ('optimize for short pulse','eliminate discontinuities','chirp bandwidth'):
            field={'optimize for short pulse':'optimize_for_short_pulse','eliminate discontinuities':'eliminate_discontinuities','chirp bandwidth':'chirp_bandwidth_hz'}[key]
            data[field]=float(value) if key=='chirp bandwidth' else bool(value)
        elif isinstance(obj,Source) and key=='pulse type':
            if obj.time_definition in ('wavelength','frequency'):
                raise ValueError('Automatic range mode selects its pulse type. Select set time domain first.')
            if value not in ('standard','broadband'):raise ValueError('Supported pulse types: standard, broadband')
            data.update(pulse='gaussian' if value=='standard' else 'broadband',time_definition='standard')
        elif isinstance(obj,Source) and key in ('use global source settings','override global source settings'):
            data['use_global_source'] = bool(value) if key.startswith('use') else not bool(value)
            if data['use_global_source'] and self.project.global_source is None:
                raise ValueError('Configure global source settings before enabling inheritance')
        elif isinstance(obj,Source) and key in ('theta','phi','dipole type'):
            if key=='dipole type':
                kind=str(value).lower()
                if kind not in ('electric dipole','magnetic dipole'):
                    raise ValueError('Dipole type must be Electric dipole or Magnetic dipole.')
                data['component']=('E' if kind=='electric dipole' else 'H')+obj.component[1]
            else:
                if obj.theta is None:
                    data.update(theta=0. if obj.component[1]=='z' else 90.,phi=90. if obj.component[1]=='y' else 0.)
                data[key]=float(value)
        elif isinstance(obj,Source) and key in ('component','polarization'):
            data.update(component=str(value),theta=None,phi=0.)
        elif isinstance(obj,Source) and key in ('injection axis','normal'):
            normal=str(value).lower().replace('-axis','')
            if normal not in ('x','y','z'):raise ValueError('Injection axis must be x, y or z.')
            old='xyz'.index(obj.normal);axis='xyz'.index(normal);data['normal']=normal
            if obj.kind=='plane' and axis!=old:
                r=self.project.region;size=list(obj.size);center=list(obj.center)
                span=r.actual_size[old] if obj.injection=='oneway' else r.interior_bounds(old)[1]-r.interior_bounds(old)[0]
                size[old]=span if old<2 or r.dimension=='3d' else 0.;size[axis]=0.;center[old]=0.
                data.update(size=size,center=center)
        elif isinstance(obj,Source) and key in ('injection','incident pml cells'):
            data[key.replace(' ','_')]=value
        elif isinstance(obj,Source) and key=='direction':
            data['direction']={'forward':'+','backward':'-'}.get(str(value).lower(),value)
        elif isinstance(obj,Source) and key in ('set time domain','pulselength','offset','frequency','phase'):
            if key == 'set time domain':
                if value and obj.time_definition in ('wavelength','frequency'):
                    params=pulse_parameters(obj)
                    data.update(wavelength=299792458/params.frequency_hz*1e6,pulse_length=params.as_dict()['pulse_length_s'],
                                pulse_offset=params.offset_s,chirp_bandwidth_hz=max(params.frequency_span_hz,1.),
                                pulse='broadband' if params.chirped else 'gaussian')
                data['time_definition'] = 'standard' if value else 'wavelength' if data['pulse']=='broadband' else 'cycles'
            elif key == 'frequency':
                if float(value) <= 0:raise ValueError('Frequency must be positive')
                data['wavelength'] = 299792458.0/float(value)*1e6
            else:data[{'pulselength':'pulse_length','offset':'pulse_offset','phase':'phase'}[key]] = float(value)
        elif isinstance(obj,Monitor) and key in ('use global monitor settings','override global monitor settings'):
            data['use_global_monitor'] = bool(value) if key.startswith('use') else not bool(value)
        elif isinstance(obj,Monitor) and key in ('inherit apodization','time downsample'):
            data[key.replace(' ','_')] = value
        elif isinstance(obj,FieldMonitor) and key in ('record fields','record poynting','record flux','dft precision','spatial interpolation','downsample xyz'):
            data[key.replace(' ','_')] = value
        elif isinstance(obj,FieldMonitor) and key in ('normal','spatial downsample'):
            if key == 'normal':
                axis = str(value).lower()
                if axis not in 'xyz' or len(axis) != 1:raise ValueError('Normal must be x, y or z.')
                old = 'xyz'.index(obj.normal); new = 'xyz'.index(axis)
                size = list(obj.size); size[old] = 1.; size[new] = 0.
                data.update(normal=axis, size=size)
            else:data['downsample'] = value
        elif isinstance(obj,Monitor) and key in ('minimum wavelength','maximum wavelength','frequency points','use wavelength spacing',
                                               'sampling','custom frequencies','chebyshev wavelength','chebyshev nodes','use source limits',
                                               'apodization','apodization center','apodization time width'):
            spec = data['spectrum']
            if key in ('minimum wavelength','maximum wavelength'):
                spec['wavelength_start' if key=='minimum wavelength' else 'wavelength_stop'] = float(value)*1e6
            elif key == 'use wavelength spacing':
                spec['sampling'] = 'wavelength' if value else 'frequency'
            elif key == 'apodization':
                spec['apodization'] = str(value).lower()
            elif key == 'custom frequencies':
                spec.update(sampling='custom',custom_frequencies_hz=np.asarray(value).reshape(-1).tolist())
            elif key == 'sampling':
                spec['sampling'] = str(value).lower()
            else:
                spec[key.replace(' ', '_')] = value
        elif key=='index' and isinstance(obj,Structure):
            name=f'{obj.name} (custom n)'
            material=next((m for m in self.project.materials if m.name==name),None)
            if material:self.project.materials.remove(material)
            self.project.materials.append(Material(name=name,index=value));data['material']=name
        else:
            aliases={'mesh order':'mesh_order','pml layers':'pml_cells',
                     'time steps':'steps','background index':'background_index','polarization':'component',
                     'dt stability factor':'courant_factor'}
            field=aliases.get(key,key)
            if field not in ('name','enabled','material','rotation','mesh_order','pml_cells','steps','background_index','component','backend','precision','amplitude','pulse','courant_factor') or field not in data:
                raise ValueError(f'Unsupported {type(obj).__name__} property: {key}')
            data[field]=value
        replacement=type(obj).model_validate(data)
        if isinstance(obj,Region):self.project.region=replacement
        else:
            for group in ('structures','sources','monitors'):
                items=getattr(self.project,group)
                for i,o in enumerate(items):
                    if o.id==obj.id:items[i]=replacement

    def setnamed(self,name,key,value):
        self.select(name);self.set(key,value)

    def setmesh(self,x,y,z):
        """Set complete, centered node arrays in SI metres, atomically.

        Domain spans follow the array endpoints. Nodes must be strictly
        increasing, with opposite endpoints centered on the native origin.
        A 2D project requires exactly two invariant z endpoints.
        """
        self._layout()
        nodes=[np.asarray(v,dtype=float).reshape(-1)*1e6 for v in (x,y,z)]
        if any(len(v)<2 for v in nodes):raise ValueError('Each axis requires at least two mesh nodes.')
        data=self.project.model_dump();r=data['region']
        r.update(mesh_type='explicit',mesh_steps=None,mesh_coordinates=[v.tolist() for v in nodes],
                 size=[float(v[-1]-v[0]) for v in nodes],material_sampling='yee',mesh_auto_refine=False)
        self.project=Project.model_validate(data)

    def setsourcesignal(self, name, t, amplitude, phase):
        """Set local time/amplitude/unwrapped-phase vectors (s, relative, rad).

        This implements the four-argument form. Automatic bandwidth estimation,
        material fitting and frequency-dependent mesh generation are not provided.
        """
        self._layout()
        matches = [s for s in self.project.sources if s.name == name]
        if len(matches) != 1:raise ValueError('Expected one uniquely named source')
        def vector(value):
            array = np.asarray(value, dtype=float)
            if array.ndim > 2 or (array.ndim == 2 and 1 not in array.shape):
                raise ValueError('Source time signal inputs must be one-dimensional vectors')
            return array.reshape(-1).tolist()
        signal = TimeSignal(time_s=vector(t), amplitude=vector(amplitude), phase_rad=vector(phase))
        source = matches[0]
        data = self.project.model_dump()
        target = next(s for s in data['sources'] if s['id'] == source.id)
        target.update(pulse='sampled', signal=signal.model_dump(), use_global_source=False)
        if target['time_definition'] in ('wavelength','frequency'):target['time_definition']='standard'
        self.project = Project.model_validate(data)

    def setglobalsource(self, key, value):
        """Edit supported global pulse settings; unsupported options raise."""
        self._layout()
        if self.project.global_source is None:
            raise ValueError('Global settings are unavailable. Assign a complete SourceTimeSettings first.')
        # Reuse the same SI conversions and property validation as a local source.
        temporary = FDTD(Project(sources=[Source(**self.project.global_source.model_dump())]))
        temporary.selection = temporary.project.sources[0].id
        temporary.set(key, value)
        fields = SourceTimeSettings.model_fields
        if key.lower() not in ('wavelength','wavelength center','set time domain','pulselength','offset','frequency','pulse',
                              'set wavelength','set frequency','wavelength start','wavelength stop','frequency start','frequency stop',
                              'pulse type','optimize for short pulse','eliminate discontinuities','chirp bandwidth'):
            raise ValueError(f'Unsupported global source property: {key}')
        data = self.project.model_dump()
        data['global_source'] = {k:v for k,v in temporary.project.sources[0].model_dump().items() if k in fields}
        self.project = Project.model_validate(data)

    def getglobalsource(self, key):
        settings = self.project.global_source
        if settings is None:raise ValueError('Global source settings are unavailable')
        params=pulse_parameters(settings)
        values = {'wavelength':299792458.0/params.frequency_hz, 'wavelength center':299792458.0/params.frequency_hz,
                  'frequency':params.frequency_hz, 'pulselength':params.as_dict()['pulse_length_s'],
                  'offset':params.offset_s, 'set time domain':settings.time_definition=='standard',
                  'pulse':settings.pulse,'pulse type':'broadband' if params.chirped else 'standard',
                  'set wavelength':settings.time_definition=='wavelength','set frequency':settings.time_definition=='frequency',
                  'wavelength start':settings.wavelength_start*1e-6,'wavelength stop':settings.wavelength_stop*1e-6,
                  'frequency start':299792458.0/(settings.wavelength_stop*1e-6),'frequency stop':299792458.0/(settings.wavelength_start*1e-6),
                  'optimize for short pulse':settings.optimize_for_short_pulse,'eliminate discontinuities':settings.eliminate_discontinuities,
                  'chirp bandwidth':settings.chirp_bandwidth_hz}
        if key.lower() not in values:raise ValueError(f'Unsupported global source property: {key}')
        return values[key.lower()]

    def setglobalmonitor(self, key, value):
        """Edit global DFT settings, using SI wavelength/frequency/time units."""
        self._layout()
        allowed = ('minimum wavelength','maximum wavelength','frequency points','use wavelength spacing',
                   'sampling','custom frequencies','chebyshev wavelength','chebyshev nodes','use source limits','apodization',
                   'apodization center','apodization time width')
        if key.lower() not in allowed:raise ValueError(f'Unsupported global monitor property: {key}')
        temporary = FDTD(self.project.model_copy(deep=True))
        temporary.project.monitors = [Monitor(spectrum=self.project.global_monitor)]
        temporary.selection = temporary.project.monitors[0].id
        temporary.set(key, value)
        data = self.project.model_dump()
        data['global_monitor'] = temporary.project.monitors[0].spectrum.model_dump()
        self.project = Project.model_validate(data)

    def getglobalmonitor(self, key):
        s = self.project.global_monitor
        values = {'minimum wavelength':s.wavelength_start*1e-6,'maximum wavelength':s.wavelength_stop*1e-6,
                  'frequency points':s.frequency_points,'use wavelength spacing':s.sampling=='wavelength',
                  'sampling':s.sampling,'custom frequencies':np.asarray(s.custom_frequencies_hz),
                  'chebyshev wavelength':s.chebyshev_wavelength,'chebyshev nodes':s.chebyshev_nodes,
                  'use source limits':s.use_source_limits,'apodization':s.apodization,
                  'apodization center':s.apodization_center,'apodization time width':s.apodization_time_width}
        if key.lower() not in values:raise ValueError(f'Unsupported global monitor property: {key}')
        return values[key.lower()]

    def delete(self):
        self._layout()
        if self.selection=='FDTD':raise ValueError('The simulation region cannot be deleted')
        for group in ('structures','sources','monitors'):
            setattr(self.project,group,[o for o in getattr(self.project,group) if o.id!=self.selection])
        self.selection='FDTD'

    def save(self,path):
        Project.model_validate(self.project.model_dump()).save(path)

    def load(self,path):
        self.project=Project.load(path);self.selection='FDTD';self.result=None

    def switchtolayout(self):self.result=None

    def run(self,**kwargs):
        self.result=Simulation(self.project).run(**kwargs)
        return self.result

    def getresult(self,name,result='E'):
        if self.result is None:raise RuntimeError('Run the simulation first')
        if name=='FDTD' and result in ('E','H'):
            return self.result.electric if result=='E' else self.result.magnetic
        planes=[m for m in self.result.frequency_fields if m['name']==name]
        if planes:
            if len(planes)!=1:raise ValueError('Expected a unique monitor name.')
            m=planes[0]
            if result not in ('E','H','P','flux','all'):raise ValueError('Supported plane results: E, H, P, flux, all.')
            if result=='all':return m
            data={k:m[k] for k in ('frequency_hz','points_um','weights','shape','normal_axis')}
            if result in ('E','H'):
                names=m.get('components',['Ex','Ey','Ez','Hx','Hy','Hz'])
                selected=[(i,c) for i,c in enumerate(names) if c.startswith(result)]
                if not selected:raise ValueError(f'No {result} components were recorded.')
                data['components']=[c for i,c in selected]
                data[result]=m['fields'][...,[i for i,c in selected]]
            else:
                if result=='flux' and m['flux'] is None:raise ValueError('Flux was not recorded.')
                data[result]=m['poynting'] if result=='P' else m['flux']
                if result=='P':data['components']=m.get('poynting_components',['x','y','z'])
            return data
        matches=[m for m in self.result.monitor_data() if m['name']==name]
        if len(matches)!=1:raise ValueError('Expected one uniquely named enabled time monitor')
        return matches[0]
