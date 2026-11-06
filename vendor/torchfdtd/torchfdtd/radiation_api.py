"""Bounded CPU diffraction postprocessing of completed native browser jobs."""
from fastapi import HTTPException
from pydantic import BaseModel, Field, StrictInt

from .models import Project
from .radiation_io import native_radiation_plane
from .radiation import diffraction_orders, diffraction_efficiency


class DiffractionRequest(BaseModel):
    monitor: str
    frequency_index: int = Field(default=0,ge=0)
    orders: list[tuple[StrictInt,StrictInt]] = Field(default_factory=lambda:[(0,0)],min_length=1,max_length=49)
    refractive_index: float = Field(default=1.,gt=0,le=20)
    confirm_homogeneous_exterior: bool = False
    reference: str | None = None
    subtract_incident: bool = False


def register_radiation_routes(app,get_job):
    def completed(key):
        job=get_job(key)
        if job.get('status')!='completed':raise HTTPException(409,'Finish a native run before diffraction postprocessing.')
        return job

    @app.get('/api/jobs/{key}/diffraction-monitors')
    def monitors(key:str):
        job=completed(key)
        return [dict(id=m['id'],name=m['name'],normal=m['normal_axis'],components=m['components'],
            frequency_thz=(m['frequency_hz']*1e-12).tolist(),points=len(m['weights']))
            for m in job.get('frequency_fields',[])]

    @app.post('/api/jobs/{key}/diffraction')
    def calculate(key:str,request:DiffractionRequest):
        job=completed(key)
        if not request.confirm_homogeneous_exterior:
            raise HTTPException(422,'Confirm that sample and reference planes lie in the declared homogeneous, lossless isotropic exterior, outside PML.')
        def plane(current):
            try:record=next(m for m in current.get('frequency_fields',[]) if m['id']==request.monitor)
            except StopIteration:raise ValueError('The selected frequency monitor is missing. Reference runs must retain the same monitor ID.')
            result=native_radiation_plane(record,frequency_index=request.frequency_index)
            region=Project.model_validate(current['project']).region
            a='xyz'.index(result.normal);b,c=(a+1)%3,(a+2)%3
            if any(region.boundaries.pair(d)[0].kind not in ('periodic','bloch') for d in (b,c)):
                raise ValueError('Diffraction requires periodic/Bloch transverse axes and a complete unit-cell plane. An isolated open plane is not a far-field surface.')
            position=float(result.points_um[0,a]);low,high=region.interior_bounds(a)
            if not low<position<high:raise ValueError('Move the frequency plane strictly outside PML and inside the physical interior, then rerun.')
            return result,dict(period_um=tuple(region.actual_size[d] for d in (b,c)),
                bloch_wavevector_per_um=tuple(region.bloch_phase[d]/region.actual_size[d] for d in (b,c)),
                refractive_index=request.refractive_index)
        try:
            sample,kwargs=plane(job)
            if len(sample.points_um)*len(request.orders)>2_000_000:
                raise ValueError('Order/point workspace limit exceeded. Request fewer orders or downsample the monitor.')
            if request.subtract_incident and not request.reference:
                raise ValueError('Incident subtraction requires a matched reference run.')
            orders=diffraction_orders(sample,request.orders,**kwargs)
            result=dict(frequency_thz=float(sample.frequency_hz[0])*1e-12,
                normal=sample.normal,transverse_axes=['xyz'[("xyz".index(sample.normal)+d)%3] for d in (1,2)],
                period_um=list(kwargs['period_um']),phase_origin_um=orders.phase_origin_um.tolist(),
                exterior_index=request.refractive_index,normalized=False,orders=orders.orders.tolist(),
                propagating=orders.propagating[0].tolist(),forward_power=orders.forward_power[0].tolist(),
                backward_power=orders.backward_power[0].tolist(),units='reduced E*H * s^2 * m^2',
                note='Raw directional powers are not efficiencies. Evanescent orders carry zero real directional power; grazing cutoff is rejected.')
            if request.reference:
                reference,reference_kwargs=plane(completed(request.reference))
                if kwargs!=reference_kwargs:raise ValueError('Reference periodic cell and Bloch wavevector must match.')
                for direction in ('forward','backward'):
                    result[direction+'_efficiency']=diffraction_efficiency(sample,reference,request.orders,
                        direction=direction,subtract_incident=request.subtract_incident,**kwargs)[0].tolist()
                result.update(normalized=True,efficiency_units='fraction of matched incident reference power',
                    subtract_incident=request.subtract_incident)
            return result
        except (ValueError,TypeError,KeyError) as error:
            raise HTTPException(422,str(error)) from error
