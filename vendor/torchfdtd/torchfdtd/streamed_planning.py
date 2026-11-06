"""Metadata-only work/admission planning, without performance predictions."""
from dataclasses import asdict, dataclass, replace
from itertools import islice
import json

import torch

from .models import Project
from .streamed import StreamedAdjointOptions, StreamedSimulation, estimate_streamed_memory
from .memory_profile import host_memory


@dataclass(frozen=True)
class StreamedWorkPlan:
    options: StreamedAdjointOptions | None
    _report_json: str
    _project_json: str

    @property
    def report(self):
        """Independent JSON-compatible copy of candidate evidence."""
        return json.loads(self._report_json)

    @property
    def project_snapshot(self):
        return json.loads(self._project_json)


def _bounded(values,name):
    result=list(islice(iter(values),65))
    if not result or len(result)>64:raise ValueError(name+' must contain one to 64 entries.')
    return result


def plan_streamed_work(project,options=None,*,diagonal=False,candidates=None,
                       temporal_depths=None,host_free_reserve_bytes=0,selection=None):
    """Report admitted Pareto candidates for fixed point-observation workloads.

    No choice is made by default. Explicit selection minimizes a declared work
    metric, not elapsed time. Caller geometry/optimizer memory and OS file cache
    are outside solver reservations. Execution must recheck live admission.
    """
    from .streamed_work import estimate_streamed_work
    if selection not in (None,'min_disk_traffic','min_compute_work'):
        raise ValueError('selection must be None, min_disk_traffic or min_compute_work.')
    if not isinstance(diagonal,bool):raise ValueError('diagonal must be boolean.')
    if isinstance(host_free_reserve_bytes,bool) or not isinstance(host_free_reserve_bytes,int) or host_free_reserve_bytes<0:
        raise ValueError('host_free_reserve_bytes must be a nonnegative integer.')
    base=options or StreamedAdjointOptions()
    if not isinstance(base,StreamedAdjointOptions):raise ValueError('options must be StreamedAdjointOptions.')
    snapshot=Project.model_validate(project.model_dump())
    # Constructor validates source/material/observer scope without field banks.
    StreamedSimulation(snapshot,base)
    if any(face.kind in ('pmc','symmetric') for a in range(3) for face in snapshot.region.boundaries.pair(a)):
        raise ValueError('PMC/symmetric faces are not implemented by the streamed work planner; StreamedSimulation executes them directly.')
    if candidates is not None and temporal_depths is not None:
        raise ValueError('Supply candidates or temporal_depths, not both.')
    if candidates is None:
        t=snapshot.region.steps;n=snapshot.region.shape[0]
        depths=_bounded(temporal_depths,'temporal_depths') if temporal_depths is not None else [1,base.temporal_depth,2*base.temporal_depth,4*base.temporal_depth]
        if any(isinstance(d,bool) or not isinstance(d,int) or d<1 for d in depths):
            raise ValueError('Temporal depths must be positive integers.')
        widths=sorted({1,min(n,max(1,base.slab_width//2)),min(n,base.slab_width),
                       min(n,2*base.slab_width),min(n,4*base.slab_width),n})
        depths=sorted({min(t,d) for d in depths})
        if len(widths)*len(depths)>64:raise ValueError('Generated candidate count exceeds 64.')
        proposed=[replace(base,slab_width=w,temporal_depth=d) for w in widths for d in depths]
    else:
        proposed=_bounded(candidates,'candidates')
    if any(not isinstance(c,StreamedAdjointOptions) for c in proposed):
        raise ValueError('Candidates must be StreamedAdjointOptions.')
    if any(c.checkpoints!=base.checkpoints for c in proposed):
        raise ValueError('Candidates must preserve the requested global checkpoint count.')
    proposed=list(dict.fromkeys(proposed))
    available=host_memory()['available_bytes']
    if available is None and host_free_reserve_bytes:
        raise ValueError('Live available RAM is unavailable, so the requested RAM floor cannot be verified.')
    records=[]
    for index,candidate in enumerate(proposed):
        record=dict(index=index,options=asdict(candidate),admitted=False)
        try:
            memory=estimate_streamed_memory(snapshot,candidate,diagonal=diagonal)
            current_available=host_memory()['available_bytes']
            record['host_available_bytes']=current_available
            if current_available is None and host_free_reserve_bytes:
                raise ValueError('Live available RAM became unavailable, so the remaining RAM floor cannot be verified.')
            if current_available is not None and memory['host_reservation_bytes']>current_available-host_free_reserve_bytes:
                raise ValueError('Solver reservation would violate the requested remaining RAM floor.')
            work=estimate_streamed_work(snapshot,candidate,diagonal=diagonal)
            record.update(admitted=True,memory=memory,work=work)
        except (ValueError,RuntimeError,OSError) as exc:
            record['reason']=str(exc)
        records.append(record)
    admitted=[r for r in records if r['admitted']]
    def metrics(row):
        return (row['work']['total_state_io_bytes'],row['work']['total_cell_steps'],
                row['memory']['host_reservation_bytes'],
                row['memory']['gpu_reservation_bytes'] if torch.device(row['options']['device']).type=='cuda' else 0)
    pareto=[]
    for row in admitted:
        values=metrics(row)
        if not any(all(a<=b for a,b in zip(metrics(other),values)) and
                   any(a<b for a,b in zip(metrics(other),values)) for other in admitted):
            pareto.append(row['index'])
    selected=None
    if selection is not None and admitted:
        column=0 if selection=='min_disk_traffic' else 1
        selected=min(admitted,key=lambda r:(metrics(r)[column],metrics(r)[1-column],
            metrics(r)[2],metrics(r)[3],r['index']))['index']
    report=dict(schema_version=1,selection=selection,selected_index=selected,candidates=records,
        admitted_indices=[r['index'] for r in admitted],pareto_indices=pareto,
        pareto_metrics=['total_state_io_bytes','total_cell_steps','host_reservation_bytes','gpu_reservation_bytes'],
        host_available_bytes=available,host_free_reserve_bytes=host_free_reserve_bytes,
        performance_prediction=False,execution_readmission_required=True,
        ram_floor_scope="Planning-time live snapshot only. Selected options do not persist the RAM floor; callers must replan with the floor before execution.",
        scope='Fixed-step nondispersive scalar/diagonal epsilon, point observations, existing streamed source scope. Caller graphs and OS file cache excluded.')
    return StreamedWorkPlan(None if selected is None else proposed[selected],
        json.dumps(report,sort_keys=True,allow_nan=False),snapshot.model_dump_json())
