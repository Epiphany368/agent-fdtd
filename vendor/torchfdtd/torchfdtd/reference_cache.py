"""Bounded CPU storage for fixed homogeneous spectral-plane references."""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
import torch


_TENSORS=('fields','frequency_hz','points_um','weights')


def _copy(planes,device):
    result={}
    for name,plane in planes.items():
        if any(getattr(plane,k).requires_grad for k in _TENSORS):
            raise ValueError('Reference cache accepts only fixed no-grad planes.')
        result[name]=replace(plane,**{k:getattr(plane,k).detach().to(device).clone() for k in _TENSORS},
                             report=deepcopy(plane.report))
    return result


class PlaneReferenceCache:
    """LRU cache of CPU plane tensors, never full time-domain field histories.

    budget_bytes limits retained tensor storage only. Python metadata and
    temporary returned copies are additional. Oversize entries are returned
    without retention. Instances are intended for sequential case replay, not
    concurrent threads. Numerical cache keys are supplied by the layer API.
    """
    def __init__(self,budget_bytes=64*1024**2):
        if isinstance(budget_bytes,bool) or not isinstance(budget_bytes,int) or budget_bytes<0:
            raise ValueError('Reference cache budget must be a nonnegative integer.')
        self.budget_bytes=budget_bytes
        self._entries=OrderedDict()
        self.tensor_bytes=0
        self.hits=self.misses=self.evictions=0

    def clear(self):
        self._entries.clear()
        self.tensor_bytes=0

    def _get(self,key,device,compute):
        if key in self._entries:
            self.hits+=1
            planes,size=self._entries.pop(key)
            self._entries[key]=(planes,size)
            return _copy(planes,device)
        self.misses+=1
        planes=compute()
        size=sum(getattr(p,k).numel()*getattr(p,k).element_size() for p in planes.values() for k in _TENSORS)
        if size<=self.budget_bytes:
            stored=_copy(planes,'cpu')
            while self.tensor_bytes+size>self.budget_bytes:
                _,(_,old_size)=self._entries.popitem(last=False)
                self.tensor_bytes-=old_size
                self.evictions+=1
            self._entries[key]=(stored,size)
            self.tensor_bytes+=size
        return planes
