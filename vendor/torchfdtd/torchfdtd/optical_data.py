"""User-supplied passive isotropic optical data, stored in micrometres."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class OpticalData(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    wavelength_um: list[float]=Field(min_length=3,max_length=8192)
    epsilon_real: list[float]=Field(min_length=3,max_length=8192)
    epsilon_imag: list[float]=Field(min_length=3,max_length=8192)
    reference: str=Field(default='',max_length=2000)

    @model_validator(mode='after')
    def validate_table(self):
        if not len(self.wavelength_um)==len(self.epsilon_real)==len(self.epsilon_imag):
            raise ValueError('Wavelength and optical data columns must have the same length.')
        if min(self.wavelength_um)<=0 or min(self.epsilon_imag)<0:
            raise ValueError('Wavelengths must be positive and imaginary permittivity nonnegative (passive data).')
        order=np.argsort(self.wavelength_um)
        if np.any(np.diff(np.asarray(self.wavelength_um)[order])<=0):
            raise ValueError('Wavelength samples must be distinct. Resolve duplicate measurements explicitly.')
        for key in ('wavelength_um','epsilon_real','epsilon_imag'):
            values=getattr(self,key);setattr(self,key,[values[i] for i in order])
        return self

    @property
    def epsilon(self):
        return np.asarray(self.epsilon_real)+1j*np.asarray(self.epsilon_imag)

    @property
    def frequency_hz(self):
        return 299792458/(np.asarray(self.wavelength_um)*1e-6)

    @property
    def fingerprint(self):
        value=self.model_dump(exclude={'reference'})
        return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    @classmethod
    def from_nk(cls,wavelength,n,k,*,unit: Literal['um','nm','m']='um',reference=''):
        factors={'um':1.,'nm':.001,'m':1e6}
        if unit not in factors:raise ValueError('Wavelength unit must be um, nm or m.')
        wavelength,n,k=(np.asarray(v,dtype=float) for v in (wavelength,n,k))
        if wavelength.ndim!=1 or n.shape!=wavelength.shape or k.shape!=wavelength.shape:
            raise ValueError('Wavelength, n and k must be matching one-dimensional arrays.')
        if np.any(~np.isfinite(n)) or np.any(~np.isfinite(k)) or np.any(n<0) or np.any(k<0):
            raise ValueError('n and k must be finite and nonnegative for passive isotropic data.')
        epsilon=(n+1j*k)**2
        return cls(wavelength_um=(wavelength*factors[unit]).tolist(),epsilon_real=epsilon.real.tolist(),
                   epsilon_imag=epsilon.imag.tolist(),reference=reference)

    @classmethod
    def from_text(cls,text,*,kind: Literal['nk','epsilon']='nk',unit: Literal['um','nm','m']='um',reference=''):
        if kind not in ('nk','epsilon') or unit not in ('um','nm','m'):
            raise ValueError('Select nk or epsilon columns and wavelength unit um, nm or m.')
        if len(text)>2_000_000:raise ValueError('Optical data text exceeds 2 MB.')
        rows=[];header_seen=False
        for lineno,line in enumerate(text.lstrip('\ufeff').splitlines(),1):
            line=line.split('#',1)[0].strip()
            if not line:continue
            cells=re.split(r'[,;\s]+',line)
            if len(cells)!=3:raise ValueError(f'Line {lineno}: expected exactly three columns.')
            try:row=[float(v) for v in cells]
            except ValueError:
                normalized=[c.lower() for c in cells]
                expected=('n','k') if kind=='nk' else ('epsilon_real','epsilon_imag')
                if rows or header_seen or not normalized[0].startswith(('wavelength','lambda')) or tuple(normalized[1:])!=expected:
                    raise ValueError(f'Line {lineno}: invalid numeric data or column header.') from None
                declared=next((u for u in ('um','nm','m') if normalized[0].endswith('_'+u)),None)
                if declared and declared!=unit:raise ValueError('Header wavelength unit differs from the selected unit.')
                header_seen=True;continue
            rows.append(row)
            if len(rows)>8192:raise ValueError('At most 8192 optical samples are supported.')
        if len(rows)<3:raise ValueError('At least three optical samples are required.')
        a=np.asarray(rows)
        if kind=='nk':return cls.from_nk(a[:,0],a[:,1],a[:,2],unit=unit,reference=reference)
        return cls(wavelength_um=(a[:,0]*{'um':1.,'nm':.001,'m':1e6}[unit]).tolist(),
                   epsilon_real=a[:,1].tolist(),epsilon_imag=a[:,2].tolist(),reference=reference)

    @classmethod
    def from_csv(cls,path,**kwargs):
        kwargs.setdefault('reference',Path(path).name)
        return cls.from_text(Path(path).read_text(encoding='utf-8-sig'),**kwargs)
