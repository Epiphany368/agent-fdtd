"""Constrained passive Drude/Lorentz fitting with independently specified loss.

Linear nonnegative strengths seed a bounded nonlinear fit of oscillator rates.
Positive strengths/damping and epsilon-infinity >= 1 enforce the implemented
isotropic passive model. A finite data band does not validate extrapolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Literal
import warnings

import numpy as np
from pydantic import Field, model_validator

from .models import Model, Material, LorentzPole, MaterialProvenance
from .optical_data import OpticalData
from .materials import permittivity


class FitOptions(Model):
    max_poles: int=Field(default=6,ge=1,le=16)
    include_drude: bool=True
    epsilon_inf: float | None=Field(default=None,ge=1,le=400)
    tolerance: float=Field(default=1e-3,gt=0,le=1)
    wavelength_range_um: tuple[float,float] | None=None
    target: Literal['analytic','ade']='analytic'
    dt_s: float=Field(default=0,ge=0)
    loss_weight: float=Field(default=1,gt=0,le=100)
    starts: int=Field(default=2,ge=1,le=6)
    max_nfev: int=Field(default=400,ge=10,le=3000)
    seed: int=Field(default=0,ge=0,le=2**32-1)

    @model_validator(mode='after')
    def valid_options(self):
        if self.target=='ade' and self.dt_s<=0:
            raise ValueError('An ADE-target fit needs a positive simulation timestep.')
        if self.wavelength_range_um is not None and not 0<self.wavelength_range_um[0]<self.wavelength_range_um[1]:
            raise ValueError('Fit wavelength range must be positive and increasing.')
        return self


class OpticalDataRequest(Model):
    text: str=Field(max_length=2_000_000)
    kind: Literal['nk','epsilon']='nk'
    unit: Literal['um','nm','m']='um'
    reference: str=Field(default='',max_length=2000)
    # Provenance of the raw table (docs/MATERIAL_FITTING.md); the server hashes the submitted text.
    source: str=Field(default='',max_length=200)
    licence: str=Field(default='',max_length=2000)
    file_name: str=Field(default='',max_length=260)


class MaterialFitRequest(Model):
    data: OpticalData
    options: FitOptions=Field(default_factory=FitOptions)
    name: str=Field(default='Fitted material',min_length=1,max_length=100)
    color: str='#6ca8dd'
    provenance: MaterialProvenance | None=None


@dataclass
class MaterialFitResult:
    material: Material
    report: dict

    @property
    def converged(self):return self.report['converged']

    def require_tolerance(self):
        if not self.converged:raise ValueError('Material fit did not reach the requested tolerance. Inspect its report.')
        return self.material

    def as_dict(self):return dict(material=self.material.model_dump(),report=self.report)

    def save(self,path):
        Path(path).write_text(json.dumps(self.as_dict(),indent=2)+'\n',encoding='utf8')


def _weights(frequency):
    order=np.argsort(frequency);f=frequency[order]
    width=np.empty(len(f));width[0]=(f[1]-f[0])/2;width[-1]=(f[-1]-f[-2])/2
    width[1:-1]=(f[2:]-f[:-2])/2
    result=np.empty_like(width);result[order]=width/width.sum()
    return result


def _metrics(expected,actual,weights):
    error=abs(actual-expected);scale=np.maximum(abs(expected),1)
    n,k=np.sqrt(expected),np.sqrt(actual)
    return dict(normalized_rms=float(np.sqrt(np.sum(weights*(error/scale)**2))),
                relative_l2=float(np.sqrt(np.sum(weights*error**2)/max(np.sum(weights*abs(expected)**2),1e-300))),
                max_normalized_error=float(np.max(error/scale)),
                max_abs_n_error=float(np.max(abs(n.real-k.real))),
                max_abs_k_error=float(np.max(abs(n.imag-k.imag))))


def material_fit_report(material,*,dt_s=0):
    """Recompute errors for current coefficients, including manual edits."""
    if material.samples is None:raise ValueError('Material has no retained optical samples.')
    data=material.samples;w=np.asarray(data.wavelength_um)
    band=material.fit_band_um or (w[0],w[-1]);mask=(w>=band[0])&(w<=band[1])
    f=data.frequency_hz[mask];measured=data.epsilon[mask];weights=_weights(f)
    analytic=permittivity(material,f);numerical=permittivity(material,f,dt_s) if dt_s else None
    if not np.all(np.isfinite(analytic)) or (numerical is not None and not np.all(np.isfinite(numerical))):
        raise ValueError('A material resonance is singular in the comparison band.')
    index=np.sqrt(measured);fit=np.sqrt(analytic);ade=np.sqrt(numerical) if numerical is not None else None
    return dict(wavelength_um=w[mask].tolist(),measured_n=index.real.tolist(),measured_k=index.imag.tolist(),
                fitted_n=fit.real.tolist(),fitted_k=fit.imag.tolist(),
                numerical_n=None if ade is None else ade.real.tolist(),numerical_k=None if ade is None else ade.imag.tolist(),
                analytic=_metrics(measured,analytic,weights),ade=None if ade is None else _metrics(measured,numerical,weights),
                dt_s=dt_s,fit_band_um=list(band),sample_count=int(mask.sum()),data_sha256=data.fingerprint)


def fit_material(data: OpticalData,*,name='Fitted material',color='#6ca8dd',options: FitOptions | dict | None=None,
                 provenance: MaterialProvenance | None=None):
    """Fit a passive isotropic material. Returns best fit plus explicit errors.

    The objective integrates squared complex-permittivity error over frequency,
    normalized pointwise by max(1, abs(epsilon)). `max_poles` is an upper bound.
    A tolerance failure is returned explicitly, never converted into success.
    ``target='ade'`` fits the bilinear discrete response at exactly ``dt_s``.
    ``provenance`` is stored on the returned material unchanged.
    """
    from scipy.optimize import least_squares, nnls, lsq_linear

    start=time.perf_counter()
    data=OpticalData.model_validate(data.model_dump() if isinstance(data,OpticalData) else data)
    opt=FitOptions.model_validate(options.model_dump() if isinstance(options,FitOptions) else options or {})
    w=np.asarray(data.wavelength_um)
    requested=opt.wavelength_range_um or (float(w[0]),float(w[-1]))
    if requested[0]<w[0] or requested[1]>w[-1]:raise ValueError('Fit range must remain inside the input data band.')
    mask=(w>=requested[0])&(w<=requested[1]);f=data.frequency_hz[mask];target=data.epsilon[mask]
    if len(f)<3:raise ValueError('Fit range needs at least three distinct measured samples.')
    if opt.dt_s and max(f)*opt.dt_s>=.5:raise ValueError('Material frequencies must lie below timestep Nyquist.')
    omega=2*np.pi*f if opt.target=='analytic' else 2/opt.dt_s*np.tan(np.pi*f*opt.dt_s)
    scale=float(np.sqrt(min(omega)*max(omega)));z=omega/scale
    if not np.isfinite(scale) or scale<=0 or max(omega)>=1e18:
        raise ValueError('Optical frequencies exceed the supported oscillator rate range.')
    weights=_weights(f);weighted=np.sqrt(weights)/np.maximum(abs(target),1)
    imag_factor=np.sqrt(opt.loss_weight)
    base=opt.epsilon_inf if opt.epsilon_inf is not None else 1.
    free_constant=opt.epsilon_inf is None
    strength_max=1e40/scale**2
    rlow,rhigh=.1*min(z),min(10*max(z),1e18/scale)
    glow,ghigh=1e-5*min(z),min(20*max(z),1e18/scale)

    def stack(value):
        if value.ndim==1:v=value*weighted
        else:v=value*weighted[:,None]
        return np.concatenate((v.real,imag_factor*v.imag),axis=0)

    def basis(rates):
        return np.column_stack([1/(r*r-z*z-1j*g*z) for r,g in rates]) if rates else np.empty((len(z),0),complex)

    def linear(rates):
        b=basis(rates)
        matrix=np.column_stack((np.ones(len(z)),b)) if free_constant else b
        if matrix.shape[1]==0:return np.empty(0),float(np.dot(stack(base-target),stack(base-target)))
        a=stack(matrix);rhs=stack(target-base);norms=np.linalg.norm(a,axis=0)
        if np.any(norms<=0) or not np.all(np.isfinite(norms)) or not np.all(np.isfinite(rhs)):
            raise ValueError('Optical data magnitude exceeds the numerical range of the fitter.')
        a=a/norms
        try:coeff=nnls(a,rhs,maxiter=max(100,10*a.shape[1]))[0]/norms
        except RuntimeError:coeff=lsq_linear(a,rhs,bounds=(0,np.inf),tol=1e-10).x/norms
        if free_constant and coeff[0]>399:
            rest=nnls(a[:,1:],rhs-399*stack(np.ones(len(z),complex)),maxiter=300)[0]/norms[1:] if rates else np.empty(0)
            coeff=np.r_[399.,rest]
        coeff=np.clip(coeff,0,np.r_[399.,np.full(len(rates),strength_max)] if free_constant else strength_max)
        residual=stack(base+matrix@coeff-target)
        return coeff,float(residual@residual)

    def refine(rates,coeff):
        count=len(rates);nonzero=[i for i,(r,g) in enumerate(rates) if r>0]
        ncoeff=count+int(free_constant)
        x=np.r_[coeff,np.log([rates[i][0] for i in nonzero]),np.log([g for r,g in rates])]
        lower=np.r_[np.zeros(ncoeff),np.full(len(nonzero),np.log(rlow)),np.full(count,np.log(glow))]
        upper=np.r_[([399.] if free_constant else []),np.full(count,strength_max),
                    np.full(len(nonzero),np.log(rhigh)),np.full(count,np.log(ghigh))]

        def model_jac(x):
            c=x[:ncoeff];r=np.zeros(count);r[nonzero]=np.exp(x[ncoeff:ncoeff+len(nonzero)]);g=np.exp(x[-count:])
            a=c[int(free_constant):];den=r[None,:]**2-z[:,None]**2-1j*g[None,:]*z[:,None]
            b=1/den;value=base+(c[0] if free_constant else 0)+b@a
            jac=np.column_stack(([np.ones(len(z))] if free_constant else [])+
                                [b[:,i] for i in range(count)]+
                                [-2*a[i]*r[i]**2/den[:,i]**2 for i in nonzero]+
                                [1j*a[i]*g[i]*z/den[:,i]**2 for i in range(count)])
            return value,jac,r,g

        result=least_squares(lambda x:stack(model_jac(x)[0]-target),np.clip(x,lower,upper),
                             jac=lambda x:stack(model_jac(x)[1]),bounds=(lower,upper),x_scale='jac',
                             max_nfev=opt.max_nfev,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        _,_,r,g=model_jac(result.x)
        return list(zip(r,g)),result.x[:ncoeff],float(result.fun@result.fun),bool(result.success),result.nfev

    def material(rates,coeff):
        eps=base+(float(coeff[0]) if free_constant else 0)
        poles=[LorentzPole(resonance_rad_s=float(r*scale),strength_rad_s_squared=float(a*scale**2),
                           damping_rad_s=float(g*scale))
               for (r,g),a in zip(rates,coeff[int(free_constant):]) if a>1e-14]
        return Material(name=name,color=color,model='multipole' if poles else 'dielectric',epsilon_inf=eps,
                        index=float(np.sqrt(eps)),poles=poles,samples=data,
                        fit_band_um=(float(w[mask][0]),float(w[mask][-1])),fit_dt_s=opt.dt_s if opt.target=='ade' else None,
                        provenance=provenance)

    rates=[];coeff,cost=linear(rates);history=[];rng=np.random.default_rng(opt.seed)
    best=material(rates,coeff);quality=material_fit_report(best,dt_s=opt.dt_s)
    current_error=quality[opt.target]['normalized_rms']
    history.append(dict(poles=0,normalized_rms=current_error,optimizer_success=True,evaluations=0))
    # Broad deterministic candidates plus measured local loss maxima.
    peaks=[z[i] for i in range(1,len(z)-1) if target.imag[i]>=target.imag[i-1] and target.imag[i]>target.imag[i+1]]
    peaks=sorted(peaks,key=lambda x:float(target.imag[np.argmin(abs(z-x))]),reverse=True)[:12]
    resonance=np.unique(np.r_[np.geomspace(max(rlow,.25*min(z)),min(rhigh,4*max(z)),32),peaks])
    candidates=[(r,g) for r in resonance for g in np.clip(np.array([.01,.05,.2,.7])*r,glow,ghigh)]
    if opt.include_drude:candidates.extend((0.,g) for g in np.geomspace(max(glow,.003*min(z)),min(ghigh,3*max(z)),12))
    pole_limit=min(opt.max_poles,max(1,(2*len(f)-1)//3))
    for count in range(1,pole_limit+1):
        if current_error<=opt.tolerance:break
        trials=[]
        for candidate in candidates:
            if candidate[0]==0 and any(r==0 for r,g in rates):continue
            trial_rates=rates+[candidate];c,score=linear(trial_rates);trials.append((score,trial_rates,c))
        trials.sort(key=lambda t:t[0]);winning=(cost,rates,coeff,False,0);total_nfev=0
        # Zero resonance is a distinct model topology. A bounded nonzero
        # resonance cannot reach it during nonlinear refinement. Comparing
        # only the lowest linear seed scores can therefore exclude Drude
        # even when the measured response is exactly a single Drude pole.
        families=[[t for t in trials if t[1][-1][0]>0],
                  [t for t in trials if t[1][-1][0]==0]]
        for family in families:
            for attempt in range(min(opt.starts,len(family))):
                _,trial_rates,c=family[attempt]
                if attempt:
                    trial_rates=[(0. if r==0 else float(np.clip(r*np.exp(rng.normal(0,.1)),rlow,rhigh)),
                                  float(np.clip(g*np.exp(rng.normal(0,.25)),glow,ghigh))) for r,g in trial_rates]
                    c,_=linear(trial_rates)
                rr,cc,score,success,nfev=refine(trial_rates,c)
                total_nfev+=nfev
                if score<winning[0]:winning=(score,rr,cc,success,nfev)
        cost,rates,coeff,success,nfev=winning
        best=material(rates,coeff);quality=material_fit_report(best,dt_s=opt.dt_s)
        current_error=quality[opt.target]['normalized_rms']
        history.append(dict(poles=len(best.poles),normalized_rms=current_error,optimizer_success=success,evaluations=total_nfev))
        if len(rates)<count:break
    quality.update(converged=current_error<=opt.tolerance,tolerance=opt.tolerance,target=opt.target,
                   pole_count=len(best.poles),requested_max_poles=opt.max_poles,allowed_max_poles=pole_limit,
                   options=opt.model_dump(),history=history,seconds=time.perf_counter()-start,
                   passivity='positive oscillator strengths and damping, epsilon infinity >= 1',
                   error_definition='Frequency-weighted RMS of abs(delta epsilon) / max(1, abs(measured epsilon)).',
                   limitations=['No accuracy claim outside the fitted sample band.',
                                'Bounded nonlinear refinement can converge to a local minimum. Failure to reach tolerance is explicit.',
                                'A passive finite-band fit does not prove consistency or uniqueness of measured data.',
                                'Validate time/mesh convergence and device observables separately.'])
    return MaterialFitResult(best,quality)


class MaterialBandWarning(UserWarning):
    """The simulation band extends beyond the fitted wavelength band of a material."""


def fit_band_extrapolation(material,wavelength_range_um,*,label=None):
    """The extrapolation warning for a simulation band outside the fitted band, or None.

    A material without a fitted band has nothing to check; the solver's estimate
    warns separately that unfitted samples carry no accuracy statement.
    """
    if material.fit_band_um is None:return None
    low,high=material.fit_band_um
    start,stop=float(min(wavelength_range_um)),float(max(wavelength_range_um))
    if not np.isfinite([start,stop]).all() or start<=0:raise ValueError('The simulation wavelength band must be finite and positive.')
    if start>=low and stop<=high:return None
    prefix=f'{label}: ' if label else ''
    return (f'{prefix}source wavelength band {start:g}\u2013{stop:g} um lies outside the {material.name} fit band '
            f'({low:g}\u2013{high:g} um). Extrapolated material accuracy is not validated.')


def discretization_report(material,dt_s,*,band_um=None,count=201):
    """n/k error of the trapezoidal ADE at timestep ``dt_s`` against the fitted continuum.

    The discrete constitutive response replaces omega by (2/dt) tan(omega dt/2)
    (``materials.permittivity(material, f, dt)``, the path tests/test_physics_g3_a.py
    checks against an independent bilinear evaluation and a driven cell). The
    band defaults to the fitted band, else the retained sample band. Frequencies
    at or above the timestep Nyquist limit are refused by ``permittivity``.
    """
    if not np.isfinite(dt_s) or dt_s<=0:raise ValueError('A positive timestep is required for the discretization error.')
    if band_um is None:
        if material.fit_band_um is not None:band_um=material.fit_band_um
        elif material.samples is not None:band_um=(material.samples.wavelength_um[0],material.samples.wavelength_um[-1])
        else:raise ValueError('Declare a wavelength band: the material has neither a fitted band nor retained samples.')
    low,high=float(min(band_um)),float(max(band_um))
    if not np.isfinite([low,high]).all() or low<=0:raise ValueError('The wavelength band must be finite and positive.')
    if not isinstance(count,int) or isinstance(count,bool) or count<2:raise ValueError('count must be an integer of at least 2.')
    wavelength=np.linspace(low,high,count);f=299792458/(wavelength*1e-6)
    analytic=permittivity(material,f);numerical=permittivity(material,f,dt_s)
    if not np.all(np.isfinite(analytic)) or not np.all(np.isfinite(numerical)):
        raise ValueError('A material resonance is singular in the comparison band.')
    delta=np.sqrt(numerical)-np.sqrt(analytic)
    n_at,k_at=int(np.argmax(abs(delta.real))),int(np.argmax(abs(delta.imag)))
    return dict(dt_s=float(dt_s),band_um=[low,high],sample_count=count,
                max_abs_n_error=float(abs(delta.real[n_at])),max_abs_n_error_at_um=float(wavelength[n_at]),
                max_abs_k_error=float(abs(delta.imag[k_at])),max_abs_k_error_at_um=float(wavelength[k_at]),
                definition='max over the band of |Re/Im(sqrt(eps_ADE(f; dt)) - sqrt(eps(f)))| with omega -> (2/dt) tan(omega dt/2); '
                           'the constitutive error of the trapezoidal ADE only, not Yee spatial dispersion.')


@dataclass
class MaterialImportResult:
    """One imported table: the fitted material with provenance, the fit report and its checks."""
    material: Material
    report: dict
    provenance: MaterialProvenance
    discretization: dict | None
    warnings: list

    @property
    def converged(self):return self.report['converged']

    def require_tolerance(self):
        if not self.converged:raise ValueError('Material fit did not reach the requested tolerance. Inspect its report.')
        return self.material

    def as_dict(self):
        return dict(material=self.material.model_dump(),report=self.report,provenance=self.provenance.model_dump(),
                    discretization=self.discretization,warnings=list(self.warnings))


def import_material_table(path=None,*,text=None,kind: Literal['nk','epsilon']='nk',unit: Literal['um','nm','m']='um',
                          source,licence='',file_name=None,name='Fitted material',color='#6ca8dd',
                          options: FitOptions | dict | None=None,dt_s=0.,simulation_band_um=None,imported=None):
    """Import a CSV/text n/k or complex-permittivity table, fit it and record its provenance.

    Exactly one of ``path`` (a file) or ``text`` (its contents) is read; the raw
    bytes are hashed into ``MaterialProvenance.raw_sha256`` before any parsing.
    The existing passive fitter (``fit_material``) runs over the declared band
    (``options.wavelength_range_um``, default the whole table). With ``dt_s`` the
    report carries the ADE error at that timestep and ``discretization`` the
    n/k error of the discrete response against the fitted continuum. With
    ``simulation_band_um`` a band outside the fitted band raises
    ``MaterialBandWarning`` (``warnings.warn``) and the message is kept in
    ``warnings``; the solver's estimate repeats the same check per source.
    """
    if (path is None)==(text is None):raise ValueError('Pass exactly one of path or text.')
    if not source or not str(source).strip():raise ValueError('Name the source of the table (publication, database entry or measurement).')
    if path is not None:
        raw=Path(path).read_bytes();file_name=Path(path).name if file_name is None else file_name
        text=raw.decode('utf-8-sig')
    else:
        raw=text.encode('utf-8');file_name=file_name or ''
    provenance=MaterialProvenance(source=str(source).strip(),licence=licence,raw_sha256=hashlib.sha256(raw).hexdigest(),
                                  file_name=file_name,columns=kind,wavelength_unit=unit,
                                  imported=date.today().isoformat() if imported is None else imported)
    data=OpticalData.from_text(text,kind=kind,unit=unit,reference=provenance.source)
    opt=FitOptions.model_validate(options.model_dump() if isinstance(options,FitOptions) else options or {})
    if dt_s and not opt.dt_s:opt=opt.model_copy(update=dict(dt_s=float(dt_s)))
    fit=fit_material(data,name=name,color=color,options=opt,provenance=provenance)
    messages=[]
    if not fit.converged:
        messages.append(f'{name}: the passive fit did not reach tolerance {opt.tolerance:g} '
                        f'(normalized RMS {fit.report[opt.target]["normalized_rms"]:.3e} with {fit.report["pole_count"]} poles).')
    discretization=discretization_report(fit.material,opt.dt_s) if opt.dt_s else None
    if simulation_band_um is not None:
        message=fit_band_extrapolation(fit.material,simulation_band_um)
        if message:
            messages.append(message);warnings.warn(message,MaterialBandWarning,stacklevel=2)
    return MaterialImportResult(fit.material,fit.report,provenance,discretization,messages)


def fit_discrete_lorentz(material=None, *, sellmeier_coefficients=None,
                         epsilon_inf=1.0, dt_s, wavelength_range_um,
                         sample_count=1001, tolerance=1e-3, grid_spacing_m=None,
                         name='Discrete Lorentz material'):
    """Fit a passive material to the implemented trapezoidal time update.

    Supply either a scalar ``Material`` or Sellmeier pairs ``(B, C)`` in
    ``epsilon = epsilon_inf + sum(B*lambda_um**2/(lambda_um**2-C))``.
    ``C`` is in um squared. The returned ``MaterialFitResult`` retains the
    continuous target samples and reports the error after FP32 coefficient
    rounding. A timestep-specific fit must be rebuilt if its timestep changes.

    Transparent lossless inputs use deterministic Gauss--Newton refinement
    of the supplied pole strengths/rates, rounded to nine decimal multiplier
    digits. Lossy inputs use the existing constrained complex ADE fitter.
    ``grid_spacing_m`` optionally fits the normal-incidence 1D Yee phase;
    this is an effective-medium compensation, not an all-angle correction.
    """
    from .models import LorentzPole

    if (material is None) == (sellmeier_coefficients is None):
        raise ValueError('Supply exactly one Material or Sellmeier coefficient sequence.')
    if (isinstance(dt_s, bool) or not np.isfinite(dt_s) or dt_s <= 0
            or type(sample_count) is not int or not 3 <= sample_count <= 8192
            or isinstance(tolerance, bool) or not np.isfinite(tolerance) or not 0 < tolerance <= 1):
        raise ValueError('Use a positive timestep, 3..8192 samples and a finite tolerance in (0,1].')
    try:
        low, high = (float(v) for v in wavelength_range_um)
    except (TypeError, ValueError):
        raise ValueError('wavelength_range_um must contain two positive increasing wavelengths.') from None
    if not np.isfinite((low, high)).all() or not 0 < low < high:
        raise ValueError('wavelength_range_um must be finite, positive and increasing.')
    c0 = 299792458.0
    wavelengths = np.linspace(low, high, sample_count)
    frequency = c0 / (wavelengths * 1e-6)
    if np.max(frequency) * dt_s >= .5:
        raise ValueError('The entire fitting band must lie below timestep Nyquist.')
    if sellmeier_coefficients is not None:
        terms = np.asarray(sellmeier_coefficients, dtype=np.float64)
        if (terms.ndim != 2 or terms.shape[1] != 2 or not 1 <= terms.shape[0] <= 16
                or not np.isfinite(terms).all() or np.any(terms <= 0)
                or isinstance(epsilon_inf, bool) or not np.isfinite(epsilon_inf) or not 1 <= epsilon_inf <= 400):
            raise ValueError('Sellmeier input needs 1..16 positive finite (B,C) pairs and epsilon_inf in [1,400].')
        rates = [2 * np.pi * c0 / (np.asarray(math.sqrt(c), dtype=np.float64) * 1e-6)
                 for _, c in terms]
        original = Material(name=name, model='multipole', epsilon_inf=epsilon_inf,
            poles=[LorentzPole(resonance_rad_s=float(w), strength_rad_s_squared=float(b*w**2), damping_rad_s=0)
                   for (b, _), w in zip(terms, rates)])
        # Preserve the input formula's arithmetic for the transparent target.
        square = wavelengths**2
        with np.errstate(divide='ignore', invalid='ignore'):
            expected = epsilon_inf + sum(b * square / (square - c) for b, c in terms)
        expected = np.asarray(expected, dtype=np.complex128)
    else:
        if not isinstance(material, Material):
            raise ValueError('material must be a validated scalar Material.')
        original = Material.model_validate(material.model_dump())
        if original.model == 'tensor' or not original.oscillators:
            raise ValueError('Discrete Lorentz fitting requires an isotropic oscillator material.')
        expected = permittivity(original, frequency)
    if not np.isfinite(expected).all():
        raise ValueError('A material resonance is singular in the fitting band.')
    data = OpticalData(wavelength_um=wavelengths.tolist(), epsilon_real=expected.real.tolist(),
        epsilon_imag=expected.imag.tolist(), reference='Continuous target of the supplied material declaration')
    poles = original.oscillators
    lossless = (all(gamma == 0 and rate > 0 for rate, _, gamma in poles)
                and bool(np.all(expected.real > 0)))
    if grid_spacing_m is not None:
        if (isinstance(grid_spacing_m, bool) or not np.isfinite(grid_spacing_m) or grid_spacing_m <= 0
                or not lossless):
            raise ValueError('Yee phase compensation requires a positive spacing and a transparent lossless target.')
    if not lossless:
        result = fit_material(data, name=name, options=FitOptions(max_poles=len(poles),
            include_drude=any(w == 0 for w, _, _ in poles), target='ade', dt_s=dt_s,
            tolerance=tolerance, starts=2), provenance=original.provenance)
        result.report['input_kind'] = 'Lorentz declaration'
        result.report['stability_contract'] = 'passive strengths/damping, epsilon_inf >= 1 and Yee CFL bound'
        return result
    continuous_n = np.sqrt(expected.real)
    target_n = continuous_n.copy()
    angular = 2 * np.pi * c0 / (wavelengths * 1e-6)
    if grid_spacing_m is not None:
        phase = continuous_n * angular * grid_spacing_m / (2 * c0)
        if np.any(phase >= np.pi / 2):
            raise ValueError('The compensated band crosses the first spatial Yee Nyquist branch.')
        target_n = c0 * dt_s * np.sin(phase) / (grid_spacing_m * np.sin(angular * dt_s / 2))
    discrete_angular = 2 / dt_s * np.tan(angular * dt_s / 2)
    count = len(poles)

    def residual(values, free):
        constant = values[-1] if free else 1.0
        # Keep the scalar expression order fixed. Refitting ill-conditioned
        # transparent poles is sensitive to double-precision rounding.
        epsilon = constant + sum(values[2*i] * strength /
            ((values[2*i+1] * rate)*(values[2*i+1] * rate) - discrete_angular*discrete_angular)
            for i, (rate, strength, _) in enumerate(poles))
        if not np.isfinite(epsilon).all() or np.any(epsilon <= 0):
            raise ValueError('Refinement left the transparent passive fitting branch.')
        return np.sqrt(epsilon) - target_n

    def refine(free):
        values = np.ones(2*count + int(free))
        if free:
            values[-1] = original.epsilon_inf
        for iteration in range(80):
            current = residual(values, free)
            jacobian = np.empty((sample_count, values.size))
            for i in range(values.size):
                delta = np.zeros_like(values)
                delta[i] = 1e-7
                jacobian[:, i] = (residual(values + delta, free) - residual(values - delta, free)) / 2e-7
            step = np.linalg.lstsq(jacobian, -current, rcond=None)[0]
            values = values + step
            if np.max(np.abs(step)) < 1e-13:
                break
        return values, iteration + 1

    try:
        values, iterations = refine(True)
        if values[-1] < 1:
            values, iterations = refine(False)
            values = np.r_[values, 1.0]
        values = np.array([float(f'{v:.9f}') for v in values])
        if not np.isfinite(values).all() or np.any(values[:-1] <= 0) or not 1 <= values[-1] <= 400:
            raise ValueError('The refined oscillator declaration is not passive.')
        fitted = Material(name=name, model='multipole', epsilon_inf=float(values[-1]),
            poles=[LorentzPole(resonance_rad_s=float(values[2*i+1]*w),
                strength_rad_s_squared=float(values[2*i]*s), damping_rad_s=0)
                for i, (w, s, _) in enumerate(poles)],
            samples=data, fit_band_um=(low, high), fit_dt_s=dt_s, provenance=original.provenance)
    except (ValueError, np.linalg.LinAlgError) as error:
        if grid_spacing_m is not None:
            raise ValueError('The compensated lossless fit did not converge on a passive branch.') from error
        fitted_result = fit_material(data, name=name, options=FitOptions(max_poles=count,
            include_drude=False, target='ade', dt_s=dt_s, tolerance=tolerance), provenance=original.provenance)
        fitted_result.report['lossless_refinement_fallback'] = str(error)
        return fitted_result

    coefficients = []
    epsilon32 = np.full(wavelengths.shape, float(np.float32(fitted.epsilon_inf)))
    for rate, strength, gamma in fitted.oscillators:
        square = (rate * dt_s)**2
        d = 1 + .5*gamma*dt_s + .25*square
        a32, d32, k32 = (float(np.float32(v)) for v in (.5*square, d, strength*dt_s*dt_s/(4*d)))
        coefficients.append(dict(a=a32, inverse_d=float(np.float32(1/d)), k=k32))
        epsilon32 += (4*d32*k32/dt_s**2)/(2*a32/dt_s**2-discrete_angular**2)
    if not np.isfinite(epsilon32).all() or np.any(epsilon32 <= 0):
        raise ValueError('Rounded ADE coefficients are singular or opaque in the fitting band.')
    index32 = np.sqrt(epsilon32)
    error = float(np.max(np.abs(index32 - target_n)))
    report = dict(converged=error <= tolerance, target='ade', lossless=True,
        dt_s=dt_s, wavelength_range_um=[low, high], sample_count=sample_count,
        max_abs_n_error=error, max_abs_continuous_n_error=float(np.max(np.abs(index32-continuous_n))),
        tolerance=tolerance, iterations=iterations, parameter_multipliers=values.tolist(),
        fp32_coefficients=coefficients, data_sha256=data.fingerprint,
        input_kind='Sellmeier coefficients' if sellmeier_coefficients is not None else 'Lorentz declaration',
        grid_spacing_m=grid_spacing_m, spatial_compensation='normal-incidence 1D Yee phase' if grid_spacing_m else None,
        stability_contract='nonnegative strengths/damping, epsilon_inf >= 1, below temporal/spatial Nyquist and Yee CFL bound')
    return MaterialFitResult(fitted, report)
