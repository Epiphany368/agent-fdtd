"""Explicit spectral quadrature to equivalent-electron Gaussian objectives."""
from dataclasses import dataclass
import torch
from .information import gaussian_target_information,shot_read_covariance


def _real(value,like,name):
    if torch.as_tensor(value).is_complex():raise ValueError(f'{name} must be real.')
    result=torch.as_tensor(value,device=like.device,dtype=like.dtype)
    if not bool(torch.isfinite(result).all()):raise ValueError(f'{name} must be finite.')
    return result


def _grid(value,like,name):
    result=_real(value,like,name)
    if result.ndim!=1 or result.numel()<2 or result.requires_grad or bool((result<=0).any()) or not bool((result[1:]>result[:-1]).all()):
        raise ValueError(f'{name} must be a fixed, positive, strictly increasing grid with at least two samples.')
    return result


def spectral_interpolate(values,sample_wavelengths_nm,query_wavelengths_nm):
    """Linear interpolation on the final axis, without extrapolation."""
    if not isinstance(values,torch.Tensor) or values.dtype not in (torch.float32,torch.float64) or values.ndim<1:
        raise ValueError('Values must be a real floating tensor with a spectral axis.')
    if not bool(torch.isfinite(values).all()):raise ValueError('Values must be finite.')
    samples=_grid(sample_wavelengths_nm,values,'Sample wavelengths')
    query=_real(query_wavelengths_nm,values,'Query wavelengths')
    if query.ndim!=1 or query.numel()<1 or query.requires_grad or values.shape[-1]!=samples.numel():
        raise ValueError('Fixed query wavelengths and values must match the spectral contract.')
    if bool((query<samples[0]).any() | (query>samples[-1]).any()):raise ValueError('Spectral extrapolation is forbidden.')
    upper=torch.searchsorted(samples,query).clamp(1,samples.numel()-1);lower=upper-1
    fraction=(query-samples[lower])/(samples[upper]-samples[lower])
    return values[...,lower]*(1-fraction)+values[...,upper]*fraction


@dataclass
class SpectralElectronModel:
    measurement_matrix: torch.Tensor
    mean_electrons: torch.Tensor
    spectral_electron_weight: torch.Tensor


def spectral_electron_model(response,wavelengths_nm,qe,mean_spectrum,scene_basis,*,
                            calibration,quadrature_nm=None):
    """Build a linear electron model on an explicitly shared spectral grid.

    response is (channels, wavelengths), scene_basis is (latent, wavelengths).
    The mean spectrum is spectral irradiance per metre. Calibration contains
    the fixed area/exposure/normalization factor. Supply quadrature_nm to retain
    a parent integration convention at passband cuts. Otherwise trapezoid
    weights are computed on the supplied grid. Nothing normalizes router power.
    """
    if not isinstance(response,torch.Tensor) or response.dtype not in (torch.float32,torch.float64) or response.ndim!=2 or response.shape[0]<1:
        raise ValueError('Response must be a real (channel, wavelength) tensor.')
    wavelength=_grid(wavelengths_nm,response,'Wavelengths')
    qe=_real(qe,response,'QE');mean=_real(mean_spectrum,response,'Mean spectrum')
    basis=_real(scene_basis,response,'Scene basis');scale=_real(calibration,response,'Calibration')
    if response.shape[-1]!=wavelength.numel() or qe.shape!=wavelength.shape or mean.shape!=wavelength.shape:
        raise ValueError('Response, QE and mean spectrum must share the wavelength grid.')
    if basis.ndim!=2 or basis.shape[0]<1 or basis.shape[1]!=wavelength.numel():
        raise ValueError('Scene basis must have latent, wavelength axes.')
    if scale.ndim!=0 or scale.requires_grad or bool(scale<=0):raise ValueError('Calibration must be a fixed positive scalar.')
    if not bool(torch.isfinite(response).all()) or bool((response<0).any() | (qe<0).any() | (mean<=0).any()):
        raise ValueError('Response/QE must be nonnegative and mean spectrum positive and finite.')
    if quadrature_nm is None:
        intervals=wavelength[1:]-wavelength[:-1]
        quadrature=torch.cat((intervals[:1]/2,(intervals[:-1]+intervals[1:])/2,intervals[-1:]/2))
    else:quadrature=_real(quadrature_nm,response,'Quadrature')
    if quadrature.shape!=wavelength.shape or quadrature.requires_grad or bool((quadrature<=0).any()):
        raise ValueError('Quadrature must be fixed positive weights on the wavelength grid.')
    photon_factor=(wavelength*1e-9)/(6.62607015e-34*299792458.)
    weight=mean*(quadrature*1e-9)*photon_factor*qe*scale
    operator=response*weight
    if not bool(torch.isfinite(operator).all()):raise ValueError('Electron conversion overflowed the selected precision.')
    return SpectralElectronModel(operator@basis.T,operator.sum(-1),weight)


@dataclass
class ExposureInformation:
    weighted_bits_per_pixel: torch.Tensor
    bits_per_pixel: torch.Tensor
    mean_electrons: torch.Tensor


def exposure_target_information(model,scene_covariance,cross_covariance,target_covariance,*,
                                exposure_scales,probabilities,read_noise_e_rms,raw_pixels):
    """Couple fixed exposures through one design and mean-dependent shot noise."""
    matrix=model.measurement_matrix
    scales=_real(exposure_scales,matrix,'Exposure scales');weights=_real(probabilities,matrix,'Probabilities')
    if scales.ndim!=1 or scales.numel()<1 or weights.shape!=scales.shape or scales.requires_grad or weights.requires_grad:
        raise ValueError('Exposure scales and probabilities must be fixed equal-length vectors.')
    if bool((scales<=0).any() | (weights<0).any()) or not torch.isclose(weights.sum(),weights.new_tensor(1.),rtol=0,atol=32*torch.finfo(weights.dtype).eps):
        raise ValueError('Exposure scales must be positive and probabilities nonnegative and sum to one.')
    if isinstance(raw_pixels,bool) or not isinstance(raw_pixels,int) or raw_pixels<1:raise ValueError('raw_pixels must be a positive integer.')
    means=scales[:,None]*model.mean_electrons
    noise=shot_read_covariance(means,read_noise_e_rms)
    result=gaussian_target_information(scales[:,None,None]*matrix,scene_covariance,cross_covariance,target_covariance,noise)
    bits=result.information_bits/raw_pixels
    return ExposureInformation((bits*weights).sum(),bits,means)
