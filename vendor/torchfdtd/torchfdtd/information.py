"""Real Gaussian target information with explicitly supplied joint moments.

This module contains no optical proxy, color prior, exposure calibration or
training data. Cov(X,Z) and Cov(Z,Z) can retain target uncertainty beyond X.
"""
from dataclasses import dataclass
import math
import torch


@dataclass
class GaussianTargetResult:
    information_bits: torch.Tensor
    posterior_target_covariance: torch.Tensor
    target_given_scene_covariance: torch.Tensor
    decoder: torch.Tensor
    recovered_trace_fraction: torch.Tensor
    recovered_whitened_fraction: torch.Tensor


def shot_read_covariance(mean_electrons,read_noise_e_rms):
    """Independent shot + read variance. Gradients through the mean are kept."""
    if not isinstance(mean_electrons,torch.Tensor) or mean_electrons.dtype not in (torch.float32,torch.float64) or mean_electrons.ndim<1:
        raise ValueError('Mean electrons must be a real floating tensor with a channel axis.')
    if mean_electrons.shape[-1]<1:raise ValueError('Mean electrons need at least one channel.')
    if torch.as_tensor(read_noise_e_rms).is_complex():raise ValueError('Read noise must be real.')
    read=torch.as_tensor(read_noise_e_rms,device=mean_electrons.device,dtype=mean_electrons.dtype)
    if read.ndim!=0 or not bool(torch.isfinite(read)) or bool(read<0):
        raise ValueError('Read noise must be a finite nonnegative scalar RMS.')
    if not bool(torch.isfinite(mean_electrons).all()) or bool((mean_electrons<0).any()):
        raise ValueError('Mean electrons must be finite and nonnegative.')
    return torch.diag_embed(mean_electrons+read.square())


def _symmetric(value,name):
    tolerance=32*value.shape[-1]*torch.finfo(value.dtype).eps
    scale=value.detach().abs().amax(dim=(-2,-1)).clamp_min(torch.finfo(value.dtype).tiny)
    error=(value-value.mT).detach().abs().amax(dim=(-2,-1))
    if bool((error>tolerance*scale).any()):raise ValueError(f'{name} must be symmetric.')
    return (value+value.mT)/2


def _cholesky(value,name):
    factor,info=torch.linalg.cholesky_ex(value)
    if bool((info!=0).any()) or not bool(torch.isfinite(factor).all()):
        raise ValueError(f'{name} must be numerically positive definite. No jitter is added.')
    return factor


def gaussian_target_information(response,scene_covariance,scene_target_covariance,target_covariance,noise_covariance):
    """I(Y;Z) for Y=A X+noise and supplied real joint moments of X,Z.

    Matrix shapes are A=(M,K), XX=(K,K), XZ=(K,D), ZZ=(D,D), N=(M,M).
    Leading batch axes must be identical or absent for shared matrices.
    XX, ZZ and N must be positive definite, with a PSD joint X/Z covariance.
    An 81-channel singular spectral prior is not admitted through this API.
    The decoder acts on centered measurements. Add the target mean separately.
    """
    values=(response,scene_covariance,scene_target_covariance,target_covariance,noise_covariance)
    if any(not isinstance(v,torch.Tensor) or v.ndim<2 or v.dtype not in (torch.float32,torch.float64) for v in values):
        raise ValueError('Inputs must be real float32/float64 matrices or matrix batches.')
    if any(v.device!=response.device or v.dtype!=response.dtype for v in values):
        raise ValueError('All inputs must have identical dtype and device.')
    if any(any(d<1 for d in v.shape) or not bool(torch.isfinite(v).all()) for v in values):
        raise ValueError('Input dimensions must be nonempty and values finite.')
    m,k=response.shape[-2:];d=target_covariance.shape[-1]
    expected=((m,k),(k,k),(k,d),(d,d),(m,m))
    if any(tuple(v.shape[-2:])!=shape for v,shape in zip(values,expected)):
        raise ValueError('Channel, scene and target matrix dimensions do not match.')
    batches={tuple(v.shape[:-2]) for v in values if v.ndim>2}
    if len(batches)>1:raise ValueError('Batch axes must be identical or absent. Partial broadcasting is not allowed.')
    batch=next(iter(batches),())
    a,xx,xz,zz,noise=[v.expand(*batch,*v.shape[-2:]) for v in values]
    xx,zz,noise=(_symmetric(v,n) for v,n in ((xx,'Scene covariance'),(zz,'Target covariance'),(noise,'Noise covariance')))
    lx=_cholesky(xx,'Scene covariance')
    lz=_cholesky(zz,'Target covariance')
    ln=_cholesky(noise,'Noise covariance')
    c=torch.linalg.solve_triangular(lx,xz,upper=False)
    residual=zz-c.mT@c
    residual=(residual+residual.mT)/2
    with torch.no_grad():
        tolerance=128*max(k,d)*torch.finfo(response.dtype).eps*zz.abs().amax(dim=(-2,-1))
        if bool((torch.linalg.eigvalsh(residual)[...,0] < -tolerance).any()):
            raise ValueError('Joint scene/target moments are not positive semidefinite.')
    w=torch.linalg.solve_triangular(ln,a@lx,upper=False)
    identity=torch.eye(k,device=a.device,dtype=a.dtype)
    lp=_cholesky(identity+w.mT@w,'Whitened posterior precision')
    solved=torch.cholesky_solve(c,lp)
    posterior=residual+c.mT@solved
    posterior=(posterior+posterior.mT)/2
    lpost=_cholesky(posterior,'Posterior target covariance')
    logdet=lambda l:2*torch.log(torch.diagonal(l,dim1=-2,dim2=-1)).sum(-1)
    bits=(logdet(lz)-logdet(lpost))/(2*math.log(2))
    decoder_white=solved.mT@w.mT
    decoder=torch.linalg.solve_triangular(ln.mT,decoder_white.mT,upper=True).mT
    trace=lambda v:torch.diagonal(v,dim1=-2,dim2=-1).sum(-1)
    return GaussianTargetResult(bits,posterior,residual,decoder,
                                1-trace(posterior)/trace(zz),1-trace(torch.cholesky_solve(posterior,lz))/d)
