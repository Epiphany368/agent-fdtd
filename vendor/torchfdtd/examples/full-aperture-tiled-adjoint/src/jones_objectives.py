"""Flux-normalized vector angular spectrum for existing calibrated full Jones pupil.
T already includes sqrt(n_out/n_in), via the original homogeneous-substrate reference.
Positive-z sign +1 is measured by the independent two-plane FDTD calibration.
"""
import os,math
from pathlib import Path
import numpy as np
import torch
_CALL=0

def optical_objective(T,*,kind,lam_nm,pitch=.29,focal_um=None,out=None):
    global _CALL
    sample=next(iter(T.values()));nl,m,_=sample.shape;device=sample.device
    aperture_um=float(os.environ.get('FF_APERTURE_UM','208'))
    focal_um=float(os.environ.get('FF_FOCAL_UM',str(aperture_um/(2*.3)))) if focal_um is None else focal_um
    n=2*m;o=(n-m)//2
    coord=(torch.arange(m,device=device,dtype=torch.float64)-m/2+.5)*pitch
    pupil=(coord[:,None]**2+coord[None,:]**2<=(aperture_um/2)**2).to(sample.real.dtype)
    fr=torch.fft.fftfreq(n,d=pitch,device=device,dtype=torch.float64)
    fx,fy=torch.meshgrid(fr,fr,indexing='ij')
    c=(torch.arange(n,device=device,dtype=torch.float64)-n/2+.5)*pitch
    x,y=torch.meshgrid(c,c,indexing='ij')
    radius=float(os.environ.get('FF_DETECTOR_RADIUS_UM','.8'))
    if kind=='focus':target=(x*x+y*y<=radius**2).double()
    elif kind=='splitter':
        shift=aperture_um*.12
        regions=[((x-a)**2+(y-b)**2<=radius**2).double() for a,b in [(shift,0),(-shift,0),(0,shift),(0,-shift)]]
        target=sum(regions)
    elif kind=='hologram':
        shift=aperture_um*.09
        r=torch.sqrt(x*x+y*y);angle=torch.atan2(y,x)
        ring_radius=aperture_um*.07
        radial_sigma=max(.4,aperture_um*.007)
        target=torch.exp(-(r-ring_radius).square()/(2*radial_sigma**2))*(.12+.88*torch.cos(4*angle).square())
        target=target+.08*torch.exp(-(r/(ring_radius*.35)).square()/2)
        target=target/target.sum()
    else:raise ValueError(kind)
    efficiencies=[];images=[];throughputs=[]
    for pol in [('xx','yx'),('xy','yy')]:
        for li,lam in enumerate(lam_nm):
            lam=lam*1e-3;sx=lam*fx;sy=lam*fy;s2=sx*sx+sy*sy
            # Cell sampling supports design NA 0.3, with conservative propagating cutoff 0.65.
            keep=s2<.65**2;sz=torch.sqrt((1-s2).clamp_min(1e-9))
            phase=torch.exp(2j*math.pi*focal_um/lam*sz)*keep
            ex=torch.zeros((n,n),device=device,dtype=torch.complex128);ey=torch.zeros_like(ex)
            ex[o:o+m,o:o+m]=T[pol[0]][li]*pupil;ey[o:o+m,o:o+m]=T[pol[1]][li]*pupil
            ex=torch.fft.fft2(ex)*phase;ey=torch.fft.fft2(ey)*phase
            ez=-(sx*ex+sy*ey)/sz
            hx=sy*ez-sz*ey;hy=sz*ex-sx*ez
            ex=torch.fft.ifft2(ex);ey=torch.fft.ifft2(ey)
            hx=torch.fft.ifft2(hx);hy=torch.fft.ifft2(hy)
            flux=(ex*hy.conj()-ey*hx.conj()).real
            area=math.pi*(aperture_um/2)**2
            throughput=flux.sum()*pitch**2/area
            if kind=='hologram':
                positive=flux.clamp_min(0);image=positive/(positive.sum()+1e-30)
                score=(image*target).sum()/torch.sqrt(image.square().sum()*target.square().sum()+1e-30)*throughput
            else:
                score=(flux*target).sum()*pitch**2/area
                if kind=='splitter':
                    powers=torch.stack([(flux*q).sum()*pitch**2/area for q in regions])
                    score=score-.2*(powers-powers.mean()).square().sum()/(powers.sum()+1e-12)
            efficiencies.append(score);throughputs.append(throughput);images.append(flux.detach())
    values=torch.stack(efficiencies).reshape(2,nl)
    # Common focal plane and equal wavelength/polarization weighting.
    value=values.mean()
    _CALL+=1
    if out is not None:
        root=Path(out)/'optical';root.mkdir(exist_ok=True)
        np.savez_compressed(root/'latest_tmp.npz',jones=np.stack([T[k].detach().cpu().numpy() for k in ['xx','yx','xy','yy']]),image=torch.stack(images).cpu().numpy().astype(np.float32),target=target.cpu().numpy().astype(np.float32),values=values.detach().cpu().numpy(),throughput=torch.stack(throughputs).detach().cpu().numpy().reshape(2,nl),wavelength_nm=lam_nm,kind=kind,call=_CALL,propagation_sign=1,aperture_um=aperture_um,focal_um=focal_um,pupil_pitch_um=pitch,fourier_convention='exp(+2 pi i f t)',detector_radius_um=radius)
        os.replace(root/'latest_tmp.npz',root/'latest.npz')
        if _CALL%20==0:
            half=min(n//2,int(math.ceil((aperture_um*.15+4)/pitch)));c0=n//2;sl=slice(c0-half,c0+half)
            np.savez_compressed(root/f'call_{_CALL:05d}.npz',jones=np.stack([T[k].detach().cpu().numpy() for k in ['xx','yx','xy','yy']]),image=torch.stack(images)[:,sl,sl].cpu().numpy().astype(np.float32),target=target[sl,sl].cpu().numpy().astype(np.float32),values=values.detach().cpu().numpy(),wavelength_nm=lam_nm,kind=kind,pupil_pitch_um=pitch,propagation_sign=1)
    return value

