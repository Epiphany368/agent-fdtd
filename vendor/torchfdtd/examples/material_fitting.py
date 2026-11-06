"""Fit user optical samples or independently authored Lorentz data.

python examples/material_fitting.py --csv measurement.csv --unit nm
"""
import argparse
from pathlib import Path

import numpy as np

from torchfdtd import OpticalData, FitOptions, fit_material


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path)
    parser.add_argument('--unit',choices=['nm','um','m'],default='um')
    parser.add_argument('--kind',choices=['nk','epsilon'],default='nk')
    parser.add_argument('--max-poles',type=int,default=6)
    parser.add_argument('--tolerance',type=float,default=1e-3)
    parser.add_argument('--output',type=Path,default=Path('results/material-fit.json'))
    args=parser.parse_args()
    if args.csv:
        data=OpticalData.from_csv(args.csv,unit=args.unit,kind=args.kind)
    else:
        wavelength=np.linspace(.7,2.2,121)
        omega=2*np.pi*299792458/(wavelength*1e-6)
        epsilon=2.3+4.913e30/(1.7e15**2-omega**2-1j*2e14*omega)
        index=np.sqrt(epsilon)
        data=OpticalData.from_nk(wavelength,index.real,index.imag,
                                reference='Authored Lorentz verification data, not a measured substance')
    result=fit_material(data,name='Imported optical material',options=FitOptions(
        max_poles=args.max_poles,tolerance=args.tolerance))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.save(args.output)
    print(f"Tolerance met: {result.converged}, poles: {result.report['pole_count']}, "
          f"normalized RMS: {result.report['analytic']['normalized_rms']:.6g}")
    print(f'Fit and original data: {args.output}')
    result.require_tolerance()


if __name__=='__main__':main()
