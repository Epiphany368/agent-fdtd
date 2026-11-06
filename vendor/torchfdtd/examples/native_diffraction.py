"""Read native NPZ diffraction: python -m examples.native_diffraction result.npz MONITOR_ID --index 1.0"""
import argparse
from torchfdtd.radiation_io import load_native_radiation_plane
from torchfdtd.radiation import diffraction_orders


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('npz');parser.add_argument('monitor_id')
    parser.add_argument('--index',type=float,required=True,help='Known homogeneous lossless exterior index; monitor must be outside PML.')
    parser.add_argument('--frequency-index',type=int,default=0)
    args=parser.parse_args()
    plane,project=load_native_radiation_plane(args.npz,args.monitor_id,frequency_index=args.frequency_index)
    a='xyz'.index(plane.normal);transverse=((a+1)%3,(a+2)%3);region=project.region
    if any(region.boundaries.pair(d)[0].kind not in ('periodic','bloch') for d in transverse):
        raise ValueError('Use periodic/Bloch transverse axes and a full unit-cell plane.')
    result=diffraction_orders(plane,[(0,0)],period_um=tuple(region.actual_size[d] for d in transverse),
        bloch_wavevector_per_um=tuple(region.bloch_phase[d]/region.actual_size[d] for d in transverse),refractive_index=args.index)
    print('Raw directional power, not efficiency:',result.forward_power.tolist(),result.backward_power.tolist())
    print('Propagating:',result.propagating.tolist())


if __name__=='__main__':main()
