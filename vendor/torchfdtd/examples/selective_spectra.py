"""Two independent CUDA cases with flux-only and high-precision selected fields."""
from torchfdtd import FieldMonitor, SpectrumSettings, run_tensor_batch
from torchfdtd.models import demo_project


def main():
    projects=[]
    for radius in (.4,.5):
        p=demo_project('3d');p.structures[0].radius=radius
        p.region.backend='cuda';p.region.cuda_kernel='fused';p.region.cuda_monitor_kernel='fused'
        spectrum=SpectrumSettings(sampling='chebyshev',chebyshev_nodes='lobatto',frequency_points=17,apodization='none')
        p.monitors=[FieldMonitor(id='flux',name='Flux only',center=(1,0,0),size=(0,1,1),
            spectrum=spectrum,record_fields=(),record_poynting=()),
            FieldMonitor(id='fields',name='Selected fields',center=(1,0,0),size=(0,1,1),
                spectrum=spectrum,record_fields=('Ez','Hy'),record_poynting=(),record_flux=False,
                dft_precision='float64',downsample_xyz=(1,2,3))]
        projects.append(p)
    report=run_tensor_batch(projects,cohort_size=2);report.raise_for_errors()
    for item in report.items:
        result=item.load()
        print(item.id,result.frequency_fields[0]['flux'],result.frequency_fields[1]['components'])


if __name__=='__main__':main()
