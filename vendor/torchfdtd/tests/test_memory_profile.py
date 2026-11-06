import pytest
import torch
from torchfdtd import profile_memory_transfers


def test_file_probe_preserves_existing_files_and_labels_cache(tmp_path):
    keep=tmp_path/'keep.txt';keep.write_text('retain')
    report=profile_memory_transfers(device=None,storage_directory=tmp_path,file_bytes=8193,repeats=1)
    assert report['storage']['file_bytes']==8193
    assert report['storage']['sustained_nvme_bandwidth_measured'] is False
    assert report['storage']['gds'] is False
    assert list(tmp_path.iterdir())==[keep]
    assert keep.read_text()=='retain'


@pytest.mark.parametrize('kwargs',[{'copy_bytes':0},{'repeats':0},{'repeats':True},{'storage_directory':'.','file_bytes':-1}])
def test_probe_rejects_invalid_or_unbounded_requests(kwargs):
    with pytest.raises(ValueError):profile_memory_transfers(device=None,**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_pinned_copy_probe_reports_positive_measured_times():
    report=profile_memory_transfers(copy_bytes=1024**2,repeats=2)
    assert report['gpu']['h2d']['median_seconds']>0
    assert report['gpu']['d2h']['median_seconds']>0
    assert report['gpu']['concurrent_copy_compute_measured'] is False
