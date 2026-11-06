import subprocess
import sys


def test_import_preserves_callers_torch_training_defaults():
    code = '''import torch
torch.set_default_dtype(torch.float32)
torch.set_grad_enabled(True)
import torchfdtd
assert torch.get_default_dtype() == torch.float32
assert torch.is_grad_enabled()
x = torch.tensor(3., requires_grad=True)
(x*x).backward()
assert x.grad.item() == 6
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
