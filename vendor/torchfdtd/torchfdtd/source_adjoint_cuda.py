"""Source waveform VJPs composed with the existing fused Yee/CPML transpose.

The source system owns packed material/waveform gradient views and performs each
source-support reduction. This wrapper changes only the ordering around the two
existing transpose launches. No material or field-update kernel is replaced.
"""
import numpy as np
import torch


class SourceFusedAdjointCUDA:
    """First-order additive-source transpose on the current Torch CUDA stream.

    ``system.material_gradient_view`` returns the epsilon subview of the final
    carrier gradient. ``system.accumulate_waveform_gradient`` accumulates one
    temporal row directly into that carrier, without a waveform-sized temporary.
    """

    def __init__(self, system, packed_gradient, signal_bar):
        if not packed_gradient.is_cuda:
            raise ValueError('The fused source adjoint requires CUDA carrier gradients.')
        from .cuda_adjoint import FusedAdjointCUDA
        from .cuda_complex_adjoint import FusedComplexAdjointCUDA

        self.system = system
        self.packed_gradient = packed_gradient
        kernel = FusedComplexAdjointCUDA if system.grid.E.is_complex() else FusedAdjointCUDA
        self.kernel = kernel(system, system.material_gradient_view(packed_gradient),
                             signal_bar, direct_views=True)
        # The checkpoint scheduler updates this buffer for each online DFT block.
        self.signal_bar = self.kernel.signal_bar

    @property
    def e_bar(self):
        return self.kernel.e_bar

    @property
    def h_bar(self):
        return self.kernel.h_bar

    @property
    def psi_bars(self):
        return self.kernel.psi_bars

    @property
    def phase(self):
        return self.kernel.phase

    @torch.no_grad()
    def step(self, index, *, observation_index=None):
        kernel = self.kernel
        sample = index if observation_index is None else observation_index
        # Complex observations use the same indexed scatter as the existing
        # complex adjoint. Calling its step() here would scatter a second time.
        if self.e_bar.is_complex():
            seed = self.signal_bar[sample]
            for target, (positions, indices) in zip(
                    (self.e_bar, self.h_bar), self.system.observation_maps):
                if indices.numel():
                    target.reshape(-1).index_add_(0, indices, seed.index_select(0, positions))
        with kernel.cp.cuda.Device(kernel.device), kernel.stream():
            if kernel.observer is not None:
                fn, arrays, _, count = kernel.observer
                fn(((count + 127) // 128,), (128,), (*arrays, np.int32(sample)))
            self.system.accumulate_waveform_gradient(
                'H', self.h_bar, self.packed_gradient, index)
            fn, arrays, _ = kernel.launches[True, kernel.phase]
            fn(((kernel.count + 255) // 256,), (256,), arrays)
            self.system.accumulate_waveform_gradient(
                'E', self.e_bar, self.packed_gradient, index)
            fn, arrays, _ = kernel.launches[False, kernel.phase]
            fn(((kernel.count + 255) // 256,), (256,), arrays)
        kernel.phase = 1 - kernel.phase
