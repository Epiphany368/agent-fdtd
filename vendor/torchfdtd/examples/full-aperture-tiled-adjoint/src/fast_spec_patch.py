# ---------------- spectral DFT blocks: observer index lists -> cached device index tensors (FF_FAST_SPEC, default on) ----------------
# SpectralObservation.accumulate / .transpose index the (32, 602112) sample block with a Python list of ~301k observer numbers
# per field family, and torch parses that list element by element on every call (4 list conversions per 32-step block).
# A100 probe 2026-09-25: 8.9 s of a 17.6 s tile forward loop; the Yee updates themselves took 8.3 s.
# The replacement gathers the same elements in the same order with a LongTensor built once per observation object, so every
# product and sum sees identical operands (bit-identical results, checked by test_fast_spec.py).
import torch
import torchfdtd.adjoint_spectrum as _asp
_ORIG_ACCUMULATE = _asp.SpectralObservation.accumulate; _ORIG_TRANSPOSE = _asp.SpectralObservation.transpose
def _spec_index(self, family):
    c = self.__dict__.get("_ff_index")
    if c is None:
        c = self._ff_index = {f: torch.tensor(ix, dtype=torch.long, device=self.device) for f, ix in self.groups.items() if ix}
    return c[family]
def _accumulate_idx(self, output, samples, start):
    stop = start + samples.shape[0]
    for family, indices in self.groups.items():
        if indices:
            ix = _spec_index(self, family)
            output[:, ix] += self.kernel(start, stop, family) @ samples[:, ix].to(self.complex_dtype)
def _transpose_idx(self, seed, start, stop):
    result = torch.empty((stop - start, len(self.components)), dtype=self.complex_dtype if self.complex_samples else self.dtype, device=self.device)
    for family, indices in self.groups.items():
        if indices:
            ix = _spec_index(self, family)
            values = self.kernel(start, stop, family).conj().T @ seed[:, ix]
            result[:, ix] = values if self.complex_samples else values.real
    return result
def enable(on=True):
    _asp.SpectralObservation.accumulate = _accumulate_idx if on else _ORIG_ACCUMULATE
    _asp.SpectralObservation.transpose = _transpose_idx if on else _ORIG_TRANSPOSE
