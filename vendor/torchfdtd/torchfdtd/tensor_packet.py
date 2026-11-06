"""Lossless internal transport for fixed mixed-dtype solver tensors."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class TensorLayout:
    dtype: torch.dtype
    count: int
    entries: tuple

    @classmethod
    def from_tensors(cls, tensors):
        """Describe a packet without allocating or making contiguous copies."""
        values = tuple(tensors)
        if not values or any(not isinstance(t, torch.Tensor) or t.layout != torch.strided for t in values):
            raise ValueError('Expected a nonempty sequence of strided tensors.')
        if any(t.device != values[0].device for t in values):
            raise ValueError('Packet tensors must share a device.')
        mixed = any(t.dtype != values[0].dtype for t in values)
        alignment = max(t.element_size() for t in values)
        entries, offset = [], 0
        for value in values:
            if mixed:offset += (-offset) % alignment
            length = value.numel() * (value.element_size() if mixed else 1)
            entries.append((value.shape, value.dtype, offset, length))
            offset += length
        return cls(torch.uint8 if mixed else values[0].dtype, offset, tuple(entries))

    def unpack(self,payload):
        if (payload.dtype!=self.dtype or payload.ndim!=1 or payload.numel()!=self.count
                or not payload.is_contiguous() or payload.is_conj() or payload.is_neg()):
            raise ValueError('Packet does not match its tensor layout.')
        result=[]
        for shape,dtype,offset,length in self.entries:
            value=payload.narrow(0,offset,length)
            if value.dtype!=dtype:value=value.view(dtype)
            result.append(value.reshape(shape))
        return tuple(result)


def pack_tensors(tensors, *, out=None):
    """Pack detached logical values, preserving dtype, shape and precision.

    Homogeneous packets retain the existing typed concatenation. Mixed packets
    use aligned byte views, avoiding promotion of real gradients to complex or
    rounding integer values through floating point. Returned views share packet
    storage. Supplying ``out`` writes strided logical values directly into a
    reusable packet, without a temporary concatenation or contiguous copies.
    This is transport for a manual solver, not an autograd operation.
    """
    values=tuple(tensors)
    if not values or any(not isinstance(t,torch.Tensor) or t.layout!=torch.strided for t in values):
        raise ValueError('Expected a nonempty sequence of strided tensors.')
    if any(t.device!=values[0].device for t in values):raise ValueError('Packet tensors must share a device.')
    if out is not None:
        layout = TensorLayout.from_tensors(values)
        if not isinstance(out, torch.Tensor) or out.device != values[0].device or out.requires_grad:
            raise ValueError('Packet destination must be a detached tensor on the source device.')
        views = layout.unpack(out)
        # Repacking an aliased input could overwrite a later input. The caller
        # owns separate transfer slots, so reject even disjoint shared storage.
        pointer = out.untyped_storage().data_ptr()
        if any(value.numel() and value.untyped_storage().data_ptr() == pointer for value in values):
            raise ValueError('Packet destination must not share storage with an input.')
        offset = 0
        for value, target, (_, _, begin, length) in zip(values, views, layout.entries):
            if begin > offset:out[offset:begin].zero_()
            target.copy_(value.detach())
            offset = begin + length
        return out, layout
    mixed=any(t.dtype!=values[0].dtype for t in values)
    alignment=max(t.element_size() for t in values)
    padding=torch.zeros(alignment,dtype=torch.uint8,device=values[0].device) if mixed else None
    parts=[];entries=[];offset=0
    for original in values:
        value=original.detach().resolve_conj().resolve_neg().contiguous().reshape(-1)
        if mixed:
            pad=(-offset)%alignment
            if pad:parts.append(padding[:pad]);offset+=pad
            value=value.view(torch.uint8)
        length=value.numel()
        entries.append((original.shape,original.dtype,offset,length))
        parts.append(value);offset+=length
    payload=torch.cat(parts)
    return payload,TensorLayout(payload.dtype,payload.numel(),tuple(entries))
