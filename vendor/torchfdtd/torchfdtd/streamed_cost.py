"""Checkpoint-aware duration extrapolation for measured streamed policies."""
from functools import lru_cache

from .differentiable import _split


@lru_cache(maxsize=1024)
def replay_blocks(blocks, checkpoints):
    """Exact forward replay count for the current binomial block schedule."""
    if blocks <= 1:return 0
    if checkpoints == 0:return blocks*(blocks-1)//2
    middle = _split(blocks, checkpoints)
    return middle+replay_blocks(blocks-middle, checkpoints-1)+replay_blocks(middle, checkpoints)


def predict_duration(short, long, *, steps, depth, checkpoints):
    """Separate startup, forward-block and transpose-block costs from two probes.

    A partial terminal block is conservatively counted as one full block. This
    is a timing model, not an optimality or runtime-bound guarantee.
    """
    blocks = lambda count: (count+depth-1)//depth
    target = blocks(steps)
    a, b = blocks(short['steps']), blocks(long['steps'])
    ra, rb, rt = (replay_blocks(n,checkpoints) for n in (a,b,target))
    if long['steps'] == steps:
        return dict(seconds=long['total'], method='measured full duration', blocks=target, replay_blocks=rt)
    if b <= a:
        return dict(seconds=long['total']*(2*target+rt)/(2*b+rb), method='work-count fallback',
                    blocks=target,replay_blocks=rt)
    forward = (long['forward']-short['forward'])/(b-a)
    remaining_a = short['backward']-ra*forward
    remaining_b = long['backward']-rb*forward
    transpose = (remaining_b-remaining_a)/(b-a)
    if forward < 0 or transpose < 0 or min(remaining_a,remaining_b) < 0:
        return dict(seconds=long['total']*(2*target+rt)/(2*b+rb), method='work-count fallback',
                    blocks=target,replay_blocks=rt)
    startup_forward = max(0.,short['forward']-a*forward)
    startup_backward = max(0.,remaining_a-a*transpose)
    outside = max(0.,short['total']-short['forward']-short['backward'],
                  long['total']-long['forward']-long['backward'])
    predicted = startup_forward+startup_backward+outside+(target+rt)*forward+target*transpose
    return dict(seconds=max(predicted,long['total']),method='two-duration replay cost',
                blocks=target,replay_blocks=rt,forward_block_seconds=forward,
                transpose_block_seconds=transpose,startup_forward_seconds=startup_forward,
                startup_backward_seconds=startup_backward,outside_solver_seconds=outside)
