"""Tensor-product local grading with exact sampling and feed anchor planes."""
import math

import torch


def graded_edges(bounds, coarse_h, fine_h, fine_bounds, anchors=(), single_interval=None):
    lo, hi = bounds
    a, b = fine_bounds
    # One transition band on either side, so a 4:1 request uses 2:1 steps.
    breaks = [lo, hi, a, b, a-coarse_h, b+coarse_h, *anchors]
    breaks = sorted(set(float(x) for x in breaks if lo <= x <= hi))
    result = [lo]
    for left, right in zip(breaks[:-1], breaks[1:]):
        middle = (left+right)/2
        target = fine_h if a <= middle <= b else min(coarse_h, 2*fine_h) if a-coarse_h <= middle <= b+coarse_h else coarse_h
        count = max(1, math.ceil((right-left)/target-1e-10))
        if single_interval is not None and abs(left-single_interval[0]) < 1e-12 and abs(right-single_interval[1]) < 1e-12:
            count = 1
        result.extend(left+(right-left)*i/count for i in range(1, count+1))
    return torch.tensor(result, dtype=torch.float64)
