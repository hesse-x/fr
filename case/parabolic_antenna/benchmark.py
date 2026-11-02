"""Measure native FP64 RK4 time; exclude build/setup time."""
import argparse
import json
import statistics
import time

import torch

from .solve import ParabolicAntennaCase


def measure(function, repeats):
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter()-start)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refinement', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if min(args.threads, args.repeats) < 1:
        parser.error('threads and repeats must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(17)
    case = ParabolicAntennaCase(refinement=args.refinement)
    state = torch.randn_like(case.initial).masked_fill(case.solid_nodes[..., None], 0.)
    dt = min(case.timestep(state, .6), .5/case.pml.sigma_max)
    seconds = measure(lambda: case.native.step(None, state, .7, dt), args.repeats)
    print(json.dumps(dict(cells=case.cells, p=case.p, threads=args.threads,
        repeats=args.repeats, native_step_seconds=seconds), indent=2))


if __name__ == '__main__':
    main()
