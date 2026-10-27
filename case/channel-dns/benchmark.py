"""Measure the retained complete native step, including means and validity/CFL."""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from channel import Channel
from grouped_fused import GroupedFused
from native_step import NativeLocalRKStepper, NativeMeanAccumulator
from runtime import configure_cpu_memory_reuse


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', type=int, nargs=3, default=[64, 24, 32])
    parser.add_argument('--mach', type=float, default=.3)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(*args.cells, args.threads, args.steps, args.repeats) < 1 or args.warmup < 0:
        parser.error('positive grid, threads, steps and repeats required')
    available = sorted(os.sched_getaffinity(0))
    pcores = [cpu for cpu in range(0, 16, 2) if cpu in available]
    if args.threads <= len(pcores):
        os.sched_setaffinity(0, pcores[:args.threads])
    torch.set_num_threads(args.threads)
    configure_cpu_memory_reuse(True)
    model = Channel(tuple(args.cells), re_tau=178.12, mach=args.mach, forcing=0.,
                    lengths=(4*math.pi, 4*math.pi/3), bulk_velocity=15.67873098359455)
    state = model.initial(.5).movedim(-1, 0).contiguous().movedim(0, -1)
    plan = GroupedFused(model)
    native = NativeLocalRKStepper(model, plan)
    integral = NativeMeanAccumulator(model).zeros()
    valid, dt = native.scan(state, 1.2)
    assert valid
    totals = {}
    def wrap(name, function):
        def timed(*arguments):
            start = time.perf_counter()
            result = function(*arguments)
            totals[name] = totals.get(name, 0.)+time.perf_counter()-start
            return result
        return timed
    plan.library.grouped_fused_rk = wrap('rhs_and_local_rk', plan.library.grouped_fused_rk)
    for name in ('flux_force', 'flux_rk_finalize', 'flux_rk_finish'):
        setattr(native.library, name, wrap(name, getattr(native.library, name)))
    def step():
        nonlocal state, dt
        state, valid, dt = native.advance_finish(state, 0., dt, 1.2, integral)
        if not valid:
            raise RuntimeError('nonphysical benchmark state')
    for _ in range(args.warmup):
        step()
    totals.clear()
    samples = []
    for repeat in range(args.repeats):
        before = resource.getrusage(resource.RUSAGE_SELF)
        start = time.perf_counter()
        for _ in range(args.steps):
            step()
        elapsed = time.perf_counter()-start
        after = resource.getrusage(resource.RUSAGE_SELF)
        sample = dict(seconds_per_step=elapsed/args.steps,
                      minor_faults=after.ru_minflt-before.ru_minflt,
                      cpu_threads_used=((after.ru_utime-before.ru_utime)+(after.ru_stime-before.ru_stime))/elapsed)
        samples.append(sample)
        print(json.dumps(dict(repeat=repeat, **sample)), flush=True)
    result = dict(cells=args.cells, mach=args.mach, threads=args.threads,
                  affinity=sorted(os.sched_getaffinity(0)), includes_mean=True,
                  seconds_per_step=statistics.median(s['seconds_per_step'] for s in samples),
                  phases={key:value/(args.steps*args.repeats) for key,value in totals.items()},
                  samples=samples, implementation=plan.metadata)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
