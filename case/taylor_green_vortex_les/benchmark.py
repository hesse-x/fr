"""Measure full native steps including CFL scans, without diagnostic/output costs."""
import argparse
import gc
import json
from pathlib import Path
import platform
import statistics
import time

import torch
from .solve import TaylorGreenVortexLES
from fr.native.cartesian_ns.runtime import configure_cpu_memory_reuse


@torch.no_grad()
def measure(n, sgs_model, steps, cfl):
    model = TaylorGreenVortexLES((n,)*3, sgs_model=sgs_model)
    state = model.initial()
    stepper = model.make_stepper('native-p3')
    valid, dt = stepper.scan(state, cfl)
    if not valid:
        raise RuntimeError('invalid initial state')
    # Warm compiled code and buffers before measuring; retain initial state.
    for _ in range(3):
        _, valid, _ = stepper.advance_finish(state, 0., dt, cfl)
        if not valid:
            raise RuntimeError('invalid warmup')
    scans = []
    for _ in range(steps):
        started = time.perf_counter()
        stepper.scan(state, cfl)
        scans.append(time.perf_counter()-started)
    durations, t = [], 0.
    for _ in range(steps):
        started = time.perf_counter()
        state, valid, next_dt = stepper.advance_finish(state, t, dt, cfl)
        durations.append(time.perf_counter()-started)
        if not valid:
            raise RuntimeError('invalid measured step')
        t, dt = t+dt, next_dt
    return dict(cells=[n]*3, p=3, sgs=sgs_model, cfl=cfl, steps=steps,
                scan_seconds_median=statistics.median(scans),
                full_step_seconds_mean=statistics.mean(durations),
                full_step_seconds_median=statistics.median(durations),
                physical_time_advanced=t, mean_dt=t/steps,
                wall_seconds_per_physical_time=sum(durations)/t)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--cfl', type=float, default=.3)
    parser.add_argument('--output', type=Path, default=Path('results/taylor_green_vortex_les/benchmark.json'))
    args = parser.parse_args()
    if args.threads < 1 or args.steps < 1:
        parser.error('positive threads and steps required')
    torch.set_num_threads(args.threads)
    allocator = configure_cpu_memory_reuse()
    runs = []
    for n, sgs in ((32, 'none'), (32, 'vreman'), (64, 'none')):
        row = measure(n, sgs, args.steps, args.cfl)
        runs.append(row)
        print(json.dumps(row), flush=True)
        gc.collect()
    result = dict(threads=args.threads, dtype='float64', backend='native-p3',
                  platform=platform.platform(), torch_version=torch.__version__,
                  cpu_model=next((line.split(':', 1)[1].strip()
                      for line in Path('/proc/cpuinfo').read_text().splitlines()
                      if line.startswith('model name')), 'unknown'),
                  allocator=allocator, warmup_steps=3, diagnostics=False, runs=runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
