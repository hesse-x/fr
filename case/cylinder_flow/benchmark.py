"""Paired complete-step benchmark against the unchanged NACA NS physics."""
import argparse
import json
import hashlib
import os
from pathlib import Path
import resource
import statistics
import time

import torch
from cylinder import module
from native import NativeCylinder


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh',type=Path,default=Path(__file__).resolve().parent/'results/mesh/structured.msh')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--steps',type=int,default=10)
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if min(args.threads,args.steps,args.repeats)<1 or args.warmup<0:
        parser.error('positive threads, steps and repeats; nonnegative warmup required')
    torch.set_default_dtype(torch.float64); torch.set_num_threads(args.threads)
    optimized = NativeCylinder(args.mesh)
    reference = module.Airfoil(args.mesh,reynolds=100.)
    initial = optimized.initial_state()
    cfl = initial.new_tensor(.2)
    # Compare the complete native spatial operator before timing RK evolution.
    expected = reference.rhs(initial)
    actual = optimized.rhs(initial)
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
    steppers = {name:model.make_step_with_rhs('global')
                for name,model in [('reference',reference),('optimized',optimized)]}
    models = dict(reference=reference,optimized=optimized)
    def run(name, count):
        model = models[name]; advance = steppers[name]
        U = initial.clone(); F = model.rhs(U)
        before = resource.getrusage(resource.RUSAGE_SELF)
        start = time.perf_counter()
        for _ in range(count):
            U,dt,F = advance(U,cfl,F)
            if not model.valid(U):
                raise RuntimeError('nonphysical benchmark state')
            residual = model.residual_norm(F) if hasattr(model,'residual_norm') else float(model.s.norm(F).max())
        elapsed = time.perf_counter()-start
        after = resource.getrusage(resource.RUSAGE_SELF)
        return U,dict(seconds_per_step=elapsed/max(count,1),
                      minor_faults=after.ru_minflt-before.ru_minflt,
                      cpu_threads_used=((after.ru_utime-before.ru_utime)+(after.ru_stime-before.ru_stime))/max(elapsed,1e-30))
    for name in models:
        run(name,max(args.warmup,1))
    samples = {name:[] for name in models}
    for repeat in range(args.repeats):
        finals = {}
        for name in (list(models) if repeat%2==0 else list(reversed(models))):
            finals[name],sample = run(name,args.steps)
            samples[name].append(sample)
            print(json.dumps(dict(repeat=repeat,implementation=name,**sample)),flush=True)
        torch.testing.assert_close(finals['optimized'],finals['reference'],rtol=1e-10,atol=1e-11)
    medians = {name:statistics.median(s['seconds_per_step'] for s in values)
               for name,values in samples.items()}
    report = dict(cells=optimized.s.blocks[0]['ne'],ndof=optimized.s.ndof,
                  p=optimized.s.p,quadrature=optimized.s.fw.numel(),dtype='float64',
                  torch_version=torch.__version__,affinity=sorted(os.sched_getaffinity(0)),
                  mesh_sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
                  threads=args.threads,implementation='torch.ops.cylinder_flow',steps=args.steps,
                  warmup=args.warmup,repeats=args.repeats,seconds_per_step=medians,
                  speedup=medians['reference']/medians['optimized'],samples=samples,
                  includes='four new RHS evaluations, CFL, validity, residual; excludes I/O and compilation',
                  correctness='complete native RHS and evolved states agree with NACA NS reference')
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
