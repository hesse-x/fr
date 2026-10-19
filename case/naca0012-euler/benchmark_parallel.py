"""Benchmark single-case CPU threads in isolated processes; exclude compilation."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    base=Path(__file__).parent/'results'
    p.add_argument('--checkpoint',type=Path,default=base/'coarse40_cfl8/solution.pt')
    p.add_argument('--output',type=Path,default=base/'parallel_benchmark')
    p.add_argument('--threads',type=int,nargs='+',default=[1,2,4,8])
    p.add_argument('--steps',type=int,default=40)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--worker',type=int)
    a=p.parse_args()
    if a.steps<1 or a.repeats<1 or any(n<1 for n in a.threads) or (a.worker is not None and a.worker<1):
        p.error('positive step, repeat and thread counts required')
    a.output.mkdir(parents=True,exist_ok=True)
    if a.worker is None:
        reports=[]
        for n in a.threads:
            subprocess.run([sys.executable,__file__,'--worker',str(n),'--checkpoint',str(a.checkpoint),
                            '--output',str(a.output),'--steps',str(a.steps),'--repeats',str(a.repeats)],check=True)
            reports.append(json.loads((a.output/f'threads_{n}.json').read_text()))
        (a.output/'report.json').write_text(json.dumps(reports,indent=2)+'\n')
        return
    import torch
    from airfoil import Airfoil
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(a.worker)
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR','/tmp/fr-naca-inductor')
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','2')
    ck=torch.load(a.checkpoint,weights_only=False);m=ck['stats']
    air=Airfoil(m['mesh'],m['p'],m['mach'],m['alpha_deg'],m['quadrature'])
    with torch.no_grad():
        initial=ck['U'];rhs=torch.compile(air.rhs,fullgraph=True)
        advance=air.make_step(compile=True)
        cfl=initial.new_tensor(8.)
        for _ in range(3):
            rhs(initial)
            advance(initial,cfl)
        rhs_times=[];step_times=[]
        for repeat in range(a.repeats):
            start=time.perf_counter()
            for _ in range(a.steps*4):rhs(initial)
            rhs_times.append((time.perf_counter()-start)/(a.steps*4))
            U=initial.clone();physical=0.;cl=air.loads(U)[0]
            start=time.perf_counter()
            for _ in range(a.steps):
                U,dt=advance(U,cfl)
                physical+=float(dt.min())
                cl=air.loads(U)[0]
            step_times.append((time.perf_counter()-start)/a.steps)
        result=dict(threads=a.worker,steps=a.steps,repeats=a.repeats,compiled_full_step=True,rhs_seconds=rhs_times,
                    step_seconds=step_times,rhs_median=statistics.median(rhs_times),
                    step_median=statistics.median(step_times),valid=air.valid(U),CL=cl,
                    residual=float(air.s.norm(rhs(U)).max()),physical_time_increment=physical,
                    torch_version=torch.__version__,checkpoint=str(a.checkpoint))
        torch.save(U,a.output/f'threads_{a.worker}.pt')
        (a.output/f'threads_{a.worker}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
