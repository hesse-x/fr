"""Equal-physical-time SSPRK(4,3) CFL screen from a common perturbed checkpoint."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from airfoil import Airfoil


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    base=Path(__file__).parent/'results'
    p.add_argument('--checkpoint',type=Path,default=base/'downstream40_global0/solution.pt')
    p.add_argument('--output',type=Path,default=base/'ssprk43_cfl_test')
    p.add_argument('--duration',type=float,default=.25)
    p.add_argument('--cfls',type=float,nargs='+',default=[3,4,6,8])
    args=p.parse_args()
    if args.duration<=0 or not math.isfinite(args.duration) or any(v<=0 or not math.isfinite(v) for v in args.cfls):
        p.error('finite positive duration and CFL values required')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR','/tmp/fr-naca-inductor')
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','2')
    args.output.mkdir(parents=True,exist_ok=True)
    ck=torch.load(args.checkpoint,weights_only=False);m=ck['stats']
    a=Airfoil(m['mesh'],m['p'],m['mach'],m['alpha_deg'],m['quadrature'])
    torch.manual_seed(714)
    initial=ck['U'].clone()+1e-8*torch.randn_like(ck['U'])
    initial_res=float(a.s.norm(a.rhs(initial)).max())
    rhs=torch.compile(a.rhs,fullgraph=True)
    rhs(initial)
    records=[];reference=None
    for cfl in sorted(args.cfls):
        U=initial.clone();elapsed=0.;steps=0;start=time.monotonic();valid=True;rows=[];dt_first=None
        while elapsed<args.duration:
            dt=min(float(a.local_dt(U,cfl).min()),args.duration-elapsed)
            if not math.isfinite(dt) or dt<=0:valid=False;break
            if dt_first is None:dt_first=dt
            U=a.ssprk43(U,torch.full((a.s.blocks[0]['ne'],1,1),dt),rhs)
            steps+=1;elapsed+=dt
            if not a.valid(U):valid=False;break
            if steps%10==0 or elapsed>=args.duration:
                residual=float(a.s.norm(rhs(U)).max())
                cl,cd,_=a.loads(U);rows.append([steps,elapsed,residual,cl,cd])
                if residual>max(10*initial_res,1.):valid=False;break
        result=dict(cfl=cfl,method='ssprk43',time_mode='global',completed=valid and elapsed>=args.duration,
                    valid=valid,steps=steps,physical_time_increment=elapsed,first_dt=dt_first,
                    runtime_seconds=time.monotonic()-start,initial_residual=initial_res,
                    perturbation_absolute_std=1e-8,seed=714)
        if valid:
            result.update(residual=float(a.s.norm(rhs(U)).max()),CL=a.loads(U)[0],CD_pressure=a.loads(U)[1])
            if reference is None:reference=U.clone()
            result['state_rms_difference_from_lowest_cfl']=a.s.norm(U-reference).tolist()
        with (args.output/f'cfl_{cfl:g}.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['iteration','physical_time_increment','residual','CL','CD_pressure']);w.writerows(rows)
        records.append(result)
        (args.output/'report.json').write_text(json.dumps(dict(checkpoint=str(args.checkpoint),duration=args.duration,cases=records),indent=2)+'\n')
        print(json.dumps(result),flush=True)
    print('Completed CFL screening; a finite test is not a long-time stability proof.',flush=True)


if __name__=='__main__':main()
