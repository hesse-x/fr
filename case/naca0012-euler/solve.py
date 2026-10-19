"""Explicit steady Euler run with auditable load/residual convergence."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import signal
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from airfoil import Airfoil
from diagnostics import Diagnostics


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    base=Path(__file__).parent/'results'
    p.add_argument('--mesh',type=Path,default=base/'mesh/moderate.msh')
    p.add_argument('--output',type=Path,default=base/'alpha_0')
    p.add_argument('--alpha',type=float,default=0.)
    p.add_argument('--mach',type=float,default=.3)
    p.add_argument('--p',type=int,default=3)
    p.add_argument('--quadrature',type=int,default=6,help='Gauss points per direction: tri volume uses n*n, each edge uses n')
    p.add_argument('--time-mode',choices=['local','global'],default='global')
    p.add_argument('--diagnostics',action='store_true')
    p.add_argument('--snapshot-every',type=int,default=1000)
    p.add_argument('--cfl',type=float,default=3.)
    p.add_argument('--rk',choices=['ssprk43','rk4'],default='ssprk43')
    p.add_argument('--threads',type=int,default=8,help='intra-op CPU threads for parallel element/face kernels')
    p.add_argument('--max-steps',type=int,default=200000)
    p.add_argument('--sample-every',type=int,default=50)
    p.add_argument('--window',type=int,default=1000,help='load stability window in iterations')
    p.add_argument('--min-steps',type=int,default=4000)
    p.add_argument('--stop-criterion',choices=['residual','lift'],default='residual')
    p.add_argument('--cl-relative-tol',type=float,default=1e-6)
    p.add_argument('--cl-relative-floor',type=float,default=1e-12,help='denominator floor for near-zero CL')
    p.add_argument('--cl-tol',type=float,default=2e-5)
    p.add_argument('--cd-tol',type=float,default=2e-6)
    p.add_argument('--residual-tol',type=float,default=1e-6)
    p.add_argument('--residual-relative-tol',type=float,default=1e-5)
    p.add_argument('--restart',type=Path)
    p.add_argument('--eager',action='store_true',help='disable torch.compile optimization')
    return p


def convergence(history,args,initial_residual,cl_pair=None):
    recent=[r for r in history if r[0]>=history[-1][0]-args.window]
    ready=history[-1][0]>=args.min_steps and recent[-1][0]-recent[0][0]>=args.window
    cl_span=max(r[2] for r in recent)-min(r[2] for r in recent)
    cd_span=max(r[3] for r in recent)-min(r[3] for r in recent)
    residual=history[-1][4]
    residual_ok=residual<=args.residual_tol and residual<=initial_residual*args.residual_relative_tol
    # Window and residual statistics remain diagnostic only. Never compare
    # sampled history rows as if they were adjacent time steps.
    relative_change=None
    if cl_pair is not None:
        previous,current=cl_pair
        relative_change=abs(current-previous)/max(abs(previous),args.cl_relative_floor)
    lift_ok=relative_change is not None and math.isfinite(relative_change) and relative_change<args.cl_relative_tol
    passed=residual_ok if args.stop_criterion=='residual' else lift_ok
    return dict(converged=bool(passed),criterion=args.stop_criterion,
                CL_relative_change=relative_change,CL_previous=cl_pair[0] if cl_pair else None,
                CL_current=cl_pair[1] if cl_pair else None,CL_span=cl_span,CD_span=cd_span,
                residual=residual,residual_relative=residual/max(initial_residual,1e-300),
                window_steps=args.window,residual_ok=bool(residual_ok))


@torch.no_grad()
def run(args):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    if args.threads<1:raise ValueError('threads must be positive')
    if not 0<args.mach<1 or not math.isfinite(args.alpha) or args.p<0 or args.cfl<=0:
        raise ValueError('subsonic Mach, finite alpha, nonnegative p and positive CFL required')
    if not math.isfinite(args.cl_relative_tol) or args.cl_relative_tol<=0 or not math.isfinite(args.cl_relative_floor) or args.cl_relative_floor<=0:
        raise ValueError('positive finite CL relative tolerance and denominator floor required')
    if args.quadrature<args.p+1 or args.window<2*args.sample_every or args.sample_every<=0 or args.max_steps<0 or args.snapshot_every<1:
        raise ValueError('invalid quadrature, window or iteration settings')
    args.output.mkdir(parents=True,exist_ok=True)
    a=Airfoil(args.mesh,args.p,args.mach,args.alpha,args.quadrature)
    s=a.s
    mesh_hash=hashlib.sha256(args.mesh.read_bytes()).hexdigest()
    U=a.far.expand(s.ndof,4).clone()
    history=[]; step=0; pseudo=0.; cfl=args.cfl; rejected=0
    residual_reference=None
    restarted_from_method=None
    if args.restart:
        checkpoint=torch.load(args.restart,weights_only=False)
        old=checkpoint['stats']
        if old['mesh_sha256']!=mesh_hash or old['p']!=args.p or old['mach']!=args.mach or old['quadrature']!=args.quadrature:
            raise ValueError('restart mesh/p/Mach/quadrature mismatch')
        U=checkpoint['U'].clone()
        if old['alpha_deg']==args.alpha and old['time_mode']==('physical_time' if args.time_mode=='global' else 'local_pseudo_time'):
            step=old['steps'];pseudo=old.get('pseudo_time_min',0.)
            residual_reference=old.get('initial_residual_reference',checkpoint['history'][0][4])
            if old['method']=='explicit_'+args.rk:
                history=checkpoint.get('history',[])
            else:
                restarted_from_method=old['method']
    if not a.valid(U): raise ValueError('nonphysical initial state')
    rhs=a.rhs
    if not args.eager:
        os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR','/tmp/fr-naca-inductor')
        os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','2')
        rhs=torch.compile(a.rhs,fullgraph=True)
    advance=a.make_step(args.rk,args.time_mode,compile=not args.eager)
    cfl_tensor=U.new_empty(())
    start=time.monotonic()
    starting_residual=float(s.norm(rhs(U)).max())
    initial_residual=residual_reference if residual_reference is not None else (history[0][4] if history else starting_residual)
    if not history:
        cl,cd,_=a.loads(U)
        history.append([step,pseudo,cl,cd,starting_residual,cfl])
    status='running'
    result=convergence(history,args,initial_residual)
    stop_requested=False
    def request_stop(signum,frame):
        nonlocal stop_requested
        stop_requested=True
    previous_signal=signal.signal(signal.SIGINT,request_stop)
    diagnostic=Diagnostics(a,args.output) if args.diagnostics else None
    if diagnostic:
        R=rhs(U);dt=a.local_dt(U,min(cfl,1.+step/200.))
        if args.time_mode=='global':dt=dt.min().expand_as(dt)
        diagnostic.sample(U,R,step,pseudo,dt)
        diagnostic.snapshot(U,R,step,pseudo)
    def save(status):
        cl,cd,pressure=a.loads(U)
        stats=dict(alpha_deg=args.alpha,mach=args.mach,p=args.p,quadrature=args.quadrature,
                   method='explicit_'+args.rk,time_mode='physical_time' if args.time_mode=='global' else 'local_pseudo_time',steps=step,
                   initial_residual_reference=initial_residual,restarted_from_method=restarted_from_method,
                   threads=args.threads,
                   pseudo_time_min=pseudo,cfl=cfl,rejected_batches=rejected,
                   physical_time=pseudo if args.time_mode=='global' else None,
                   gauss_points_per_direction=args.quadrature,volume_quadrature_points=args.quadrature**2,
                   face_quadrature_points=args.quadrature,diagnostics=diagnostic.latest if diagnostic else None,
                   CL=cl,CD_pressure=cd,Cf=0.,steady_convergence_verified=result['converged'],
                   convergence=result,status=status,mesh=str(args.mesh.resolve()),mesh_sha256=mesh_hash,
                   ndof=s.ndof,cells=s.blocks[0]['ne'],elapsed_seconds=time.monotonic()-start,
                   criteria=dict(stopping_rule=args.stop_criterion,
                                 CL_relative_tolerance=args.cl_relative_tol,CL_denominator_floor=args.cl_relative_floor,
                                 CL_span=args.cl_tol,CD_span=args.cd_tol,residual_absolute=args.residual_tol,
                                 residual_relative=args.residual_relative_tol,window_steps=args.window,min_steps=args.min_steps))
        with (args.output/'history.csv').open('w') as f:
            time_column='physical_time' if args.time_mode=='global' else 'pseudo_time_min'
            w=csv.writer(f);w.writerow(['iteration',time_column,'CL','CD_pressure','residual_rms_max','CFL']);w.writerows(history)
        xy=s.face_coordinates[a.wall_ids]
        cp=(pressure-1/1.4)/(.5*args.mach**2)
        with (args.output/'cp.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['x','y','Cp','Cf'])
            w.writerows(torch.cat((xy,cp[...,None],torch.zeros_like(cp[...,None])),-1).reshape(-1,4).tolist())
        temporary=args.output/'solution.tmp'
        torch.save(dict(U=U,coords=s.coords,history=history,stats=stats),temporary)
        temporary.replace(args.output/'solution.pt')
        temporary=args.output/'metrics.tmp'
        temporary.write_text(json.dumps(stats,indent=2)+'\n');temporary.replace(args.output/'metrics.json')
        return stats
    save(status)
    while step<args.max_steps and not result['converged'] and not stop_requested:
        old=U.clone(); oldpseudo=pseudo
        count=min(args.sample_every,args.max_steps-step)
        # CFL is held constant in each complete RK step. Uniform startup ramp avoids
        # impulsively imposing a large wall correction on a uniform flow.
        completed=0
        previous_cl=a.loads(U)[0] if args.stop_criterion=='lift' else None
        cl_pair=None
        for j in range(count):
            if stop_requested:break
            actual_cfl=min(cfl,1.+(step+j)/200.)
            cfl_tensor.fill_(actual_cfl)
            U,dt=advance(U,cfl_tensor)
            pseudo+=float(dt.min())
            completed+=1
            if args.stop_criterion=='lift':
                current_cl=a.loads(U)[0]
                cl_pair=(previous_cl,current_cl)
                relative_change=abs(current_cl-previous_cl)/max(abs(previous_cl),args.cl_relative_floor)
                previous_cl=current_cl
                if math.isfinite(relative_change) and relative_change<args.cl_relative_tol:
                    break
        if not a.valid(U):
            U=old;pseudo=oldpseudo;cfl*=.5;rejected+=1
            print(f'alpha={args.alpha:g} reject at {step}, reduce CFL to {cfl:g}',flush=True)
            if cfl<.05:
                status='nonphysical_state';break
            continue
        step+=completed
        R=rhs(U)
        residual=float(s.norm(R).max())
        cl,cd,_=a.loads(U)
        history.append([step,pseudo,cl,cd,residual,cfl])
        result=convergence(history,args,initial_residual,cl_pair)
        if diagnostic:
            diagnostic.sample(U,R,step,pseudo,dt)
            if step%args.snapshot_every==0 or stop_requested or step==args.max_steps or result['converged']:
                diagnostic.snapshot(U,R,step,pseudo)
        if step%500==0 or result['converged'] or step==args.max_steps:
            stats=save('converged' if result['converged'] else 'running')
            print(f'alpha={args.alpha:g} step={step} time={pseudo:.6f} CL={cl:.8f} CD={cd:.8f} R={residual:.3e} '
                  f'spans=({result["CL_span"]:.2e},{result["CD_span"]:.2e}) sec={time.monotonic()-start:.1f}',flush=True)
    if result['converged']: status='converged'
    elif stop_requested: status='stopped_by_user'
    elif status=='running': status='max_steps'
    stats=save(status)
    if diagnostic:diagnostic.close()
    signal.signal(signal.SIGINT,previous_signal)
    print(json.dumps(stats),flush=True)
    return stats


if __name__=='__main__':
    run(parser().parse_args())
