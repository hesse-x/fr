"""Explicit steady Euler run with auditable load/residual convergence."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import signal
import importlib.util
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    base=Path(__file__).parent/'results'
    p.add_argument('--mesh',type=Path,default=base/'mesh/structured.msh')
    p.add_argument('--output',type=Path,default=base/'alpha_0')
    p.add_argument('--alpha',type=float,default=0.)
    p.add_argument('--mach',type=float,default=.3)
    p.add_argument('--p',type=int,default=3)
    p.add_argument('--quadrature',type=int,default=6,help='Gauss points per direction: tri volume uses n*n, each edge uses n')
    p.add_argument('--time-mode',choices=['local','global'],default='global')
    p.add_argument('--cfl',type=float,default=3.)
    p.add_argument('--rk',choices=['ssprk43','rk4'],default='ssprk43')
    p.add_argument('--threads',type=int,default=8,help='intra-op CPU threads for parallel element/face kernels')
    p.add_argument('--max-steps',type=int,default=200000)
    p.add_argument('--sample-every',type=int,default=50)
    p.add_argument('--window',type=int,default=1000,help='load stability window in iterations')
    p.add_argument('--residual-tol',type=float,default=1e-6)
    p.add_argument('--residual-relative-tol',type=float,default=1e-5)
    p.add_argument('--restart',type=Path)
    return p


def convergence(history,args,initial_residual):
    recent=[r for r in history if r[0]>=history[-1][0]-args.window]
    residual=history[-1][4]
    passed=residual<=args.residual_tol and residual<=initial_residual*args.residual_relative_tol
    return dict(converged=bool(passed),criterion='residual',
                CL_span=max(r[2] for r in recent)-min(r[2] for r in recent),
                CD_span=max(r[3] for r in recent)-min(r[3] for r in recent),
                residual=residual,residual_relative=residual/max(initial_residual,1e-300),
                window_steps=args.window,residual_ok=bool(passed))


@torch.no_grad()
def run(args):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    if args.threads<1:raise ValueError('threads must be positive')
    if not 0<args.mach<1 or not math.isfinite(args.alpha) or args.p<0 or args.cfl<=0:
        raise ValueError('subsonic Mach, finite alpha, nonnegative p and positive CFL required')
    if args.quadrature<args.p+1 or args.window<2*args.sample_every or args.sample_every<=0 or args.max_steps<0:
        raise ValueError('invalid quadrature, window or iteration settings')
    args.output.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('naca_euler_native',Path(__file__).with_name('native.py'))
    native=importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
    a=native.NativeAirfoil(args.mesh,args.p,args.mach,args.alpha,args.quadrature)
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
    advance=a.make_step(args.rk,args.time_mode)
    residual_norm=a.residual_norm
    cfl_tensor=U.new_empty(())
    start=time.monotonic()
    starting_residual=residual_norm(rhs(U))
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
    def save(status):
        cl,cd,pressure=a.loads(U)
        stats=dict(alpha_deg=args.alpha,mach=args.mach,p=args.p,quadrature=args.quadrature,
                   method='explicit_'+args.rk,time_mode='physical_time' if args.time_mode=='global' else 'local_pseudo_time',steps=step,
                   initial_residual_reference=initial_residual,restarted_from_method=restarted_from_method,
                   threads=args.threads,implementation=a.implementation,
                   pseudo_time_min=pseudo,cfl=cfl,rejected_batches=rejected,
                   physical_time=pseudo if args.time_mode=='global' else None,
                   gauss_points_per_direction=args.quadrature,volume_quadrature_points=args.quadrature**2,
                   face_quadrature_points=args.quadrature,
                   CL=cl,CD_pressure=cd,Cf=0.,steady_convergence_verified=result['converged'],
                   convergence=result,status=status,mesh=str(args.mesh.resolve()),mesh_sha256=mesh_hash,
                   ndof=s.ndof,cells=s.blocks[0]['ne'],elapsed_seconds=time.monotonic()-start,
                   criteria=dict(stopping_rule='residual',residual_absolute=args.residual_tol,
                                 residual_relative=args.residual_relative_tol,window_steps=args.window))
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
        for j in range(count):
            if stop_requested:break
            actual_cfl=min(cfl,1.+(step+j)/200.)
            cfl_tensor.fill_(actual_cfl)
            U,dt=advance(U,cfl_tensor)
            pseudo+=float(dt.min())
            completed+=1
        if not a.valid(U):
            U=old;pseudo=oldpseudo;cfl*=.5;rejected+=1
            print(f'alpha={args.alpha:g} reject at {step}, reduce CFL to {cfl:g}',flush=True)
            if cfl<.05:
                status='nonphysical_state';break
            continue
        step+=completed
        R=rhs(U)
        residual=residual_norm(R)
        cl,cd,_=a.loads(U)
        history.append([step,pseudo,cl,cd,residual,cfl])
        result=convergence(history,args,initial_residual)
        if step%500==0 or result['converged'] or step==args.max_steps:
            stats=save('converged' if result['converged'] else 'running')
            print(f'alpha={args.alpha:g} step={step} time={pseudo:.6f} CL={cl:.8f} CD={cd:.8f} R={residual:.3e} '
                  f'spans=({result["CL_span"]:.2e},{result["CD_span"]:.2e}) sec={time.monotonic()-start:.1f}',flush=True)
    if result['converged']: status='converged'
    elif stop_requested: status='stopped_by_user'
    elif status=='running': status='max_steps'
    stats=save(status)
    signal.signal(signal.SIGINT,previous_signal)
    print(json.dumps(stats),flush=True)
    return stats


if __name__=='__main__':
    run(parser().parse_args())
