"""Explicit SSPRK(4,3) laminar NS with residual stopping and wall loads."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from airfoil import Airfoil


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).parent/'results'
    p.add_argument('--mesh',type=Path,default=base/'mesh/coarse.msh')
    p.add_argument('--output',type=Path,default=base/'alpha0_re1000_coarse')
    p.add_argument('--re',type=float,default=1000.)
    p.add_argument('--mach',type=float,default=.1)
    p.add_argument('--alpha',type=float,default=0.)
    p.add_argument('--p',type=int,default=2)
    p.add_argument('--quadrature',type=int,default=6)
    p.add_argument('--cfl',type=float,default=.2)
    p.add_argument('--viscous-step-factor',type=float,default=1.)
    p.add_argument('--time',type=float,default=10.,help='target convective time; local mode accumulates minimum pseudo time')
    p.add_argument('--max-steps',type=int,default=10000)
    p.add_argument('--output-every',type=int,default=100,help='residual console/CSV output interval in steps (default: 100); stopping is checked every step')
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--residual-tol',type=float,default=1e-5)
    p.add_argument('--time-mode',choices=['global','local'],default='global')
    p.add_argument('--restart',type=Path)
    p.add_argument('--compile',action='store_true')
    a = p.parse_args()
    if not (0<a.mach<1 and math.isfinite(a.re) and a.re>0 and math.isfinite(a.alpha)
            and math.isfinite(a.cfl) and a.cfl>0 and math.isfinite(a.time) and a.time>=0
            and math.isfinite(a.viscous_step_factor) and a.viscous_step_factor>0
            and math.isfinite(a.residual_tol) and a.residual_tol>0
            and a.p>=1 and a.quadrature>=a.p+1 and a.max_steps>=0 and a.output_every>0 and a.threads>0):
        p.error('invalid physical or numerical parameters')
    torch.set_default_dtype(torch.float64); torch.set_num_threads(a.threads)
    physics = Airfoil(a.mesh,a.p,a.mach,a.alpha,a.quadrature,reynolds=a.re,viscous_step_factor=a.viscous_step_factor)
    s = physics.s; U = physics.far.expand(s.ndof,4).clone()
    mesh_hash=hashlib.sha256(a.mesh.read_bytes()).hexdigest()
    a.output.mkdir(parents=True,exist_ok=True)
    t=0.; step=0; history=[]; status='running'; rejected=0
    if a.restart:
        checkpoint=torch.load(a.restart,weights_only=False); old=checkpoint['stats']
        for key,value in dict(mesh_sha256=mesh_hash,reynolds=a.re,mach=a.mach,alpha_deg=a.alpha,p=a.p,quadrature=a.quadrature).items():
            if old.get(key)!=value: raise ValueError(f'restart mismatch: {key}')
        if old.get('method')!='explicit_ssprk43' or old.get('time_mode')!=a.time_mode:
            raise ValueError('restart requires same explicit method and time mode')
        U=checkpoint['U'].clone(); t=old['pseudo_time_min']; step=old['steps']; history=checkpoint['history']
    if not physics.valid(U): raise ValueError('invalid initial state')
    rhs=physics.rhs
    if a.compile:
        os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR','/tmp/fr-ns-steady-inductor')
        os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','2')
        rhs=torch.compile(rhs,fullgraph=True)
    advance=physics.make_step_with_rhs(a.time_mode,compile=a.compile)
    start=time.monotonic(); residual=math.inf; cfl=a.cfl
    # Keep the RHS of exactly U: it serves both logging and RK stage one.
    F=rhs(U)
    residual=float(s.norm(F).max())
    residual_file=a.output/'residuals.csv'
    residual_log=residual_file.open('a' if a.restart else 'w')
    residual_writer=csv.writer(residual_log)
    if residual_log.tell()==0:
        residual_writer.writerow(['step','physical_or_min_pseudo_time','convective_time','residual','CFL','elapsed_seconds'])
    last_reported_step=None
    def report():
        nonlocal last_reported_step
        elapsed=time.monotonic()-start
        residual_writer.writerow([step,t,t*a.mach,residual,cfl,elapsed])
        residual_log.flush()
        print(f'step={step} t_conv={t*a.mach:.6g} residual={residual:.9g} CFL={cfl:.5g} elapsed={elapsed:.1f}',flush=True)
        last_reported_step=step
    def save():
        loads,wall=physics.wall_data(U)
        row=[step,t,t*a.mach,loads['CL'],loads['CD_total'],residual,cfl,time.monotonic()-start]
        if history and history[-1][0]==step:
            history[-1]=row
        else:
            history.append(row)
        residual_log.flush()
        stats=dict(status=status,steps=step,physical_time=t if a.time_mode=='global' else None,
                   pseudo_time_min=t,convective_time=t*a.mach,time_mode=a.time_mode,
                   method='explicit_ssprk43',reynolds=a.re,mach=a.mach,alpha_deg=a.alpha,
                   mu=physics.mu,prandtl=physics.prandtl,p=a.p,quadrature=a.quadrature,cfl=cfl,
                   viscous_step_factor=a.viscous_step_factor,cells=s.blocks[0]['ne'],ndof=s.ndof,residual=residual,residual_tolerance=a.residual_tol,
                   residual_definition='max over conservative components of physical-volume-weighted RMS RHS',
                   output_every=a.output_every,
                   steady_convergence_verified=bool(residual<a.residual_tol),mesh_sha256=mesh_hash,
                   elapsed_seconds=time.monotonic()-start,rejected_steps=rejected,
                   model='2D compressible laminar NS',wall='stationary no-slip adiabatic',**loads)
        with (a.output/'history.csv').open('w') as f:
            w=csv.writer(f); w.writerow(['step','physical_or_min_pseudo_time','convective_time','CL','CD_total','residual','CFL','elapsed_seconds']); w.writerows(history)
        with (a.output/'wall.csv').open('w') as f:
            w=csv.writer(f); w.writerow(['x','y','Cp','Cf','nx_fluid','ny_fluid','viscous_traction_x_over_q','viscous_traction_y_over_q']); w.writerows(wall.tolist())
        torch.save(dict(U=U,coords=s.coords,stats=stats,history=history),a.output/'solution.tmp')
        (a.output/'solution.tmp').replace(a.output/'solution.pt')
        (a.output/'metrics.json').write_text(json.dumps(stats,indent=2)+'\n')
    report(); save()
    cfl_tensor=U.new_empty(())
    try:
        while step<a.max_steps and t*a.mach<a.time and residual>=a.residual_tol:
            # Hold one CFL throughout all stages; never accept an invalid step.
            cfl_tensor.fill_(cfl)
            candidate,dt,next_rhs=advance(U,cfl_tensor,F)
            if not physics.valid(candidate):
                cfl*=.5; rejected+=1
                print(f'rejected step, reducing CFL={cfl:.6g}',flush=True)
                if cfl<1e-6: status='nonphysical_state'; break
                continue
            U=candidate; F=next_rhs; t+=float(dt.min()); step+=1
            residual=float(s.norm(F).max())
            if step%a.output_every==0:
                report()
            if step%1000==0:
                save()
        else:
            status='converged' if residual<a.residual_tol else ('final_time' if t*a.mach>=a.time else 'max_steps')
    except KeyboardInterrupt:
        status='stopped_by_user'
    if residual<a.residual_tol: status='converged'
    if last_reported_step!=step:
        report()
    save()
    residual_log.close()
    if status=='nonphysical_state': raise RuntimeError('nonphysical candidate rejected; last valid state saved')


if __name__=='__main__': main()
