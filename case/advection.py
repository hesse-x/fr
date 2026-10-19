"""Run a configurable scalar advection problem in 1D/2D/3D.

From the repository root: venv/bin/python -m case.advection --dim 3
"""
from pathlib import Path
from fr.plotting import PlotRecorder,add_plot_arguments
import argparse
import json
import math
import torch
from fr import CartesianFR, integrate



def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dim',type=int,choices=(1,2,3),default=1)
    parser.add_argument('--n',type=int,default=4)
    parser.add_argument('--p',type=int,default=3)
    parser.add_argument('--time',type=float,default=.1)
    parser.add_argument('--output',default='results/advection')
    add_plot_arguments(parser)
    a=parser.parse_args()
    torch.set_num_threads(1)
    velocity=(1.,-.5,.25)[:a.dim]
    # Constant-velocity upwind flux can be expressed using only Fminus/Fplus.
    # Use the rich numerical_flux interface instead when state-dependent
    # dissipation is required (see rusanov_flux and Euler).
    s=CartesianFR((a.n,)*a.dim,((0.,1.),)*a.dim,a.p,
                  flux=lambda U,axis: velocity[axis]*U,
                  numerical_flux=lambda minus,plus,axis: minus.flux if velocity[axis]>=0 else plus.flux,
                  speed=lambda U,axis: torch.full_like(U[...,0],abs(velocity[axis])),
                  periodic=True)
    exact=lambda t: (1+.1*sum(torch.sin(2*math.pi*(x-v*t)) for x,v in zip(s.coords,velocity)))[...,None]
    U0=exact(0.)
    plot=PlotRecorder(s,{'Scalar':lambda U:U[...,0]},a.time,frames=a.frames,fps=a.fps,enabled=not a.no_plot)
    U,stats=integrate(s.rhs,U0,a.time,dt=lambda U,t:s.timestep(U,.15),method='ssprk43',monitor=plot)
    plot.finish(Path(a.output)/f'advection_{a.dim}d_n{a.n}_p{a.p}',U,stats['time'])
    stats.update(dim=a.dim,n=a.n,p=a.p,L2=s.norm(U-exact(a.time)).tolist(),
                 mass_change=(s.integral(U)-s.integral(U0)).tolist())
    print(json.dumps(stats,indent=2))


if __name__ == '__main__':
    main()
