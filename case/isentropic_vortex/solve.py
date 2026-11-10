"""Convecting 2-D isentropic Euler vortex (Shu benchmark).

The exact infinite-plane solution is translated by the background velocity.
On a periodic box, nearest-image displacements are used: the Gaussian tail
must be negligible at half the box size (default box length 20).
"""
import math
from fr.plotting import PlotRecorder,add_plot_arguments,euler_fields
import argparse
import json
from pathlib import Path

import torch

from fr.core import CartesianFR
from fr.fluxes import make_numerical_flux
from fr.equations import make_equation
from fr.time import integrate
from fr.output import export


class IsentropicVortex:
    def __init__(self,beta=5.,gamma=1.4,velocity=(1.,1.),center=(0.,0.)):
        if len(velocity)!=2 or len(center)!=2:
            raise ValueError('velocity and center must have two components')
        if not all(math.isfinite(v) for v in (beta,gamma,*velocity,*center)) or gamma<=1:
            raise ValueError('finite parameters and gamma>1 required')
        # The minimum temperature occurs at the vortex center.
        depression=(gamma-1)*beta**2*math.e/(8*gamma*math.pi**2)
        if depression>=1:
            raise ValueError('vortex strength gives nonpositive central temperature')
        self.beta,self.gamma=beta,gamma
        self.velocity,self.center=tuple(velocity),tuple(center)
        self.equation=make_equation('euler',dim=2,gamma=gamma)

    def fields(self,x,y,t=0.,periods=None):
        """Return rho, velocity, pressure; far field rho=p=T=1.

        periods=None gives the infinite-plane exact solution. If provided,
        periods=(Lx,Ly) wraps each displacement into [-L/2,L/2).
        """
        if not math.isfinite(t):
            raise ValueError('time must be finite')
        offsets=[x-self.center[0]-self.velocity[0]*t,
                 y-self.center[1]-self.velocity[1]*t]
        if periods is not None:
            if len(periods)!=2 or any(not math.isfinite(L) or L<=0 for L in periods):
                raise ValueError('periods must contain two finite positive lengths')
            offsets=[torch.remainder(a+L/2,L)-L/2 for a,L in zip(offsets,periods)]
        dx,dy=offsets
        r2=dx*dx+dy*dy
        swirl=self.beta/(2*math.pi)*torch.exp((1-r2)/2)
        T=1-(self.gamma-1)*self.beta**2/(8*self.gamma*math.pi**2)*torch.exp(1-r2)
        rho=T**(1/(self.gamma-1))
        pressure=rho**self.gamma
        velocity=torch.stack((self.velocity[0]-swirl*dy,self.velocity[1]+swirl*dx),-1)
        return rho,velocity,pressure

    def exact(self,x,y,t=0.,periods=None):
        rho,velocity,pressure=self.fields(x,y,t,periods)
        energy=pressure/(self.gamma-1)+.5*rho*(velocity*velocity).sum(-1)
        return torch.cat((rho[...,None],rho[...,None]*velocity,energy[...,None]),-1)


class IsentropicVortexCase(CartesianFR):
    def __init__(self,n=16,p=3,device='cpu',*,vortex=None,length=20.):
        if not math.isfinite(length) or length<=0:
            raise ValueError('box length must be finite and positive')
        self.vortex=IsentropicVortex() if vortex is None else vortex
        self.equation=self.vortex.equation
        self.n,self.length=n,length
        super().__init__((n,n),((-length/2,length/2),)*2,p,device=device,
                         flux=self.equation.flux,numerical_flux=make_numerical_flux('rusanov',self.equation),
                         speed=self.equation.speed,periodic=True)
        self.x,self.y=self.coords
        self.initial=self.exact_at(0.)
        self.exact=self.initial

    def exact_at(self,t):
        return self.vortex.exact(*self.coords,t,periods=(self.length,)*2)

    def run(self,final_time=.1,cfl=.15,method='ssprk43',monitor=None):
        U,stats=integrate(self.rhs,self.initial,final_time,
                          dt=lambda U,t:self.timestep(U,cfl),method=method,validate=self.equation.valid,monitor=monitor)
        # Export and errors must refer to the moved vortex, not the initial state.
        self.exact=self.exact_at(stats['time'])
        rho,velocity,pressure=self.equation.primitive(U)
        center=[(c+v*stats['time']+self.length/2)%self.length-self.length/2
                for c,v in zip(self.vortex.center,self.vortex.velocity)]
        initial_integrals=self.integral(self.initial)
        final_integrals=self.integral(U)
        stats.update(n=self.n,p=self.p,rk=method,bounds=self.bounds,beta=self.vortex.beta,
                     gamma=self.vortex.gamma,background_velocity=self.vortex.velocity,
                     center=center,l2=self.norm(U-self.exact).tolist(),
                     min_density=rho.min().item(),min_pressure=pressure.min().item(),
                     max_entropy_error=(pressure/rho**self.vortex.gamma-1).abs().max().item(),
                     initial_integrals=initial_integrals.tolist(),final_integrals=final_integrals.tolist(),
                     conservation_change=(final_integrals-initial_integrals).tolist())
        return U,stats




def main():
    parser=argparse.ArgumentParser(description='2-D Euler periodic isentropic vortex convection')
    parser.add_argument('--n',type=int,default=16)
    parser.add_argument('--p',type=int,default=3)
    parser.add_argument('--beta',type=float,default=5.)
    parser.add_argument('--gamma',type=float,default=1.4)
    parser.add_argument('--velocity',type=float,nargs=2,default=(1.,1.),metavar=('U','V'))
    parser.add_argument('--center',type=float,nargs=2,default=(0.,0.),metavar=('X','Y'))
    parser.add_argument('--length',type=float,default=20.)
    parser.add_argument('--time',type=float,default=.1)
    parser.add_argument('--cfl',type=float,default=.15)
    parser.add_argument('--rk',choices=('rk4','ssprk43'),default='ssprk43')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--convergence',action='store_true')
    parser.add_argument('--output',default='results/isentropic_vortex')
    add_plot_arguments(parser)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    vortex=IsentropicVortex(args.beta,args.gamma,args.velocity,args.center)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    for n in ([args.n,2*args.n,4*args.n] if args.convergence else [args.n]):
        case=IsentropicVortexCase(n,args.p,args.device,vortex=vortex,length=args.length)
        plot=PlotRecorder(case,euler_fields(case.equation),args.time,frames=args.frames,fps=args.fps,enabled=not args.no_plot)
        U,stats=case.run(args.time,args.cfl,args.rk,monitor=plot)
        if records and stats['l2'][0]>0 and records[-1]['l2'][0]>0:
            stats['density_order']=math.log2(records[-1]['l2'][0]/stats['l2'][0])
        stem=out/f'vortex_n{n}_p{args.p}'
        export(stem,case,U)
        plot.finish(stem,U,stats["time"])
        torch.save(dict(x=case.x.cpu(),y=case.y.cpu(),U=U.cpu(),initial=case.initial.cpu(),exact=case.exact.cpu(),stats=stats),stem.with_suffix('.pt'))
        records.append(stats)
        print(json.dumps(stats),flush=True)
    (out/'metrics.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':
    main()
