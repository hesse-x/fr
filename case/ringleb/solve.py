"""Ringleb exact solution and benchmark adapter."""
import argparse
import json
from pathlib import Path

import torch

from fr.equations import GAMMA
from fr.core import CartesianFR
from fr.fluxes import make_numerical_flux
from fr.equations import make_equation
from fr.time import integrate
from fr.output import export, export_cloud
from fr.plotting import add_plot_arguments

def ringleb(x, y):
    """Positive-y, positive-velocity subsonic branch; gamma=1.4.

    Invert the hodograph radius equation by bracketed bisection.
    This routine deliberately supports only the benchmark rectangle below.
    """
    def quantities(q):
        c = (1 - .2*q*q).sqrt()
        rho = c**5
        J = 1/c + 1/(3*c**3) + 1/(5*c**5) - .5*torch.log((1+c)/(1-c))
        return c, rho, J
    def residual(q):
        _, rho, J = quantities(q)
        return (x-J/2)**2 + y*y - 1/(4*rho*rho*q**4)
    lo, hi = torch.full_like(x, .2), torch.full_like(x, .9)
    if not ((residual(lo) < 0) & (residual(hi) > 0)).all():
        raise ValueError('Point outside supported subsonic Ringleb branch')
    for _ in range(55):
        mid = (lo+hi)/2
        neg = residual(mid) < 0
        lo, hi = torch.where(neg, mid, lo), torch.where(neg, hi, mid)
    q = (lo+hi)/2
    c, rho, J = quantities(q)
    s2 = (1 - 2*rho*q*q*(x-J/2))/2
    if not ((s2 > 0) & (s2 < 1)).all():
        raise ValueError('Invalid hodograph angle')
    u, v = q*(1-s2).sqrt(), q*s2.sqrt()
    p = rho*c*c/GAMMA
    return torch.stack((rho, rho*u, rho*v, p/(GAMMA-1)+.5*rho*q*q), -1)




class RinglebCase(CartesianFR):
    def __init__(self,n=8,p=3,device='cpu'):
        self.equation=make_equation('euler',dim=2)
        super().__init__((n,n),((-.5,0.),(1.5,2.)),p,
                         flux=self.equation.flux,
                         numerical_flux=make_numerical_flux('rusanov',self.equation),
                         boundary=lambda coords,inside,axis,side,t: ringleb(*coords),
                         speed=self.equation.speed,device=device)
        self.n=n
        self.x,self.y=self.coords
        self.hx,self.hy=self.h
        self.exact=ringleb(self.x,self.y)
        # Stationary exact boundary values can be computed once.
        self.bc=[tuple(ringleb(*coords) for coords in pair) for pair in self.face_coords]
        self.boundary=lambda coords,inside,axis,side,t: self.bc[axis][0 if side==-1 else 1]

    def run(self,final_time=.1,cfl=.15,perturb=0.,method='ssprk43'):
        U=self.exact.clone()
        if perturb:
            bump=torch.sin(torch.pi*(self.x+.5)/.5)*torch.sin(torch.pi*(self.y-1.5)/.5)
            U=U*(1+perturb*bump[...,None])
        U,stats=integrate(self.rhs,U,final_time,dt=lambda U,t:self.timestep(U,cfl),
                          method=method,validate=self.equation.valid)
        stats.update(n=self.n,p=self.p,rk=method,l2=self.norm(U-self.exact).tolist(),
                     residual=self.norm(self.rhs(U,stats['time'])).tolist())
        return U,stats




def main():
    parser = argparse.ArgumentParser(description='PyTorch FR/DG Ringleb benchmark')
    parser.add_argument('--n',type=int,default=8)
    parser.add_argument('--p',type=int,default=3)
    parser.add_argument('--time',type=float,default=.1)
    parser.add_argument('--cfl',type=float,default=.15)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--rk',choices=['rk4','ssprk43'],default='ssprk43')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--perturb',type=float,default=0.)
    parser.add_argument('--convergence',action='store_true')
    parser.add_argument('--output',default='results/ringleb')
    add_plot_arguments(parser,unsteady=False)
    a = parser.parse_args()
    if a.time < 0 or a.cfl <= 0:
        parser.error('time >= 0 and cfl > 0 required')
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(a.threads)
    out = Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    records = []
    for n in ([a.n,2*a.n,4*a.n] if a.convergence else [a.n]):
        s = RinglebCase(n,a.p,a.device)
        U, stats = s.run(a.time,a.cfl,a.perturb,method=a.rk)
        if records:
            import math
            stats['density_order'] = math.log2(records[-1]['l2'][0]/stats['l2'][0])
        records.append(stats)
        print(json.dumps(stats),flush=True)
        stem=out/f'ringleb_n{n}_p{a.p}'
        export(stem,s,U)
        if not a.no_plot: export_cloud(stem,s,U,stats['time'])
        torch.save(dict(x=s.x.cpu(),y=s.y.cpu(),U=U.cpu(),exact=s.exact.cpu(),stats=stats),out/f'ringleb_n{n}_p{a.p}.pt')
    (out/'metrics.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__ == '__main__':
    main()
