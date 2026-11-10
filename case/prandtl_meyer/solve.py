"""Centered, clockwise Prandtl--Meyer expansion, corner at (0,0).

The upstream velocity points along +x. The fluid lies above a wall turning
clockwise by turn_degrees. All supported sample points have x>0, above the
turned wall. Angles are radians internally. No source terms are added.
"""
import math
import argparse
import json
from pathlib import Path

import torch

from fr.core import CartesianFR
from fr.fluxes import make_numerical_flux
from fr.equations import make_equation
from fr.time import integrate
from fr.output import export, export_cloud
from fr.plotting import add_plot_arguments


def prandtl_meyer(mach, gamma=1.4):
    """Prandtl--Meyer angle nu(M), radians; M>=1, gamma>1."""
    if not math.isfinite(gamma) or gamma<=1:
        raise ValueError('gamma must be finite and >1')
    if not isinstance(mach,torch.Tensor):
        mach=torch.as_tensor(mach,dtype=torch.float64)
    if not (torch.isfinite(mach).all() and (mach>=1).all()):
        raise ValueError('Prandtl--Meyer function requires finite Mach>=1')
    root=(mach*mach-1).sqrt()
    factor=math.sqrt((gamma+1)/(gamma-1))
    return factor*torch.atan(root/factor)-torch.atan(root)


def _nu(mach,gamma):
    factor=math.sqrt((gamma+1)/(gamma-1))
    root=math.sqrt(mach*mach-1)
    return factor*math.atan(root/factor)-math.atan(root)


class PrandtlMeyerFan:
    def __init__(self,mach=2.5,turn_degrees=15.,gamma=1.4,rho=1.,pressure=1.):
        if not all(math.isfinite(v) for v in (mach,turn_degrees,gamma,rho,pressure)):
            raise ValueError('all fan parameters must be finite')
        if mach<=1 or gamma<=1 or rho<=0 or pressure<=0 or not 0<turn_degrees<90:
            raise ValueError('require Mach>1, gamma>1, rho,p>0 and 0<turn<90 degrees')
        self.mach1,self.gamma,self.rho1,self.pressure1=mach,gamma,rho,pressure
        self.turn=math.radians(turn_degrees)
        self.nu1=_nu(mach,gamma)
        target=self.nu1+self.turn
        limit=math.pi/2*(math.sqrt((gamma+1)/(gamma-1))-1)
        if target>=limit:
            raise ValueError('turn exceeds the finite-Mach expansion limit')
        lo,hi=mach,2*mach
        while _nu(hi,gamma)<target:
            hi*=2
        for _ in range(70):
            mid=(lo+hi)/2
            if _nu(mid,gamma)<target:
                lo=mid
            else:
                hi=mid
        self.mach2=(lo+hi)/2
        self.head=math.asin(1/mach)
        self.tail=math.asin(1/self.mach2)-self.turn
        self.equation=make_equation('euler',dim=2,gamma=gamma)

    def fields(self,x,y):
        """Return density, velocity vector, pressure and Mach on the fan.

        Regions: phi>=head is upstream, phi<=tail is downstream. In the fan,
        phi=mu(M)-[nu(M)-nu(M1)]. Constant regions join continuously.
        """
        if not (torch.isfinite(x).all() and torch.isfinite(y).all() and (x>0).all()):
            raise ValueError('fan coordinates must be finite and x>0')
        phi=torch.atan2(y,x)
        if (phi < -self.turn-1e-12).any():
            raise ValueError('point is below the turned wall')
        target=phi.clamp(self.tail,self.head)
        lo,hi=torch.full_like(x,self.mach1),torch.full_like(x,self.mach2)
        for _ in range(55):
            mid=(lo+hi)/2
            angle=torch.asin(1/mid)-(prandtl_meyer(mid,self.gamma)-self.nu1)
            higher=angle>target
            lo,hi=torch.where(higher,mid,lo),torch.where(higher,hi,mid)
        M=(lo+hi)/2
        # Use exact constants outside the fan to avoid endpoint roundoff.
        M=torch.where(phi>=self.head,self.mach1,torch.where(phi<=self.tail,self.mach2,M))
        theta=-(prandtl_meyer(M,self.gamma)-self.nu1)
        ratio=(1+.5*(self.gamma-1)*self.mach1**2)/(1+.5*(self.gamma-1)*M*M)
        rho=self.rho1*ratio**(1/(self.gamma-1))
        pressure=self.pressure1*ratio**(self.gamma/(self.gamma-1))
        speed=M*(self.gamma*pressure/rho).sqrt()
        velocity=torch.stack((speed*theta.cos(),speed*theta.sin()),-1)
        return rho,velocity,pressure,M

    def exact(self,x,y):
        rho,velocity,pressure,_=self.fields(x,y)
        energy=pressure/(self.gamma-1)+.5*rho*(velocity*velocity).sum(-1)
        return torch.cat((rho[...,None],rho[...,None]*velocity,energy[...,None]),-1)

    def reference(self):
        ratio=(1+.5*(self.gamma-1)*self.mach1**2)/(1+.5*(self.gamma-1)*self.mach2**2)
        return dict(mach1=self.mach1,mach2=self.mach2,gamma=self.gamma,
                    turn_degrees=math.degrees(self.turn),head_degrees=math.degrees(self.head),
                    tail_degrees=math.degrees(self.tail),pressure_ratio=ratio**(self.gamma/(self.gamma-1)),
                    density_ratio=ratio**(1/(self.gamma-1)),temperature_ratio=ratio)

    def bounds(self,region='full'):
        if region=='smooth':
            angle=(self.head+self.tail)/2
            width=.5*math.sin((self.head-self.tail)/2)
            cx,cy=math.cos(angle),math.sin(angle)
            return ((cx-width,cx+width),(cy-width,cy+width))
        if region!='full':
            raise ValueError('region must be full or smooth')
        # Upper edge includes unexpanded flow even at the rightmost x.
        return ((.5,1.5),(0.,1.6*math.tan(self.head)))


class PrandtlMeyerCase(CartesianFR):
    def __init__(self,n=16,p=3,device='cpu',*,fan=None,region='full'):
        self.fan=PrandtlMeyerFan() if fan is None else fan
        self.equation=self.fan.equation
        self.region,self.n=region,n
        super().__init__((n,n),self.fan.bounds(region),p,device=device,
                         flux=self.equation.flux,numerical_flux=make_numerical_flux('rusanov',self.equation),
                         speed=self.equation.speed,
                         boundary=lambda coords,inside,axis,side,t: self.fan.exact(*coords))
        self.x,self.y=self.coords
        self.exact=self.fan.exact(*self.coords)
        self.bc=[tuple(self.fan.exact(*coords) for coords in pair) for pair in self.face_coords]
        self.boundary=lambda coords,inside,axis,side,t:self.bc[axis][0 if side==-1 else 1]

    def run(self,final_time=.1,cfl=.15,method='ssprk43'):
        U,stats=integrate(self.rhs,self.exact,final_time,
                          dt=lambda U,t:self.timestep(U,cfl),method=method,validate=self.equation.valid)
        rho,velocity,pressure=self.equation.primitive(U)
        M=(velocity*velocity).sum(-1).sqrt()/(self.fan.gamma*pressure/rho).sqrt()
        exact_M=self.fan.fields(*self.coords)[-1]
        entropy=pressure/rho**self.fan.gamma
        stats.update(n=self.n,p=self.p,rk=method,region=self.region,bounds=self.bounds,
                     reference=self.fan.reference(),l2=self.norm(U-self.exact).tolist(),
                     mach_l2=self.norm((M-exact_M)[...,None]).item(),
                     residual=self.norm(self.rhs(U,stats['time'])).tolist(),
                     min_density=rho.min().item(),min_pressure=pressure.min().item(),
                     max_entropy_error=(entropy-self.fan.pressure1/self.fan.rho1**self.fan.gamma).abs().max().item())
        return U,stats




def main():
    parser=argparse.ArgumentParser(description='2-D Euler centered Prandtl--Meyer expansion fan')
    parser.add_argument('--n',type=int,default=16)
    parser.add_argument('--p',type=int,default=3)
    parser.add_argument('--mach',type=float,default=2.5)
    parser.add_argument('--turn',type=float,default=15.,help='clockwise turn in degrees')
    parser.add_argument('--gamma',type=float,default=1.4)
    parser.add_argument('--region',choices=('full','smooth'),default='full')
    parser.add_argument('--time',type=float,default=.1)
    parser.add_argument('--cfl',type=float,default=.15)
    parser.add_argument('--rk',choices=('rk4','ssprk43'),default='ssprk43')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--convergence',action='store_true')
    parser.add_argument('--output',default='results/prandtl_meyer')
    add_plot_arguments(parser,unsteady=False)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    fan=PrandtlMeyerFan(args.mach,args.turn,args.gamma)
    records=[]
    for n in ([args.n,2*args.n,4*args.n] if args.convergence else [args.n]):
        case=PrandtlMeyerCase(n,args.p,args.device,fan=fan,region=args.region)
        U,stats=case.run(args.time,args.cfl,args.rk)
        if records and stats['l2'][0]>0 and records[-1]['l2'][0]>0:
            stats['density_order']=math.log2(records[-1]['l2'][0]/stats['l2'][0])
        stem=out/f'pm_{args.region}_n{n}_p{args.p}'
        export(stem,case,U)
        if not args.no_plot: export_cloud(stem,case,U,stats["time"])
        torch.save(dict(x=case.x.cpu(),y=case.y.cpu(),U=U.cpu(),exact=case.exact.cpu(),stats=stats),stem.with_suffix('.pt'))
        records.append(stats)
        print(json.dumps(stats),flush=True)
    (out/'metrics.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':
    main()
