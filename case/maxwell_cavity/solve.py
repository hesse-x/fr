"""3-D standing electromagnetic mode in a rectangular PEC cavity.

Run: python -m case.maxwell_cavity --n 4 --periods 1 --flux central
"""
from fr.plotting import PlotRecorder,add_plot_arguments,euler_fields
import argparse
import csv
import json
import math
from pathlib import Path

import torch

from fr.core import CartesianFR, central
from fr.equations import make_equation
from fr.fluxes import make_numerical_flux
from fr.time import integrate


class CavityMode:
    """Divergence-free analytic mode, E=E0*cos(omega*t), H=H0*sin(omega*t).

    All three mode indices are positive. This family has full 3-D spatial
    dependence; its polarization obeys k dot A=0 and all six PEC walls.
    """
    def __init__(self,lengths=(1.,1.3,1.7),mode=(1,1,1),epsilon=1.,mu=1.,amplitude=1.):
        if len(lengths)!=3 or any(not math.isfinite(L) or L<=0 for L in lengths):
            raise ValueError('three finite positive cavity lengths required')
        if len(mode)!=3 or any(not isinstance(m,int) or m<1 for m in mode):
            raise ValueError('three positive integer mode indices required')
        if not math.isfinite(amplitude) or amplitude==0:
            raise ValueError('finite nonzero electric amplitude required')
        self.lengths,self.mode=tuple(lengths),tuple(mode)
        self.equation=make_equation('maxwell',epsilon=epsilon,mu=mu)
        self.k=tuple(math.pi*m/L for m,L in zip(mode,lengths))
        self.omega=self.equation.c*math.sqrt(sum(k*k for k in self.k))
        self.period=2*math.pi/self.omega
        ax,ay,az=1.,1.,-(self.k[0]+self.k[1])/self.k[2]
        scale=amplitude/math.sqrt(ax*ax+ay*ay+az*az)
        self.polarization=(ax*scale,ay*scale,az*scale)
        self.amplitude=amplitude

    def basis(self,x,y,z):
        kx,ky,kz=self.k
        ax,ay,az=self.polarization
        sx,sy,sz=(k*q for k,q in zip(self.k,(x,y,z)))
        cx,cy,cz=sx.cos(),sy.cos(),sz.cos()
        sx,sy,sz=sx.sin(),sy.sin(),sz.sin()
        E=torch.stack((ax*cx*sy*sz,ay*sx*cy*sz,az*sx*sy*cz),-1)
        curl=torch.stack(((ky*az-kz*ay)*sx*cy*cz,
                          (kz*ax-kx*az)*cx*sy*cz,
                          (kx*ay-ky*ax)*cx*cy*sz),-1)
        H=-curl/(self.equation.mu*self.omega)
        return E,H

    def exact(self,x,y,z,t=0.):
        if not math.isfinite(t):
            raise ValueError('time must be finite')
        E,H=self.basis(x,y,z)
        return torch.cat((E*math.cos(self.omega*t),H*math.sin(self.omega*t)),-1)

    def energy(self):
        return self.equation.epsilon*math.prod(self.lengths)*self.amplitude**2/16


class MaxwellCavityCase(CartesianFR):
    def __init__(self,n=4,p=3,device='cpu',*,mode=None,cells=None,numerical_flux='central'):
        self.mode=CavityMode() if mode is None else mode
        self.equation=self.mode.equation
        if isinstance(numerical_flux,str):
            choices={name:make_numerical_flux(name,self.equation) for name in ('central','roe')}
            if numerical_flux not in choices:
                raise ValueError('flux must be central, roe or a callable')
            face_flux=choices[numerical_flux]
            self.flux_name=numerical_flux
        elif callable(numerical_flux):
            face_flux=numerical_flux
            self.flux_name='custom'
        else:
            raise ValueError('flux must be central, roe or a callable')
        super().__init__((n,)*3 if cells is None else cells,
                         tuple((0.,L) for L in self.mode.lengths),p,device=device,
                         flux=self.equation.flux,numerical_flux=face_flux,
                         boundary=self.equation.pec_boundary,speed=self.equation.speed)
        if self.dim!=3:
            raise ValueError('cavity requires three cell counts')
        self.initial=self.mode.exact(*self.coords,0.)
        self.exact=self.initial
        self.Ebasis,self.Hbasis=self.mode.basis(*self.coords)
        self.Enorm=self.integral((self.Ebasis**2).sum(-1)[...,None]).item()
        self.Hnorm=self.integral((self.Hbasis**2).sum(-1)[...,None]).item()
        if min(self.Enorm,self.Hnorm)<=0:
            raise ValueError('mode is unresolved by these solution points')
        self.history=[]

    def energy(self,U):
        return self.integral(self.equation.energy_density(U)[...,None]).item()

    def energy_rate(self,U):
        """Semi-discrete dW/dt, before any RK time discretization."""
        rhs=self.rhs(U)
        density=self.equation.epsilon*(U[...,:3]*rhs[...,:3]).sum(-1)
        density+=self.equation.mu*(U[...,3:]*rhs[...,3:]).sum(-1)
        return self.integral(density[...,None]).item()

    def broken_divergence(self,U):
        """Element-interior div(D), div(B); excludes face distribution terms."""
        divD=torch.zeros_like(U[...,0])
        divB=torch.zeros_like(U[...,0])
        for axis,h in enumerate(self.h):
            for offset,scale,result in ((0,self.equation.epsilon,divD),(3,self.equation.mu,divB)):
                component=U[...,offset+axis]
                derivative=(component.movedim(self.dim+axis,-1)@self.D.T).movedim(-1,self.dim+axis)
                result+=2*scale/h*derivative
        return divD,divB

    def wall_trace_errors(self,U):
        et,hn=0.,0.
        for axis in range(3):
            A=U.permute(self._permutation(axis))
            for vector,index in ((self.L,0),(self.R,-1)):
                face=self._trace(A,vector)[index]
                tangent=[a for a in range(3) if a!=axis]
                et=max(et,face[...,tangent].abs().max().item())
                hn=max(hn,face[...,3+axis].abs().max().item())
        return et,hn

    def modal_coefficients(self,U):
        e=self.integral((U[...,:3]*self.Ebasis).sum(-1)[...,None]).item()/self.Enorm
        h=self.integral((U[...,3:]*self.Hbasis).sum(-1)[...,None]).item()/self.Hnorm
        return e,h

    def run(self,final_time=None,cfl=.15,method='ssprk43',periods=1.,sample_every=10,monitor=None):
        if final_time is None:
            if not math.isfinite(periods) or periods<0:
                raise ValueError('periods must be finite and nonnegative')
            final_time=periods*self.mode.period
        if not isinstance(sample_every,int) or sample_every<1:
            raise ValueError('sample_every must be a positive integer')
        dt=self.timestep(self.initial,cfl)
        self.history=[]
        W0=self.energy(self.initial)
        calls=0
        def record(U,t):
            e,h=self.modal_coefficients(U)
            W=self.energy(U)
            self.history.append(dict(time=t,energy=W,relative_energy_change=(W-W0)/W0,
                                     electric_mode=e,magnetic_mode=h,
                                     exact_cos=math.cos(self.mode.omega*t),exact_sin=math.sin(self.mode.omega*t)))
        def choose_dt(U,t):
            # integrate calls dt once per accepted state, never at RK substages.
            nonlocal calls
            if calls%sample_every==0:
                record(U,t)
            calls+=1
            return dt
        U,stats=integrate(self.rhs,self.initial,final_time,dt=choose_dt,method=method,
                          validate=self.equation.valid,monitor=monitor)
        record(U,stats['time'])
        self.exact=self.mode.exact(*self.coords,stats['time'])
        divD,divB=self.broken_divergence(U)
        initialD,initialB=self.broken_divergence(self.initial)
        wallE,wallH=self.wall_trace_errors(U)
        l2=self.norm(U-self.exact)
        W=self.energy(U)
        e,h=self.modal_coefficients(U)
        stats.update(cells=self.cells,p=self.p,rk=method,flux=self.flux_name,bounds=self.bounds,
                     mode=self.mode.mode,polarization=self.mode.polarization,
                     epsilon=self.equation.epsilon,mu=self.equation.mu,c=self.equation.c,
                     omega=self.mode.omega,frequency=self.mode.omega/(2*math.pi),period=self.mode.period,
                     periods=stats['time']/self.mode.period,l2=l2.tolist(),
                     field_l2=l2.square().sum().sqrt().item(),
                     electric_l2=l2[:3].square().sum().sqrt().item(),magnetic_l2=l2[3:].square().sum().sqrt().item(),
                     initial_energy=W0,analytic_energy=self.mode.energy(),energy=W,
                     relative_energy_change=(W-W0)/W0,semidiscrete_energy_rate=self.energy_rate(U),
                     broken_divD_l2=self.norm(divD[...,None]).item(),broken_divB_l2=self.norm(divB[...,None]).item(),
                     initial_broken_divD_l2=self.norm(initialD[...,None]).item(),initial_broken_divB_l2=self.norm(initialB[...,None]).item(),
                     interior_wall_Et_max=wallE,interior_wall_Hn_max=wallH,
                     electric_mode=e,magnetic_mode=h,modal_amplitude=math.hypot(e,h))
        # Frequency is estimated from the unwrapped phase of the two modal
        # projections, not copied from the prescribed analytic frequency.
        if len(self.history)>=3:
            phase=[math.atan2(row['magnetic_mode'],row['electric_mode']) for row in self.history]
            unwrapped=[phase[0]]
            for a,b in zip(phase,phase[1:]):
                unwrapped.append(unwrapped[-1]+math.atan2(math.sin(b-a),math.cos(b-a)))
            ts=[row['time'] for row in self.history]
            mean_t,mean_p=sum(ts)/len(ts),sum(unwrapped)/len(ts)
            variance=sum((t-mean_t)**2 for t in ts)
            if variance:
                omega=sum((t-mean_t)*(p-mean_p) for t,p in zip(ts,unwrapped))/variance
                stats['measured_frequency']=omega/(2*math.pi)
                stats['relative_frequency_error']=(omega-self.mode.omega)/self.mode.omega
        return U,stats


def export(path,case,U):
    """VTK structured-grid samples for ParaView, plus modal/energy history."""
    extent=' '.join(f'0 {n*(case.p+1)-1}' for n in case.cells)
    order=(2,5,1,4,0,3,6)  # VTK: x index varies fastest, then y, then z.
    def values(A):
        return A.permute(order).reshape(-1,A.shape[-1]).detach().cpu().tolist()
    def data_array(name,A):
        rows=values(A)
        return f'<DataArray type="Float64" Name="{name}" NumberOfComponents="{A.shape[-1]}" format="ascii">\n'+'\n'.join(' '.join(f'{v:.16e}' for v in row) for row in rows)+'\n</DataArray>'
    xyz=torch.stack(case.coords,-1)
    arrays=[data_array('E',U[...,:3]),data_array('H',U[...,3:]),
            data_array('E_exact',case.exact[...,:3]),data_array('H_exact',case.exact[...,3:]),
            data_array('field_error',(U-case.exact).square().sum(-1).sqrt()[...,None]),
            data_array('energy_density',case.equation.energy_density(U)[...,None])]
    xml='<?xml version="1.0"?>\n<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">\n'
    xml+=f'<StructuredGrid WholeExtent="{extent}"><Piece Extent="{extent}">\n<PointData>\n'
    xml+='\n'.join(arrays)+'\n</PointData><CellData/><Points>\n'+data_array('coordinates',xyz)
    xml+='\n</Points></Piece></StructuredGrid></VTKFile>\n'
    path.with_suffix('.vts').write_text(xml)
    with path.with_name(path.name+'_history.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(case.history[0]))
        writer.writeheader()
        writer.writerows(case.history)


def main():
    parser=argparse.ArgumentParser(description='3-D Maxwell standing mode in a rectangular PEC cavity')
    parser.add_argument('--n',type=int,default=4,help='cells per direction')
    parser.add_argument('--cells',type=int,nargs=3,metavar=('NX','NY','NZ'),help='override --n with three cell counts')
    parser.add_argument('--p',type=int,default=3)
    parser.add_argument('--lengths',type=float,nargs=3,default=(1.,1.3,1.7),metavar=('LX','LY','LZ'))
    parser.add_argument('--mode',type=int,nargs=3,default=(1,1,1),metavar=('M','N','P'))
    parser.add_argument('--epsilon',type=float,default=1.)
    parser.add_argument('--mu',type=float,default=1.)
    parser.add_argument('--amplitude',type=float,default=1.)
    parser.add_argument('--flux',choices=('central','roe'),default='central')
    parser.add_argument('--time',type=float,help='final time; overrides --periods')
    parser.add_argument('--periods',type=float,default=1.,help='number of analytic oscillation periods')
    parser.add_argument('--cfl',type=float,default=.15)
    parser.add_argument('--rk',choices=('rk4','ssprk43'),default='ssprk43')
    parser.add_argument('--sample-every',type=int,default=10)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--convergence',action='store_true')
    parser.add_argument('--output',default='results/maxwell_cavity')
    add_plot_arguments(parser)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    mode=CavityMode(args.lengths,args.mode,args.epsilon,args.mu,args.amplitude)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    for multiplier in ((1,2,4) if args.convergence else (1,)):
        cells=tuple(multiplier*n for n in (args.cells if args.cells else (args.n,)*3))
        case=MaxwellCavityCase(p=args.p,device=args.device,mode=mode,cells=cells,numerical_flux=args.flux)
        end=args.time if args.time is not None else args.periods*mode.period
        plot=PlotRecorder(case,{'Ez':lambda U:U[...,2], '|E|':lambda U:U[...,:3].norm(dim=-1), '|H|':lambda U:U[...,3:].norm(dim=-1)},end,frames=args.frames,fps=args.fps,enabled=not args.no_plot)
        U,stats=case.run(args.time,args.cfl,args.rk,args.periods,args.sample_every,monitor=plot)
        if records and stats['field_l2']>0 and records[-1]['field_l2']>0:
            stats['field_order']=math.log2(records[-1]['field_l2']/stats['field_l2'])
        records.append(stats)
        stem=out/f"cavity_{args.flux}_{'x'.join(str(n) for n in cells)}_p{args.p}"
        export(stem,case,U)
        plot.finish(stem,U,stats["time"])
        torch.save(dict(coords=tuple(x.cpu() for x in case.coords),U=U.cpu(),initial=case.initial.cpu(),
                        exact=case.exact.cpu(),stats=stats,history=case.history),stem.with_suffix('.pt'))
        print(json.dumps(stats),flush=True)
    (out/'metrics.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':
    main()
