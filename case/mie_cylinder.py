"""Body-fitted 2D dielectric cylinder TMz scattering, curved FR-DG and PML."""
from fr.plotting import PlotRecorder,add_plot_arguments
import argparse
import cmath
import csv
import json
import math
from pathlib import Path
import torch
from fr import Mesh2D,MeshFR2D,make_equation,integrate
from fr.pml import SplitPML
from fr.gmsh import generate_mesh
from .mesh_generation import cylinder_quad_geometry
from .cylinder_exact import CylinderSeries,cylindrical_functions


class CylinderScattering:
    def __init__(self,mesh,p=3,*,radius=.5,wavelength=2.,epsilon_r=4.,mu_r=1.,inner=1.5,thickness=1.):
        if min(radius,wavelength,epsilon_r,mu_r,inner,thickness)<=0 or radius>=inner:
            raise ValueError('positive physical parameters and radius<inner required')
        if not any((b.regions==2).any() for b in mesh.blocks):
            raise ValueError('physical surface tag 2 must identify the cylinder')
        self.equation=make_equation('maxwell-tmz')
        self.layer=SplitPML(self.equation,((-inner,inner),)*2,thickness)
        self.radius,self.period,self.omega=radius,wavelength,2*math.pi/wavelength
        self.epsilon_r,self.mu_r,self.inner=epsilon_r,mu_r,inner
        self.delay=radius+.25
        self.ramp_duration=wavelength
        def capacity(region):
            eps=torch.where(region==2,torch.full_like(region,epsilon_r,dtype=torch.float64),1.)
            mu=torch.where(region==2,torch.full_like(region,mu_r,dtype=torch.float64),1.)
            return torch.stack((eps,mu,mu),-1).to(torch.float64)
        self.capacity=capacity
        def interface(m,p,normal):
            um,up=self.layer.physical(m.state),self.layer.physical(p.state)
            cm,cp=capacity(m.region),capacity(p.region)
            star=self.equation.interface_state(um,up,normal,cm[...,0],cp[...,0],cm[...,1],cp[...,1])
            return sum(self.layer.lift(normal[...,a,None]*self.equation.flux(star,a),a) for a in (0,1))
        def boundary(coords,Q,normal,tag,t):
            U=self.layer.physical(Q)
            ghost=self.equation.normal_pec_boundary(coords,U,normal,tag,t)
            return Q+self.layer.pack(ghost-U)
        def source(coords,Q,t,region):
            C=capacity(region)
            split_C=torch.cat((C,C),-1)
            incident_dt=self.incident(coords,t,derivative=True)
            return split_C*self.layer.source(coords,Q,t)+self.layer.pack((1-C)*incident_dt)
        cmax=max(1.,1/math.sqrt(epsilon_r*mu_r))
        self.solver=MeshFR2D(mesh,p,flux=self.layer.flux,numerical_flux=interface,boundary=boundary,
                            source=source,speed=lambda Q,n:torch.full_like(Q[...,0],cmax))
        C=capacity(self.solver.nodal_regions)
        self.split_capacity=torch.cat((C,C),-1)
        self.initial=torch.zeros(self.solver.ndof,6,dtype=torch.float64)
        self.history=[]

    def incident(self,coords,t,derivative=False):
        s=t-coords[0]
        q=((s-self.delay)/self.ramp_duration).clamp(0.,1.)
        ramp=q**3*(10-15*q+6*q*q)
        phase=self.omega*s
        if derivative:
            dr=30*q*q*(1-q)**2/self.ramp_duration
            value=dr*torch.cos(phase)-self.omega*ramp*torch.sin(phase)
        else: value=ramp*torch.cos(phase)
        return torch.stack((value,torch.zeros_like(value),-value),-1)

    def rhs(self,Q,t):
        return self.solver.rhs(Q,t)/self.split_capacity

    def run(self,warmup_periods=6,sample_periods=2,cfl=1.,monitor=None):
        if warmup_periods<1 or sample_periods<1: raise ValueError('positive integer period counts required')
        start=warmup_periods*self.period
        end=(warmup_periods+sample_periods)*self.period
        if start<=self.delay+self.radius+self.ramp_duration:
            raise ValueError('warmup must outlast incident ramp over the cylinder')
        step=1/(1/self.solver.timestep(self.initial,cfl)+self.layer.sigma_max)
        accum=[torch.zeros(self.solver.ndof,3,dtype=torch.complex128) for _ in range(sample_periods)]
        previous=None
        steps=0
        boundaries=[start+i*self.period for i in range(sample_periods+1)]
        def observe(Q,t):
            nonlocal previous,steps
            U=self.layer.physical(Q)
            if not torch.isfinite(Q).all(): raise RuntimeError('nonfinite scattering state')
            if steps%20==0 or t==end:
                self.history.append((t,U.abs().max().item(),U[...,0].abs().max().item()))
            steps+=1
            if t>=start-1e-12:
                phasor=U.to(torch.complex128)*cmath.exp(1j*self.omega*t)
                if previous is not None:
                    pt,PV=previous
                    cycle=min(sample_periods-1,max(0,int((pt-start+1e-10)/self.period)))
                    accum[cycle]+=(t-pt)/self.period*(PV+phasor)
                previous=(t,phasor)
        def dt(Q,t):
            observe(Q,t)
            target=next((b for b in boundaries if b>t+1e-12),end)
            return min(step,target-t) if target>t else step
        Q,stats=integrate(self.rhs,self.initial,end,dt=dt,monitor=monitor)
        observe(Q,end)
        phasor=sum(accum)/sample_periods
        denominator=self.solver.norm(accum[-1]).norm().item()
        change=self.solver.norm(accum[-1]-accum[-2]).norm().item()/max(denominator,1e-30) if sample_periods>1 else None
        circle=self.solver.face_coordinates[self.solver.tags==2]
        circle_error=(circle.norm(dim=-1)-self.radius).abs().max().item() if circle.numel() else None
        cylinder_area=sum((b['areas']*(b['region']==2)).sum().item() for b in self.solver.blocks)
        stats.update(circle_max_radial_error=circle_error,cylinder_area=cylinder_area,
                     elements=self.solver.mesh.nelements,ndof=self.solver.ndof,p=self.solver.p,
                     geometry_order=max(b.order for b in self.solver.mesh.blocks),minimum_jacobian=self.solver.minimum_jacobian,
                     radius=self.radius,wavelength=self.period,epsilon_r=self.epsilon_r,mu_r=self.mu_r,
                     warmup_periods=warmup_periods,sample_periods=sample_periods,dt=step,
                     last_cycle_relative_change=change,finite=bool(torch.isfinite(Q).all()))
        return Q,phasor,stats

    def validate(self,phasor):
        reference=CylinderSeries(self.radius,self.period,self.epsilon_r,self.mu_r)
        xs,ys,values,weights=[],[],[],[]
        error=reference_norm=0.
        maximum=0.
        for coords,V,w,region in self.solver.integration(phasor):
            x,y=coords
            r=torch.hypot(x,y)
            mask=(region==1)&(r>1.2*self.radius)&(r<.9*self.inner)
            if not mask.any(): continue
            x,y,value,weight=x[mask],y[mask],V[...,0][mask],w[mask]
            exact=reference.scattered(x,y)
            error+=(weight*(value-exact).abs().square()).sum().item()
            reference_norm+=(weight*exact.abs().square()).sum().item()
            maximum=max(maximum,(value-exact).abs().max().item())
            xs.append(x);ys.append(y);values.append(value);weights.append(weight)
        if not xs: raise ValueError('mesh has no exterior quadrature points in validation annulus')
        x,y,V,w=map(torch.cat,(xs,ys,values,weights))
        modes=range(-6,7)
        _,H,_,_=cylindrical_functions(reference.k*torch.hypot(x,y),6)
        angle=torch.atan2(y,x)
        A=torch.stack([H[abs(m)]*torch.exp(1j*m*angle) for m in modes],-1)
        weighted=A*w.sqrt()[:,None]
        scale=weighted.norm(dim=0)
        coefficients=torch.linalg.lstsq(weighted/scale,(V*w.sqrt())[:,None]).solution[:,0]/scale
        theta=torch.linspace(-math.pi,math.pi,361,dtype=torch.float64)
        far=sum(c*cmath.exp(-.5j*math.pi*abs(m))*torch.exp(1j*m*theta) for c,m in zip(coefficients,modes))
        numeric=2/(math.pi*reference.k)*far.abs().square()
        exact=reference.angular_width(theta)
        numeric_width=4/reference.k*coefficients.abs().square().sum().item()
        exact_width=reference.total_width()
        metrics=dict(scattered_Ez_relative_L2=math.sqrt(error/max(reference_norm,1e-30)),
                     scattered_Ez_max_error=maximum,analytic_total_scattering_width=exact_width,
                     fitted_total_scattering_width=numeric_width,
                     total_width_relative_error=abs(numeric_width-exact_width)/max(exact_width,1e-30),
                     far_field_relative_L2=((numeric-exact).norm()/exact.norm().clamp(min=1e-30)).item(),
                     reference='Circular-cylinder Fourier-Bessel series; exp(-i omega t), outgoing Hankel H1',
                     validation_region=f'{1.2*self.radius}<r<{.9*self.inner}, exterior only')
        return metrics,(theta,numeric,exact)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh',help='Gmsh mesh with surface tags air=1, cylinder=2')
    p.add_argument('--p',type=int,default=3)
    p.add_argument('--geometry-order',type=int,default=3)
    p.add_argument('--angular',type=int,default=4)
    p.add_argument('--radial',type=int,default=8)
    p.add_argument('--radius',type=float,default=.5)
    p.add_argument('--wavelength',type=float,default=2.)
    p.add_argument('--epsilon-r',type=float,default=4.)
    p.add_argument('--mu-r',type=float,default=1.)
    p.add_argument('--warmup-periods',type=int,default=6)
    p.add_argument('--sample-periods',type=int,default=2)
    p.add_argument('--cfl',type=float,default=1.)
    p.add_argument('--validate',action='store_true')
    p.add_argument('--output',default='results/mie_cylinder')
    add_plot_arguments(p)
    a=p.parse_args()
    torch.set_num_threads(1)
    output=Path(a.output)
    output.mkdir(parents=True,exist_ok=True)
    if a.mesh: mesh=Mesh2D.read(a.mesh)
    else:
        geo=output/'cylinder.geo'
        geo.write_text(cylinder_quad_geometry(a.radius,2.5,a.angular,a.radial))
        mesh=Mesh2D.read(generate_mesh(geo,output/'cylinder.msh',order=a.geometry_order))
    case=CylinderScattering(mesh,a.p,radius=a.radius,wavelength=a.wavelength,epsilon_r=a.epsilon_r,mu_r=a.mu_r)
    case.solver.write_svg(output/'mesh.svg')
    end=(a.warmup_periods+a.sample_periods)*case.period
    plot=PlotRecorder(case.solver,{'Scattered Ez':lambda U:case.layer.physical(U)[...,0]},end,frames=a.frames,fps=a.fps,enabled=not a.no_plot)
    Q,phasor,stats=case.run(a.warmup_periods,a.sample_periods,a.cfl,monitor=plot)
    plot.finish(output/'scattered_Ez',Q,stats['time'])
    amplitude=PlotRecorder(case.solver,{'Scattered Ez amplitude':lambda U:U[...,0].abs()},end,frames=0,enabled=not a.no_plot)
    amplitude.finish(output/'phasor_amplitude',phasor,end)
    if a.validate:
        metrics,pattern=case.validate(phasor)
        stats.update(metrics)
        with (output/'scattering_width.csv').open('w') as f:
            writer=csv.writer(f)
            writer.writerow(('theta','numerical_width','analytic_width'))
            writer.writerows(zip(*(v.tolist() for v in pattern)))
    (output/'metrics.json').write_text(json.dumps(stats,indent=2)+'\n')
    with (output/'history.csv').open('w') as f:
        writer=csv.writer(f); writer.writerow(('time','max_abs_scattered_field','max_abs_scattered_Ez'));writer.writerows(case.history)
    with (output/'field.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(('x','y','region','scattered_Ez_real','scattered_Ez_imag'))
        writer.writerows(zip(*(v.tolist() for v in case.solver.coords),case.solver.nodal_regions.tolist(),phasor[:,0].real.tolist(),phasor[:,0].imag.tolist()))
    torch.save({'scattered_state':case.layer.physical(Q),'phasor':phasor,'coords':case.solver.coords,'metrics':stats},output/'solution.pt')
    print(json.dumps(stats,indent=2))


if __name__=='__main__': main()
