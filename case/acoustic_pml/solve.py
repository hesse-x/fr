"""Initial cylindrical acoustic pulse; uniform oblique mean flow, moving PML."""
from fr.plotting import PlotRecorder,add_plot_arguments
import argparse
import csv
import json
import math
from pathlib import Path
import torch
from fr import CartesianFR, integrate
from fr.equations import make_equation
from fr.boundaries import acoustic_wall
from fr.fluxes import make_numerical_flux
from fr.pml import SplitPML


class AcousticPMLCase:
    def __init__(self, n=16, p=3, mean=(.3,.2), flux='roe', thickness=1., extent=2., absorb=True, reflection=1e-6, order=3, equation=None):
        if len(mean) != 2 or not all(math.isfinite(v) for v in mean):
            raise ValueError('two finite mean velocity components required')
        self.mean = tuple(mean)
        self.equation = make_equation('acoustics',dim=2) if equation is None else equation
        self.layer = SplitPML(self.equation, ((-1.,1.),)*2, thickness, reflection=reflection, order=order)
        self.absorb = absorb
        self.flux_name = flux
        self.extent = extent
        self.reference_parameters = {}
        self.face_flux = make_numerical_flux(flux,self.equation)
        self.mesh = CartesianFR((n,n), ((-extent,extent),)*2, p,
            flux=self.layer.flux, numerical_flux=self.layer.numerical_flux(self.face_flux),
            boundary=self.layer.boundary(acoustic_wall),
            source=self.layer.source if absorb else None,
            speed=lambda Q,a: self.equation.speed(self.layer.physical(Q),a))
        x,y = self.mesh.coords
        U = torch.zeros(*self.mesh.shape,3,dtype=torch.float64)
        U[...,0] = torch.exp(-(x*x+y*y)/(2*.15**2))
        if hasattr(self.equation,'from_acoustic'):
            U = self.equation.from_acoustic(U)
        self.initial = self.layer.pack(U)

    def acoustic_state(self, U):
        return self.equation.acoustic_state(U) if hasattr(self.equation,'acoustic_state') else U

    def timestep(self, cfl=.3):
        return 1/(1/self.mesh.timestep(self.initial,cfl)+(self.layer.sigma_max if self.absorb else 0))

    def run(self, time=3., cfl=.3, dt=None, monitor=None):
        mesh = self.mesh
        def prepare(dt,t):
            if hasattr(self.face_flux,'prepare_step'):
                self.face_flux.prepare_step(dt,mesh.h)
        if dt is None:
            dt = self.timestep(cfl)
        def step_dt(Q,t):
            if monitor is not None:
                monitor(self.layer.physical(Q),t)
            return dt
        Q, stats = integrate(mesh.rhs,self.initial,time,dt=step_dt,prepare_step=prepare,
                             validate=lambda Q: bool(torch.isfinite(Q).all()))
        if monitor is not None:
            monitor(self.layer.physical(Q),time)
        U = self.layer.physical(Q)
        core = (mesh.coords[0].abs()<1)&(mesh.coords[1].abs()<1)
        energy = self.equation.energy(U)
        stats.update(dt=dt, core_energy=mesh.integral((energy*core)[...,None]).item(),
                     total_energy=mesh.integral(energy[...,None]).item(), mean_velocity=self.mean,
                     frame='translating: x=xi+u0*t, y=eta+v0*t; PML moves with frame')
        return U, stats


    def compare_reference(self, U, time, dt):
        """Same spatial mesh and nominal RK step; core error is not reflectance."""
        n = self.mesh.cells[0]
        h = self.mesh.h[0]
        physical_initial = self.layer.physical(self.initial)
        max_speed = max(self.equation.speed(physical_initial,a).max().item() for a in range(2))
        extra = math.ceil(max_speed*time/h)+1
        ref = type(self)(n+2*extra,self.mesh.p,self.mean,self.flux_name,
                              extent=self.extent+extra*h,absorb=False,**self.reference_parameters)
        V, _ = ref.run(time,dt=dt)
        V = V[extra:extra+n,extra:extra+n]
        wall = type(self)(n,self.mesh.p,self.mean,self.flux_name,
                               extent=self.extent,absorb=False,**self.reference_parameters)
        W, wall_stats = wall.run(time,dt=dt)
        core = (self.mesh.coords[0].abs()<1)&(self.mesh.coords[1].abs()<1)
        def error(A):
            return self.mesh.integral(((A-V).square().sum(-1)*core)[...,None]).sqrt().item()
        initial_U = self.layer.physical(self.initial)
        initial_energy = self.mesh.integral(self.equation.energy(initial_U)[...,None]).item()
        error_energy = self.mesh.integral((self.equation.energy(U-V)*core)[...,None]).item()
        pressure_error = self.acoustic_state(U-V)[...,0][core].abs().max().item()
        return dict(pml_reference_L2=error(U), wall_reference_L2=error(W),
                    error_ratio=error(U)/max(error(W),1e-30), wall_core_energy=wall_stats['core_energy'],
                    core_pressure_max_error=pressure_error,
                    core_pressure_error_over_initial_peak=pressure_error/self.acoustic_state(initial_U)[...,0].abs().max().item(),
                    core_error_energy_over_initial=error_energy/initial_energy,
                    validation_note='Final-time core difference from enlarged-domain solution; not an incident/reflected-wave coefficient.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n',type=int,default=16)
    parser.add_argument('--p',type=int,default=3)
    parser.add_argument('--time',type=float,default=3.)
    parser.add_argument('--u0',type=float,default=.3)
    parser.add_argument('--v0',type=float,default=.2)
    parser.add_argument('--flux',choices=('central','roe','upwind','rusanov','lax-wendroff'),default='roe')
    parser.add_argument('--pml-thickness',type=float,default=1.,help='layer width; outer half-width = 1 + thickness')
    parser.add_argument('--reflection',type=float,default=1e-6,help='profile design parameter, not measured reflectance')
    parser.add_argument('--pml-order',type=int,default=3)
    parser.add_argument('--dt',type=float,default=None)
    parser.add_argument('--validate',action='store_true')
    parser.add_argument('--output',default='results/acoustic_pml')
    add_plot_arguments(parser)
    args = parser.parse_args()
    torch.set_num_threads(1)
    case = AcousticPMLCase(args.n,args.p,(args.u0,args.v0),args.flux,
                           thickness=args.pml_thickness,extent=1+args.pml_thickness,
                           reflection=args.reflection,order=args.pml_order)
    plot=PlotRecorder(case.mesh,{'Pressure perturbation':lambda U:case.acoustic_state(U)[...,0]},args.time,frames=args.frames,fps=args.fps,enabled=not args.no_plot)
    U, stats = case.run(args.time,dt=args.dt,monitor=plot)
    plot.finish(Path(args.output)/'pressure',U,stats['time'])
    if args.validate:
        stats.update(case.compare_reference(U,args.time,stats['dt']))
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    (output/'metrics.json').write_text(json.dumps(stats,indent=2)+'\n')
    x,y = case.mesh.coords
    with (output/'field.csv').open('w') as f:
        writer=csv.writer(f)
        writer.writerow(('xi','eta','x','y','pressure','u_perturbation','v_perturbation','u_total','v_total'))
        for xi,eta,values in zip(x.flatten().tolist(),y.flatten().tolist(),U.reshape(-1,3).tolist()):
            pressure,u,v=values
            writer.writerow((xi,eta,xi+args.u0*args.time,eta+args.v0*args.time,pressure,u,v,u+args.u0,v+args.v0))
    print(json.dumps(stats,indent=2))


if __name__ == '__main__':
    main()
