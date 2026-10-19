"""Fixed-grid convected acoustic pulse with the existing unmodified split PML."""
from fr.plotting import PlotRecorder,add_plot_arguments
import argparse
import csv
import json
from pathlib import Path
import torch
from fr.equations import make_equation
from .acoustic_pml import AcousticPMLCase


class MeanFlowPMLCase(AcousticPMLCase):
    def __init__(self, n=24, p=3, mean=(.3,.2), flux='roe', thickness=2.,
                 extent=3., absorb=True, reflection=1e-6, order=3, model='acoustics'):
        if model not in ('acoustics','linearized-euler'):
            raise ValueError('model must be acoustics or linearized-euler')
        equation = make_equation('convected-acoustics',mean=mean) if model=='acoustics' else make_equation('linearized-euler',dim=2,mean=mean)
        super().__init__(n,p,mean,flux,thickness,extent,absorb,reflection,order,equation=equation)
        self.model = model
        self.reference_parameters = {'model': model}
        self.history = []

    def run(self, time=4., cfl=.3, dt=None, monitor=None):
        self.history = []
        initial_energy = self.mesh.integral(self.equation.energy(self.layer.physical(self.initial))[...,None]).item()
        def observe(U,t):
            energy = self.mesh.integral(self.equation.energy(U)[...,None]).item()
            self.history.append((t,energy,self.acoustic_state(U)[...,0].abs().max().item()))
            if monitor is not None:
                monitor(U,t)
        U, stats = super().run(time,cfl,dt,monitor=observe)
        stats.update(equation=self.model, frame='fixed laboratory grid; background advection included in PDE',
                     pml_model='Existing SplitPML unchanged, applied experimentally to convected fluxes',
                     initial_energy=initial_energy,
                     peak_energy_over_initial=max(row[1] for row in self.history)/initial_energy,
                     final_energy_over_initial=stats['total_energy']/initial_energy,
                     peak_pressure_over_run=max(row[2] for row in self.history),
                     finite=bool(torch.isfinite(U).all()))
        return U, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--equation',choices=('acoustics','linearized-euler'),default='acoustics')
    parser.add_argument('--n',type=int,default=24)
    parser.add_argument('--p',type=int,default=3)
    parser.add_argument('--u0',type=float,default=.3)
    parser.add_argument('--v0',type=float,default=.2)
    parser.add_argument('--time',type=float,default=4.)
    parser.add_argument('--pml-thickness',type=float,default=2.)
    parser.add_argument('--reflection',type=float,default=1e-6)
    parser.add_argument('--pml-order',type=int,default=3)
    parser.add_argument('--flux',choices=('central','roe','upwind','rusanov','lax-wendroff'),default='roe')
    parser.add_argument('--dt',type=float,default=None)
    parser.add_argument('--validate',action='store_true')
    parser.add_argument('--output',default='results/acoustic_mean_flow_pml')
    add_plot_arguments(parser)
    args = parser.parse_args()
    torch.set_num_threads(1)
    case = MeanFlowPMLCase(args.n,args.p,(args.u0,args.v0),args.flux,
                           thickness=args.pml_thickness,extent=1+args.pml_thickness,
                           reflection=args.reflection,order=args.pml_order,model=args.equation)
    plot=PlotRecorder(case.mesh,{'Pressure perturbation':lambda U:case.acoustic_state(U)[...,0]},args.time,frames=args.frames,fps=args.fps,enabled=not args.no_plot)
    U, stats = case.run(args.time,dt=args.dt,monitor=plot)
    plot.finish(Path(args.output)/'pressure',U,stats['time'])
    if args.validate:
        stats.update(case.compare_reference(U,args.time,stats['dt']))
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    (output/'metrics.json').write_text(json.dumps(stats,indent=2)+'\n')
    with (output/'history.csv').open('w') as f:
        writer=csv.writer(f)
        writer.writerow(('time','total_energy','max_abs_pressure'))
        writer.writerows(case.history)
    x,y = case.mesh.coords
    with (output/'field.csv').open('w') as f:
        writer=csv.writer(f)
        full = args.equation=='linearized-euler'
        writer.writerow(('x','y','pressure','u_perturbation','v_perturbation','u_total','v_total')+
                        (('rho_perturbation',) if full else ()))
        density = U[...,0].flatten().tolist() if full else None
        for index,(xi,yi,(pressure,u,v)) in enumerate(zip(x.flatten().tolist(),y.flatten().tolist(),case.acoustic_state(U).reshape(-1,3).tolist())):
            writer.writerow((xi,yi,pressure,u,v,args.u0+u,args.v0+v)+
                            ((density[index],) if full else ()))
    print(json.dumps(stats,indent=2))


if __name__ == '__main__':
    main()
