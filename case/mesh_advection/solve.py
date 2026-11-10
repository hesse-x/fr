"""Advection on irregular, curved, mixed triangle/quad meshes."""
from fr.plotting import PlotRecorder,add_plot_arguments
import argparse
import csv
import json
import math
from pathlib import Path
import torch
from fr.mesh import Mesh2D
from fr.unstructured import MeshFR2D,normal_rusanov_flux
from fr.time import integrate


def run(n=4,p=3,kind='mixed',time=.1,mesh=None,geometry_order=3,plot_stem=None,frames=60,fps=12):
    velocity=(1.,.3)
    if mesh is None:
        mesh=Mesh2D.rectangle(n,kind=kind,jitter=.18,seed=7).mapped(
            lambda x,y:(x+.03*torch.sin(math.pi*x)*torch.sin(math.pi*y),
                        y+.02*torch.sin(2*math.pi*x)*torch.sin(math.pi*y)),geometry_order)
    def exact(coords,t):
        x,y=coords
        return (1+.1*torch.sin(2*math.pi*(x-velocity[0]*t))*torch.cos(2*math.pi*(y-velocity[1]*t)))[...,None]
    speed=lambda U,normal:(normal[...,0]*velocity[0]+normal[...,1]*velocity[1]).abs()
    solver=MeshFR2D(mesh,p,flux=lambda U,a:velocity[a]*U,numerical_flux=normal_rusanov_flux(speed),
                    speed=speed,boundary=lambda coords,U,normal,tag,t:exact(coords,t))
    initial=exact(solver.coords,0.)
    plot=PlotRecorder(solver,{'Scalar':lambda U:U[...,0]},time,frames=frames,fps=fps,enabled=plot_stem is not None)
    U,stats=integrate(solver.rhs,initial,time,dt=solver.timestep(initial,.6),monitor=plot)
    if plot_stem is not None: plot.finish(plot_stem,U,stats['time'])
    stats.update(elements=mesh.nelements,ndof=solver.ndof,p=p,
                 geometry_order=max(b.order for b in mesh.blocks),volume=solver.volume,
                 minimum_jacobian=solver.minimum_jacobian,L2=solver.norm(U-exact(solver.coords,time)).item())
    return solver,U,stats


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh',help='ASCII .msh 2.2 or 4.1; overrides generated rectangle')
    parser.add_argument('--kind',choices=('tri','quad','mixed'),default='mixed')
    parser.add_argument('--n',type=int,default=4)
    parser.add_argument('--p',type=int,default=3)
    parser.add_argument('--geometry-order',type=int,default=3)
    parser.add_argument('--time',type=float,default=.1)
    parser.add_argument('--convergence',action='store_true')
    parser.add_argument('--output',default='results/mesh_advection')
    add_plot_arguments(parser)
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.mesh and args.convergence: parser.error('use a sequence of generated meshes for --convergence')
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    mesh=Mesh2D.read(args.mesh) if args.mesh else None
    rows=[]
    for n in ([args.n,2*args.n,4*args.n] if args.convergence else [args.n]):
        solver,U,stats=run(n,args.p,args.kind,args.time,mesh,args.geometry_order,
                           plot_stem=None if args.no_plot else output/f'scalar_n{n}_p{args.p}',frames=args.frames,fps=args.fps)
        if rows: stats['observed_order']=math.log2(rows[-1]['L2']/stats['L2'])
        rows.append(stats)
        print(json.dumps(stats,indent=2))
    solver.write_svg(output/'mesh.svg')
    with (output/'field.csv').open('w') as f:
        writer=csv.writer(f)
        writer.writerow(('x','y','scalar'))
        writer.writerows(zip(*(x.tolist() for x in solver.coords),U[:,0].tolist()))
    (output/'metrics.json').write_text(json.dumps(rows if args.convergence else stats,indent=2)+'\n')


if __name__=='__main__': main()
